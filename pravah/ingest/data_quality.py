"""Data-quality checks that apply to any table, process log or not."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import DATA_QUALITY_WEIGHTS, MIN_CASES_FOR_RATE, Z_95
from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import RateReport
from pravah.ingest.proportion import wilson_interval
from pravah.stats.outliers import modified_z_scores

_FORMULA = (
    f"n = {Z_95}² × {0.5} × {0.5} / {0.10}² = {MIN_CASES_FOR_RATE}. "
    "That is the worst-case size for a ±10 point interval on a rate."
)


def _report(
    name: str,
    bad_count: int,
    total: int,
    *,
    unit: str,
    problem: str,
    applicable: bool = True,
    reason: str = "",
) -> RateReport:
    if not applicable:
        return RateReport(
            name=name,
            bad_count=0,
            total=0,
            bad_fraction=0.0,
            score=0.0,
            lower=0.0,
            upper=1.0,
            required_total=MIN_CASES_FOR_RATE,
            sufficient=False,
            unit=unit,
            detail=f"Not applicable. {reason}",
            applicable=False,
            rough=False,
            technical=reason,
        )
    if total <= 0:
        return _report(name, 0, 0, unit=unit, problem=problem, applicable=False, reason=f"There were no {unit} to count.")
    _point, lower, upper = wilson_interval(bad_count, total)
    fraction = bad_count / total
    return RateReport(
        name=name,
        bad_count=bad_count,
        total=total,
        bad_fraction=fraction,
        score=100.0 * (1.0 - fraction),
        lower=lower,
        upper=upper,
        required_total=MIN_CASES_FOR_RATE,
        sufficient=total >= MIN_CASES_FOR_RATE,
        unit=unit,
        detail=f"{bad_count} of {total} {unit} ({fraction:.0%}) {problem}.",
        applicable=True,
        rough=total < MIN_CASES_FOR_RATE,
        technical=(
            f"Wilson 95% interval {lower:.0%} to {upper:.0%}. {_FORMULA}"
        ),
    )


def _mixed_type_cells(frame: pd.DataFrame) -> tuple[int, int]:
    bad = 0
    total = 0
    for name in frame.columns:
        series = frame[name]
        present = series.dropna()
        total += int(len(present))
        if present.empty or pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        text = present.astype(str).str.strip()
        parsed = parse_datetimes(text)
        date_rate = float(parsed.notna().mean())
        numeric_rate = float(pd.to_numeric(text, errors="coerce").notna().mean())
        if date_rate >= 0.8:
            bad += int(parsed.isna().sum())
        elif numeric_rate >= 0.8:
            bad += int(pd.to_numeric(text, errors="coerce").isna().sum())
    return bad, total


def _outlier_cells(frame: pd.DataFrame) -> tuple[int, int, bool]:
    flagged = 0
    total = 0
    scored = False
    for name in frame.columns:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        values = [float(value) for value in series.dropna().tolist()]
        if len(values) < 8:
            continue
        scores = modified_z_scores(values)
        if not scores:
            continue
        scored = True
        total += len(values)
        flagged += sum(abs(score) > 3.5 for score in scores)
    return flagged, total, scored


def score_data_quality(frame: pd.DataFrame) -> tuple[dict[str, RateReport], float]:
    """Score missing cells, duplicate rows, mixed types, and numeric outliers.

    WHAT: Four rates on the uploaded table. Missing cells are empty cells over every cell. Duplicate rows are exact copies beyond the first. Mixed types are cells that disagree with a column that is otherwise a date or a number. Outliers use the modified z-score and are skipped when no numeric column has 8 values. The score renormalises over the checks that ran.
    WHY: A price list is not a process log. Punishing it for a missing timestamp says the table is broken when the check never applied.
    WHY NOT: Reusing the process weights would put "events go backwards" on a table that has no events.
    ASSUMPTIONS: An exact duplicate row is the same business record typed twice. A column that is 80% dates is a date column.
    COMPLEXITY: O(n m) time for ``n`` rows and ``m`` columns, plus the outlier sorts.
    FAILS WHEN: Two different people share every column, so the duplicate check flags a real coincidence. A free-text column with a few numbers in it is not marked mixed, because it never reaches the 80% bar.
    """
    cells = int(frame.shape[0] * frame.shape[1])
    missing = int(frame.isna().sum().sum()) if cells else 0
    duplicates = int(frame.duplicated().sum()) if len(frame) else 0
    mixed_bad, mixed_total = _mixed_type_cells(frame)
    outlier_bad, outlier_total, outlier_ran = _outlier_cells(frame)
    components = {
        "missing_cells": _report(
            "missing_cells",
            missing,
            cells,
            unit="cells",
            problem="are empty",
        ),
        "duplicate_rows": _report(
            "duplicate_rows",
            duplicates,
            len(frame),
            unit="rows",
            problem="are exact copies of an earlier row",
        ),
        "mixed_types": _report(
            "mixed_types",
            mixed_bad,
            mixed_total,
            unit="filled cells",
            problem="do not match the type of their column",
        ),
        "outliers": _report(
            "outliers",
            outlier_bad,
            outlier_total if outlier_ran else 0,
            unit="numeric values",
            problem="sit far from the middle of their column",
            applicable=outlier_ran,
            reason="No numeric column has 8 values, so a modified z-score would not be stable.",
        ),
    }
    used = {name: weight for name, weight in DATA_QUALITY_WEIGHTS.items() if components[name].applicable}
    total_weight = sum(used.values()) or 1.0
    score = sum(used[name] * components[name].score for name in used) / total_weight
    return components, score
