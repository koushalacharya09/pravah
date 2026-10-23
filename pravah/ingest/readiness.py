"""Which later analyses this file can support. Rules, not a model."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import (
    EVENTS_PER_FEATURE,
    EVENTS_PER_FEATURE_COMFORTABLE,
    MIN_CASES_FOR_RATE,
    Z_95,
)
from pravah.ingest.models import ReconstructionResult, SchemaResult, ShapeResult, ValidationResult


def _row(name: str, status: str, reason: str) -> dict[str, str]:
    return {"analysis": name, "status": status, "reason": reason}


def _weeks(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return None
    span = parsed.max() - parsed.min()
    return float(span.total_seconds()) / 86400.0 / 7.0


def readiness_matrix(
    frame: pd.DataFrame,
    *,
    shape: ShapeResult,
    schema: SchemaResult,
    reconstruction: ReconstructionResult | None,
    validation: ValidationResult,
) -> list[dict[str, str]]:
    """Say which later analyses this upload can support.

    WHAT: A fixed table of analyses. Process map is possible when a stage order of at least two steps exists, or when an event log has a case, an activity, and a time. Survival is possible when case durations vary. Bottleneck rates use cases against ``n = z^2 * 0.5 * 0.5 / 0.10^2``. Prediction uses 10 events per feature as the floor and 20 as the comfortable size, counted in cases because the outcome is one per case. Forecast needs at least 8 weeks, and is comfortable at 16. Causes is a caution to choose a treatment when those three roles exist, and No when they do not.
    WHY: A founder should see the refusal before a later page draws a chart on data that cannot support it. The rules are the same ones the later phases will have to obey.
    WHY NOT: Letting every page run and then explaining the failure wastes the upload. A learned recommender would need labelled "this analysis worked" outcomes this project does not have.
    ASSUMPTIONS: The shape, schema, and validation already computed for this file are the inputs. "Open cases" are approximated by cases that do not all have the same number of events.
    COMPLEXITY: O(n) time to measure the time span and the case lengths. Space O(1) beyond the frame.
    FAILS WHEN: Every case happens to have the same number of events even though some are still open. Survival is then marked unavailable. A forecast on a process that is seasonal inside one week is also marked ready once 16 weeks exist, which is only a minimum.
    """
    rows: list[dict[str, str]] = []
    multi_event = False
    varied_length = False
    if schema.case_id and schema.case_id in frame.columns:
        lengths = frame[schema.case_id].dropna().value_counts()
        if not lengths.empty:
            multi_event = bool((lengths >= 2).mean() >= 0.5)
            varied_length = int(lengths.min()) != int(lengths.max())
    cases = validation.n_cases or 0
    features = _feature_count(frame, schema)
    stage_count = len(reconstruction.order) if reconstruction is not None else 0
    process_ok = stage_count >= 2 or (
        shape.shape == "event_log" and schema.case_id and schema.activity and schema.timestamp and multi_event
    )
    if process_ok:
        rows.append(_row("Process map", "yes", f"{max(stage_count, 2)} ordered steps can be read from this file."))
    else:
        rows.append(_row("Process map", "no", "There is no repeating case with an ordered activity or stage date."))

    rate_rule = f"n = {Z_95}² × 0.5 × 0.5 / 0.10² = {MIN_CASES_FOR_RATE} cases"
    if process_ok and cases >= MIN_CASES_FOR_RATE:
        rows.append(_row("Bottlenecks", "yes", f"{cases} cases meet {rate_rule}."))
    elif process_ok:
        rows.append(
            _row(
                "Bottlenecks",
                "caution",
                f"Only {cases} cases. A bottleneck rate needs {rate_rule}.",
            )
        )
    else:
        rows.append(_row("Bottlenecks", "no", "Bottlenecks need a process map first."))

    if schema.case_id and schema.timestamp and varied_length:
        rows.append(
            _row(
                "Survival",
                "yes",
                "Cases have different lengths, so unfinished cases can be treated as censored rather than as failures.",
            )
        )
    else:
        rows.append(
            _row(
                "Survival",
                "no",
                "Survival needs cases of different lengths. Here every case looks equally finished, so censoring is not visible.",
            )
        )

    floor = features * EVENTS_PER_FEATURE
    comfort = features * EVENTS_PER_FEATURE_COMFORTABLE
    prediction_rule = (
        f"{features} feature(s) × {EVENTS_PER_FEATURE} events per feature = {floor} cases, "
        f"and {EVENTS_PER_FEATURE_COMFORTABLE} per feature ({comfort} cases) is comfortable"
    )
    if schema.case_id and schema.activity and cases >= comfort:
        rows.append(_row("Prediction", "yes", f"{cases} cases meet the rule: {prediction_rule}."))
    elif schema.case_id and schema.activity and cases >= floor:
        rows.append(_row("Prediction", "caution", f"{cases} cases. {prediction_rule}."))
    elif schema.case_id and schema.activity:
        rows.append(_row("Prediction", "no", f"Only {cases} cases. {prediction_rule}."))
    else:
        rows.append(
            _row(
                "Prediction",
                "no",
                "Prediction needs a case and an activity. The size rule is 10 to 20 events per feature.",
            )
        )

    weeks = _weeks(frame[schema.timestamp]) if schema.timestamp and schema.timestamp in frame.columns else None
    if weeks is None:
        rows.append(_row("Forecast", "no", "There is no event time to build a weekly history."))
    elif weeks < 8:
        rows.append(_row("Forecast", "no", f"The file covers only {weeks:.0f} week(s). A weekly forecast needs 8."))
    elif weeks < 16:
        rows.append(_row("Forecast", "caution", f"The file covers {weeks:.0f} weeks. That is a short history."))
    else:
        rows.append(_row("Forecast", "yes", f"The file covers {weeks:.0f} weeks of arrivals."))

    if schema.case_id and schema.activity and schema.timestamp:
        rows.append(_row("Causes", "caution", "Choose a treatment to unlock."))
    else:
        rows.append(
            _row(
                "Causes",
                "no",
                "A cause needs a column that marks who received an action. This file has not named one, so only associations can be read.",
            )
        )
    return rows


def _feature_count(frame: pd.DataFrame, schema: SchemaResult) -> int:
    roles = {schema.case_id, schema.activity, schema.timestamp}
    count = 0
    for name in frame.columns:
        if name in roles:
            continue
        series = frame[name].dropna()
        if series.empty:
            continue
        distinct = int(series.nunique())
        if distinct <= 1:
            continue
        if distinct / max(len(frame), 1) >= 0.9 and not pd.api.types.is_numeric_dtype(series):
            continue
        count += 1
    return max(count, 1)


def analysis_verdict(quality_score: float, sufficient: bool, *, process_ready: bool = True) -> dict[str, str]:
    """Safe-to-analyse band from data quality and the Wald case floor."""
    if quality_score >= 80 and sufficient:
        label, band = "Yes", "yes"
        reason = "The sheet is clean and large enough for a first analysis."
    elif quality_score >= 80 and not process_ready:
        label, band = "Caution", "caution"
        reason = "The table is clean. It is small, and it is not a process log, so process checks were not applied."
    elif quality_score >= 50:
        label, band = "Caution", "caution"
        reason = (
            "The sheet can be read, but the case count is below the ±10 point formula. Treat the numbers as a preview."
            if not sufficient
            else "The sheet is large enough, but the data-quality score is only middling."
        )
    else:
        label, band = "No", "no"
        reason = "Too much of the table is missing, duplicated, or the wrong type."
    return {"label": label, "band": band, "reason": reason}
