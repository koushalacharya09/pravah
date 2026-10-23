"""Health score for an event table. Weights are documented, not fitted."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import MIN_CASES_FOR_RATE, MIN_ROWS_FOR_INFERENCE, QUALITY_WEIGHTS, Z_95
from pravah.ingest.data_quality import score_data_quality
from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import RateReport, SchemaResult, ValidationResult
from pravah.ingest.proportion import wilson_interval

def _rate(
    name: str,
    bad_count: int,
    total: int,
    *,
    unit: str,
    problem: str,
) -> RateReport:
    if total <= 0:
        return RateReport(
            name=name,
            bad_count=0,
            total=0,
            bad_fraction=1.0,
            score=0.0,
            lower=0.0,
            upper=1.0,
            required_total=MIN_ROWS_FOR_INFERENCE,
            sufficient=False,
            unit=unit,
            detail=f"Not applicable. There were no {unit} to count.",
            applicable=False,
            technical=f"There were no {unit} to count.",
        )
    _point, lower, upper = wilson_interval(bad_count, total)
    bad_fraction = bad_count / total
    score = 100.0 * (1.0 - bad_fraction)
    sufficient = total >= MIN_ROWS_FOR_INFERENCE
    detail = f"{bad_count} of {total} {unit} ({bad_fraction:.0%}) {problem}."
    return RateReport(
        name=name,
        bad_count=bad_count,
        total=total,
        bad_fraction=bad_fraction,
        score=score,
        lower=lower,
        upper=upper,
        required_total=MIN_ROWS_FOR_INFERENCE,
        sufficient=sufficient,
        unit=unit,
        detail=detail,
        rough=not sufficient,
        technical=(
            f"Wilson 95% interval {lower:.0%} to {upper:.0%}. "
            f"n = {Z_95}² × 0.5 × 0.5 / 0.10² = {MIN_CASES_FOR_RATE}."
        ),
    )


def _missing_mask(frame: pd.DataFrame, schema: SchemaResult) -> pd.Series:
    required = [name for name in (schema.case_id, schema.activity, schema.timestamp) if name]
    if len(required) < 3:
        return pd.Series(True, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for name in required:
        mask = mask | frame[name].isna()
    return mask


def _duplicate_count(frame: pd.DataFrame, schema: SchemaResult) -> tuple[int, int]:
    keys = [name for name in (schema.case_id, schema.activity, schema.timestamp) if name]
    if len(keys) < 3:
        return len(frame), len(frame)
    complete = frame.loc[~_missing_mask(frame, schema), keys]
    if complete.empty:
        return 0, len(frame)
    extra = int(len(complete) - complete.drop_duplicates().shape[0])
    return extra, len(frame)


def _inversion_count(frame: pd.DataFrame, schema: SchemaResult) -> tuple[int, int]:
    if schema.case_id is None or schema.timestamp is None:
        return 0, 0
    times = parse_datetimes(frame[schema.timestamp])
    work = pd.DataFrame({schema.case_id: frame[schema.case_id], "_t": times})
    work = work.dropna()
    inversions = 0
    pairs = 0
    for _case, group in work.groupby(schema.case_id, sort=False):
        stamps = group["_t"].tolist()
        for earlier, later in zip(stamps, stamps[1:]):
            pairs += 1
            if later < earlier:
                inversions += 1
    return inversions, pairs


def _single_event_count(frame: pd.DataFrame, schema: SchemaResult) -> tuple[int, int]:
    if schema.case_id is None:
        return 0, 0
    counts = frame[schema.case_id].dropna().value_counts()
    if counts.empty:
        return 0, 0
    singles = int((counts == 1).sum())
    return singles, int(len(counts))


def _unparseable_count(frame: pd.DataFrame, schema: SchemaResult) -> tuple[int, int]:
    if schema.timestamp is None:
        return 0, 0
    raw = frame[schema.timestamp]
    present = raw.dropna()
    if present.empty:
        return 0, 0
    parsed = parse_datetimes(present)
    bad = int(parsed.isna().sum())
    return bad, int(len(present))


def combine_quality(components: dict[str, RateReport], weights: dict[str, float] | None = None) -> float | None:
    """Weighted mean of the checks that apply. ``None`` when none of them do."""
    used = weights or QUALITY_WEIGHTS
    active = {name: weight for name, weight in used.items() if components[name].applicable}
    total_weight = sum(active.values())
    if total_weight <= 0:
        return None
    return sum(active[name] * components[name].score for name in active) / total_weight


def _not_applicable(name: str, reason: str) -> RateReport:
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
        unit="rows",
        detail=f"Not applicable. {reason}",
        applicable=False,
        technical=reason,
    )


def validate_event_log(
    frame: pd.DataFrame,
    schema: SchemaResult,
    *,
    weights: dict[str, float] | None = None,
    source: pd.DataFrame | None = None,
) -> ValidationResult:
    """Score an event table from 0 to 100.

    WHAT: Five rates, each turned into a 0-100 component by ``100 * (1 - bad_fraction)``.
    The health score is the weighted mean with weights missing 25, unreadable
    times 25, backwards times 20, duplicate events 15, single-event cases 15.
    A missing required role (case, activity, or time) makes the missing-data
    rate 1. Duplicate extras are complete-key rows beyond the first copy of
    ``(case, activity, timestamp)``. Backwards time counts consecutive pairs
    inside a case, in the current row order, whose timestamp decreases. Equal
    timestamps are not backwards. Single-event cases are cases with one row.
    Each rate carries a 95% Wilson interval. The health score itself does not:
    the five rates share rows, so adding their variances would overstate precision.

    WHY: A founder needs one number and a sentence about the weakest part,
    while an interviewer needs the weights written down. Time fields carry half
    the weight because every later phase orders events by them. Single-event
    cases are penalised because a process log with no sequences cannot show a
    bottleneck; the penalty is moderate because some businesses really are one
    step.

    WHY NOT: A single "percent of clean rows" is the alternative and is rejected
    because a row can be clean and the file can still be one event per case,
    which is useless for process discovery. Fitting the weights from data is
    rejected because there is no labelled set of good and bad logs, and fitted
    weights cannot be explained to a founder. A bootstrap interval on the score
    is the right uncertainty later; it is a separate phase, and quoting one now
    would be a second algorithm smuggled into ingestion.

    ASSUMPTIONS: Row order is the order the file presented, or the stage order
    produced by reconstruction. The schema roles are the ones detection assigned.
    The worst-case ±10 point sample size is ``ceil(z^2 * 0.5 * 0.5 / 0.10^2)``.

    COMPLEXITY: ``O(n)`` time after grouping, ``O(n)`` space, for ``n`` rows.

    FAILS WHEN: A one-step business is scored down on single-event cases even
    though the file is complete. Backwards time is counted in file order, so a
    file that was sorted for display can hide a bad capture, and a file that
    was saved unsorted is penalised even if the timestamps themselves are true.
    Duplicate keys that differ only by a resource are penalised, which is right
    for a double submit and wrong if the business logs two resources on one event.
    """
    used = dict(weights or QUALITY_WEIGHTS)
    roles_ready = bool(schema.case_id and schema.activity and schema.timestamp)
    if not roles_ready:
        reason = "This check runs only when a case, an activity, and a time are all detected."
        components = {name: _not_applicable(name, reason) for name in used}
        process_score = None
        case_total = _single_event_count(frame, schema)[1] if schema.case_id else 0
    else:
        missing = int(_missing_mask(frame, schema).sum()) if len(frame) else 0
        duplicates, _duplicate_total = _duplicate_count(frame, schema)
        unparseable, unparseable_total = _unparseable_count(frame, schema)
        inversions, pair_total = _inversion_count(frame, schema)
        singles, case_total = _single_event_count(frame, schema)
        components = {
            "missing_required": _rate(
                "missing_required", missing, len(frame),
                unit="events", problem="are missing a case, an activity, or a time",
            ),
            "unparseable_timestamps": _rate(
                "unparseable_timestamps", unparseable, unparseable_total,
                unit="timestamps", problem="could not be read as a date",
            ),
            "out_of_order": _rate(
                "out_of_order", inversions, pair_total,
                unit="consecutive event pairs", problem="go backwards in time",
            ),
            "duplicate_events": _rate(
                "duplicate_events", duplicates, len(frame),
                unit="events", problem="repeat a case, activity, and time that already appeared",
            ),
            "single_event_cases": _rate(
                "single_event_cases", singles, case_total,
                unit="cases", problem="contain only one event",
            ),
        }
        if pair_total == 0:
            components["out_of_order"] = RateReport(
                name="out_of_order",
                bad_count=0,
                total=0,
                bad_fraction=0.0,
                score=100.0,
                lower=0.0,
                upper=1.0,
                required_total=MIN_CASES_FOR_RATE,
                sufficient=False,
                unit="consecutive event pairs",
                detail="No case has two dated events, so time order cannot be checked.",
                applicable=True,
                technical="Single-event cases already cover a log with no pairs.",
            )
        process_score = combine_quality(components, used)

    data_components, data_score = score_data_quality(source if source is not None else frame)
    n_cases = case_total if schema.case_id is not None else None
    n_events = len(frame)
    if n_cases:
        count_text = f"{n_cases} cases, {n_events} events"
        sample_n = n_cases
        sample_unit = "cases"
    else:
        count_text = f"{n_events} rows"
        sample_n = n_events
        sample_unit = "rows"
    formula = f"n = {Z_95}² × 0.5 × 0.5 / 0.10² = {MIN_CASES_FOR_RATE}"
    sufficient = sample_n >= MIN_CASES_FOR_RATE
    sample_sentence = (
        f"{count_text}. A rate stays within about ±10 points once there are "
        f"{MIN_CASES_FOR_RATE} {sample_unit} ({formula})."
    )
    if not sufficient:
        sample_sentence += " Treat this as a preview."
    shown_data = round(data_score, 1)
    if process_score is None:
        summary = f"Data quality is {shown_data} out of 100. Process readiness is not applicable."
    else:
        summary = (
            f"Data quality is {shown_data} out of 100. "
            f"Process readiness is {round(process_score, 1)} out of 100."
        )
    return ValidationResult(
        quality_score=data_score,
        weights=used,
        components=components,
        n_rows=n_events,
        n_cases=n_cases,
        sufficient_for_inference=sufficient,
        minimum_rows_recommended=MIN_CASES_FOR_RATE,
        sample_sentence=sample_sentence,
        summary=summary,
        data_quality_score=data_score,
        process_readiness_score=process_score,
        data_components=data_components,
    )
