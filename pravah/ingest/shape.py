"""Decide whether a sheet is a long event log, a wide stage sheet, or neither."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import (
    DATETIME_PARSE_RATE,
    SHAPE_MARGIN_FOR_FULL_CONFIDENCE,
    SHAPE_MIN_CONFIDENCE,
    TWO_DATE_GAP_ADJUSTMENT,
)
from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import ShapeResult
from pravah.ingest.profile import ColumnProfile, profile_frame


def _entity_strength(frame: pd.DataFrame, profiles: list[ColumnProfile]) -> float:
    """Highest cardinality ratio among short, non-date columns."""
    best = 0.0
    for profile in profiles:
        if profile.parse_rate >= DATETIME_PARSE_RATE or profile.freetext:
            continue
        if profile.cardinality_ratio > best:
            best = profile.cardinality_ratio
    return best


def _case_candidate(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> tuple[ColumnProfile | None, float]:
    """Finest column that still repeats. Strength is 1 when one exists."""
    candidates = [
        profile
        for profile in profiles
        if profile.parse_rate < DATETIME_PARSE_RATE
        and not profile.freetext
        and profile.mean_repeats >= 2.0
        and profile.nunique >= 2
        and profile.null_fraction <= 0.5
    ]
    if not candidates:
        return None, 0.0
    chosen = max(candidates, key=lambda profile: (profile.nunique, profile.name))
    return chosen, 1.0


def _activity_strength(
    frame: pd.DataFrame,
    profiles: list[ColumnProfile],
    case: ColumnProfile | None,
) -> float:
    """Fraction of multi-row cases in which the best label column changes."""
    if case is None:
        return 0.0
    best = 0.0
    for profile in profiles:
        if profile.name == case.name or profile.parse_rate >= DATETIME_PARSE_RATE:
            continue
        if profile.freetext or not (2 <= profile.nunique <= 40):
            continue
        sub = frame[[case.name, profile.name]].dropna()
        if sub.empty:
            continue
        counts = sub.groupby(case.name, sort=False)[profile.name].size()
        nunique = sub.groupby(case.name, sort=False)[profile.name].nunique()
        eligible = counts[counts >= 2]
        if eligible.empty:
            continue
        varies = float((nunique.loc[eligible.index] >= 2).mean())
        if varies > best:
            best = varies
    return best if best >= 0.5 else 0.0


def _median_days(deltas: pd.Series) -> float | None:
    if deltas.empty:
        return None
    days = deltas.dt.total_seconds() / 86400.0
    positive = days[days >= 0]
    if positive.empty:
        return None
    return float(positive.median())


def _two_date_gaps(
    frame: pd.DataFrame,
    date_columns: list[str],
    case_name: str | None,
) -> tuple[float | None, float | None]:
    """Median within-row gap and median between-row gap, in days."""
    if len(date_columns) != 2:
        return None, None
    left_name, right_name = date_columns
    parsed = pd.DataFrame(
        {
            left_name: parse_datetimes(frame[left_name]),
            right_name: parse_datetimes(frame[right_name]),
        },
        index=frame.index,
    )
    both = parsed.dropna()
    if both.empty:
        return None, None
    within = _median_days((both[left_name] - both[right_name]).abs())

    anchor = both.min(axis=1).sort_values()
    if case_name is not None and case_name in frame.columns:
        grouped = frame.loc[both.index, case_name]
        gaps: list[pd.Series] = []
        for _key, idx in grouped.groupby(grouped, sort=False).groups.items():
            times = both.loc[idx].min(axis=1).sort_values()
            if len(times) >= 2:
                gaps.append(times.diff().dropna())
        between_series = pd.concat(gaps) if gaps else pd.Series(dtype="timedelta64[ns]")
    else:
        between_series = anchor.diff().dropna()
    between = _median_days(between_series)
    return within, between


def _confidence(primary: float, secondary: float) -> float:
    margin = primary - secondary
    reach = min(1.0, max(0.0, margin) / SHAPE_MARGIN_FOR_FULL_CONFIDENCE)
    discount = 0.5 + 0.5 * reach
    return primary * discount


def detect_shape(frame: pd.DataFrame) -> ShapeResult:
    """Classify a table as an event log, a wide stage sheet, or a generic table.

    WHAT: Two scores in ``[0, 1]``. The long score is
    ``0.45 * time_part + 0.30 * case_strength + 0.25 * activity_strength``
    when a repeating case column and a label that changes inside the case both
    exist. ``time_part`` is 1 with one date column, 0.7 with two, and 0.3 with
    three or more. The wide score is ``0.50 * min(1, (n_dates - 1) / 2) + 0.50 * entity``
    when at least two columns parse as dates and some non-date column is unique
    on at least 85% of rows. When exactly two date columns exist and both
    scores are positive, a gap test moves 0.15 from one score to the other:
    a within-row gap of at least two days that is also more than twice the
    between-row gap supports two stages (wide); a within-row gap under a third
    of the between-row gap supports a start and an end on the same event (long).
    The winning score must reach 0.55. Confidence is the winning score, reduced
    when the runner-up is within 0.20 of it. Below 0.55 the shape is generic
    and confidence is ``1 - max(long, wide)``.

    WHY: A long log stores one event per row, so the case id repeats and a
    label changes inside the case. A wide sheet stores one entity per row and
    one column per stage, so a code is nearly unique and several columns are
    dates. The scores are the documented form of that sentence. The two-date
    gap test separates "start and end of one tutoring session" from "enquiry
    date and enrol date", which have the same column counts.

    WHY NOT: A trained classifier is the usual alternative and is rejected
    because a new founder has no labelled archive of spreadsheets, and a wrong
    label would be unauditable. Column-name dictionaries (``case_id``,
    ``start_date``) are rejected because the spec forbids assuming names, and
    tuition files use ``student_id`` and ``enquiry_date`` interchangeably with
    names in another language.

    ASSUMPTIONS: Dates are parseable by ``parse_datetimes``. An entity code is
    a short value, not a comment. Stage dates on one row are usually days
    apart, while a start and end on one event are usually much closer than the
    gap between events.

    COMPLEXITY: ``O(n m)`` time to profile ``m`` columns of ``n`` rows, plus
    ``O(n log n)`` for the gap sort. Space ``O(n)`` for the parsed date copies.

    FAILS WHEN: An event log has two timestamps per event, the activity itself
    lasts many days, and events follow each other quickly. The gap test then
    votes wide. A wide sheet with exactly two stage dates on the same day, plus
    a region column and a status column, can vote long. Three or more stage
    dates with a row id still vote wide. A one-row-per-case export with a
    single date is generic: there is no sequence to reconstruct.
    """
    if frame.empty or len(frame.columns) == 0:
        return ShapeResult(
            shape="generic_table",
            confidence=1.0,
            long_score=0.0,
            wide_score=0.0,
            signals={"n_rows": float(len(frame)), "n_datetime_columns": 0.0},
            reasons=["The sheet has no rows, so there is no process to read."],
        )

    profiles = profile_frame(frame)
    date_columns = [p.name for p in profiles if p.parse_rate >= DATETIME_PARSE_RATE]
    n_dt = len(date_columns)
    entity = _entity_strength(frame, profiles)
    case, case_strength = _case_candidate(frame, profiles)
    activity_strength = _activity_strength(frame, profiles, case)

    long_score = 0.0
    if n_dt >= 1 and case_strength > 0 and activity_strength > 0:
        if n_dt == 1:
            time_part = 1.0
        elif n_dt == 2:
            time_part = 0.7
        else:
            time_part = 0.3
        long_score = 0.45 * time_part + 0.30 * case_strength + 0.25 * activity_strength

    wide_score = 0.0
    if n_dt >= 2 and entity >= 0.85:
        date_part = min(1.0, (n_dt - 1) / 2.0)
        wide_score = 0.50 * date_part + 0.50 * min(1.0, entity)

    within_gap, between_gap = (None, None)
    if n_dt == 2 and long_score > 0 and wide_score > 0:
        within_gap, between_gap = _two_date_gaps(
            frame, date_columns, None if case is None else case.name
        )
        if within_gap is not None and between_gap is not None and between_gap > 0:
            if within_gap >= 2.0 and within_gap > 2.0 * between_gap:
                wide_score += TWO_DATE_GAP_ADJUSTMENT
                long_score -= TWO_DATE_GAP_ADJUSTMENT
            elif within_gap * 3.0 < between_gap:
                long_score += TWO_DATE_GAP_ADJUSTMENT
                wide_score -= TWO_DATE_GAP_ADJUSTMENT

    long_score = min(1.0, max(0.0, long_score))
    wide_score = min(1.0, max(0.0, wide_score))
    signals = {
        "n_rows": float(len(frame)),
        "n_datetime_columns": float(n_dt),
        "entity_strength": entity,
        "case_strength": case_strength,
        "activity_strength": activity_strength,
        "long_score": long_score,
        "wide_score": wide_score,
    }
    if within_gap is not None:
        signals["within_row_gap_days"] = within_gap
    if between_gap is not None:
        signals["between_row_gap_days"] = between_gap

    best = max(long_score, wide_score)
    if best < SHAPE_MIN_CONFIDENCE:
        confidence = 1.0 - best
        reasons = [
            (
                f"This sheet does not look like a process log "
                f"(event-log score {long_score:.0%}, wide-sheet score {wide_score:.0%}). "
                f"{n_dt} date column(s) were found."
            )
        ]
        return ShapeResult("generic_table", confidence, long_score, wide_score, signals, reasons)

    if long_score > wide_score or (
        long_score == wide_score and n_dt <= 1
    ):
        shape = "event_log"
        primary, secondary = long_score, wide_score
        lead = (
            f"This sheet looks like an event log: a case column repeats and a "
            f"label changes inside the case, across {n_dt} date column(s)."
        )
    else:
        shape = "wide_sheet"
        primary, secondary = wide_score, long_score
        lead = (
            f"This sheet looks like one row per case with {n_dt} stage-date "
            f"columns. The strongest identifier is filled on {entity:.0%} of rows "
            f"as a distinct value."
        )
    confidence = _confidence(primary, secondary)
    reasons = [
        f"{lead} Confidence is {confidence:.0%} "
        f"(event-log score {long_score:.0%}, wide-sheet score {wide_score:.0%})."
    ]
    if abs(long_score - wide_score) < SHAPE_MARGIN_FOR_FULL_CONFIDENCE:
        reasons.append(
            "The two readings are close. The choice above is the better score, "
            "not a certain one."
        )
    return ShapeResult(shape, confidence, long_score, wide_score, signals, reasons)
