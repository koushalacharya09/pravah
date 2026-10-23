"""Column-role scores used by schema detection. Names are not evidence."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.anonymise import name_match_rate
from pravah.ingest.constants import DATETIME_PARSE_RATE
from pravah.ingest.profile import ColumnProfile

def _plateau(
    value: float,
    *,
    zero_below: float,
    full_from: float,
    full_to: float,
    zero_above: float,
) -> float:
    """Trapezoid score: 0 outside the outer edges, 1 on the flat middle."""
    if value <= zero_below or value >= zero_above:
        return 0.0
    if full_from <= value <= full_to:
        return 1.0
    if value < full_from:
        return (value - zero_below) / (full_from - zero_below)
    return (zero_above - value) / (zero_above - full_to)


def _coverage(frame: pd.DataFrame, case_col: str, col: str) -> float:
    """Median, over cases, of (labels seen in the case) / (labels seen anywhere)."""
    if case_col == col:
        return 0.0
    sub = frame[[case_col, col]].dropna()
    if sub.empty:
        return 0.0
    global_n = int(sub[col].nunique())
    if global_n == 0:
        return 0.0
    per_case = sub.groupby(case_col, sort=False)[col].nunique()
    return float((per_case / global_n).median())


def _within_variation(frame: pd.DataFrame, case_col: str, col: str) -> float:
    """Share of cases with at least two rows whose values are not all equal."""
    if case_col == col:
        return 0.0
    sub = frame[[case_col, col]].dropna()
    if sub.empty:
        return 0.0
    counts = sub.groupby(case_col, sort=False)[col].size()
    nunique = sub.groupby(case_col, sort=False)[col].nunique()
    eligible = counts[counts >= 2]
    if eligible.empty:
        return 0.0
    return float((nunique.loc[eligible.index] >= 2).mean())


def _cross_case(frame: pd.DataFrame, case_col: str, col: str) -> float:
    """1 when the average value shows up in three or more cases, else scaled down."""
    if case_col == col:
        return 0.0
    sub = frame[[case_col, col]].dropna()
    if sub.empty:
        return 0.0
    mean_cases = float(sub.groupby(col, sort=False)[case_col].nunique().mean())
    return float(min(1.0, max(0.0, (mean_cases - 1.0) / 2.0)))


def _case_score(
    profile: ColumnProfile,
    *,
    max_nunique: int,
    anonymised: bool,
) -> float:
    if anonymised or profile.parse_rate >= DATETIME_PARSE_RATE or profile.freetext:
        return 0.0
    if profile.nunique < 2 or profile.n_non_null == 0 or max_nunique <= 0:
        return 0.0
    repeat = _plateau(
        profile.mean_repeats, zero_below=1.2, full_from=2.0, full_to=30.0, zero_above=80.0
    )
    fineness = _plateau(
        profile.cardinality_ratio, zero_below=0.02, full_from=0.08, full_to=0.60, zero_above=0.95
    )
    relative = profile.nunique / max_nunique
    return repeat * fineness * (1.0 - profile.null_fraction) * (0.55 + 0.45 * relative)


def _activity_score(
    frame: pd.DataFrame,
    profile: ColumnProfile,
    case_col: str | None,
) -> float:
    if profile.name == case_col or profile.parse_rate >= DATETIME_PARSE_RATE or profile.freetext:
        return 0.0
    if profile.nunique < 2 or profile.nunique > 150:
        return 0.0
    card = _plateau(float(profile.nunique), zero_below=1, full_from=2, full_to=25, zero_above=80)
    reuse = _plateau(
        profile.mean_repeats, zero_below=1.0, full_from=3.0, full_to=500.0, zero_above=100_000.0
    )
    if case_col is None:
        coverage = 0.0
        varies = 0.0
    else:
        coverage = _coverage(frame, case_col, profile.name)
        varies = _within_variation(frame, case_col, profile.name)
    string_bonus = 1.0 if profile.stringy else 0.6
    return (0.30 * card + 0.30 * coverage + 0.25 * varies + 0.15 * reuse) * string_bonus


def _timestamp_score(
    frame: pd.DataFrame,
    profile: ColumnProfile,
    case_col: str | None,
) -> float:
    if profile.name == case_col or profile.parse_rate < 0.60:
        return 0.0
    if case_col is None:
        varies = 1.0
    else:
        varies = _within_variation(frame, case_col, profile.name)
    return profile.parse_rate * (0.70 + 0.30 * varies)


def _resource_score(
    frame: pd.DataFrame,
    profile: ColumnProfile,
    case_col: str | None,
    n_cases: int,
) -> float:
    if profile.name == case_col or profile.parse_rate >= DATETIME_PARSE_RATE or profile.freetext:
        return 0.0
    if case_col is None:
        cross = 0.0
        varies = 0.0
    else:
        cross = _cross_case(frame, case_col, profile.name)
        varies = _within_variation(frame, case_col, profile.name)
    card = _plateau(
        profile.nunique / max(n_cases, 1),
        zero_below=0.0,
        full_from=0.10,
        full_to=0.85,
        zero_above=1.20,
    )
    string_bonus = 1.0 if profile.stringy else 0.5
    raw = (0.40 * cross + 0.25 * varies + 0.20 * card + 0.15 * string_bonus) * (
        1.0 - profile.null_fraction
    )
    if varies < 0.15:
        raw *= 0.5
    return raw


def _best(
    scores: dict[str, float],
    blocked: set[str],
    order: list[str],
) -> tuple[str | None, float]:
    """Highest score. Equal scores keep the left-most column in ``order``."""
    chosen: str | None = None
    chosen_score = -1.0
    for name in order:
        if name in blocked:
            continue
        score = scores.get(name, 0.0)
        if score > chosen_score:
            chosen = name
            chosen_score = score
    if chosen is None:
        return None, 0.0
    return chosen, chosen_score


def _wide_case_scores(frame: pd.DataFrame, profiles: list[ColumnProfile], anonymised: set[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for profile in profiles:
        if profile.name in anonymised or profile.parse_rate >= DATETIME_PARSE_RATE or profile.freetext:
            scores[profile.name] = 0.0
            continue
        name_rate = name_match_rate(frame[profile.name]) if profile.stringy else 0.0
        scores[profile.name] = (
            profile.cardinality_ratio * (1.0 - profile.null_fraction) * (1.0 - 0.5 * name_rate)
        )
    return scores


