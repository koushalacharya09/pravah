"""Thresholds for ingestion.

Every number here is a decision. The derivation lives in ``docs/THEORY.md``
and the alternatives live in ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import math

# Later phases draw bootstrap samples from this seed. Ingestion itself is a
# pure function of the file and does not draw random numbers.
RANDOM_SEED: int = 42

# Two-sided 5% normal quantile, to two decimals. The more precise 0.975
# quantile is 1.95996..., which rounds to the same textbook value.
Z_95: float = 1.96

# Practical precedence: the earlier column must be strictly earlier at least
# this often. A coin-flip (0.5) is not a stage order.
PRECEDENCE_THRESHOLD: float = 0.75

# The 95% Wilson lower bound must clear a coin-flip, or the direction is
# still compatible with noise.
PRECEDENCE_LOWER_BOUND: float = 0.50

# Fewer co-observed rows than this never cast a vote. One swapped date in
# four rows moves a proportion by 0.25.
MIN_PAIR_SUPPORT: int = 5

# Wald sample size for a ±10 point interval at p=0.75 (the decision boundary,
# where variance inside the accept region is largest):
# n = 1.96^2 * 0.75 * 0.25 / 0.10^2 = 72.03, rounded up.
PRECEDENCE_ROWS_FOR_PM10: int = 73

# Worst-case Wald size for a ±10 point interval: n = z^2 * p * (1-p) / E^2
# with p = 0.5 and E = 0.10. That is 96.04, and the next whole sample is 97.
# There is no separate round-number floor.
RATE_HALF_WIDTH: float = 0.10
RATE_WORST_CASE_P: float = 0.5
MIN_CASES_FOR_RATE: int = math.ceil(
    (Z_95**2) * RATE_WORST_CASE_P * (1.0 - RATE_WORST_CASE_P) / (RATE_HALF_WIDTH**2)
)
MIN_ROWS_FOR_INFERENCE: int = MIN_CASES_FOR_RATE

# Prediction is not the same formula. A model wants about 10 events per feature
# before it is worth fitting, and about 20 before the fit is comfortable.
EVENTS_PER_FEATURE: int = 10
EVENTS_PER_FEATURE_COMFORTABLE: int = 20

# Cox models need more events than a log-rank. Forty observed events is the
# floor before lifelines is asked to fit; below that the Hessian is often
# singular and a ratio would be noise.
MIN_COX_EVENTS: int = 40

# A shape score below this is a generic table, not a process log.
SHAPE_MIN_CONFIDENCE: float = 0.55

# A column is date-like at or above this parse rate.
DATETIME_PARSE_RATE: float = 0.80

# A schema role below this is left unassigned rather than guessed.
ROLE_MIN_SCORE: float = 0.45

# How close two shape scores can be before confidence is discounted.
SHAPE_MARGIN_FOR_FULL_CONFIDENCE: float = 0.20

# Added only when a two-date sheet's within-row gap and between-row gap
# disagree about "two stages" versus "start and end of one event".
# 0.15 is just larger than the base gap between those two scores (~0.12).
TWO_DATE_GAP_ADJUSTMENT: float = 0.15

DEFAULT_PII_SALT: str = "pravah-ingest-v1"
ANON_PREFIX: str = "anon:"

# Data-quality weights. Outliers drop out when a column is too small to score,
# and the remaining weights are renormalised so a skipped check is not a penalty.
DATA_QUALITY_WEIGHTS: dict[str, float] = {
    "missing_cells": 40.0,
    "duplicate_rows": 30.0,
    "mixed_types": 30.0,
    "outliers": 15.0,
}

# Process-readiness weights. A check that does not apply is left out and the
# rest are renormalised. They are not a data-quality penalty.
QUALITY_WEIGHTS: dict[str, float] = {
    "missing_required": 25.0,
    "unparseable_timestamps": 25.0,
    "out_of_order": 20.0,
    "duplicate_events": 15.0,
    "single_event_cases": 15.0,
}
