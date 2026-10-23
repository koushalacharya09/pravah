"""Propensity scores, overlap, IPW, and matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pravah.ingest.constants import RANDOM_SEED


def standardised_mean_difference(treated: np.ndarray, control: np.ndarray, weights_t: np.ndarray | None = None, weights_c: np.ndarray | None = None) -> float:
    """Difference of means divided by the pooled standard deviation.

    WHAT: ``(mean_t - mean_c) / sqrt((var_t + var_c) / 2)``. Weights, when given, are the IPW weights.
    WHY: A raw mean difference grows with the units of the column. The standardised difference is comparable across a fee and a day count. Under 0.1 is the usual sign that the groups are close enough to compare.
    WHY NOT: A t-test. A large file makes a tiny leftover difference "significant". Balance is a size, not a p-value.
    ASSUMPTIONS: Weights are non-negative. A column with zero pooled variance has difference 0 when the means match, and infinity otherwise, which the caller treats as unbalanced.
    COMPLEXITY: O(n) time. Space O(1).
    FAILS WHEN: One side is empty. The caller skips the column. A weight of zero on every row of one side is the same failure.
    """
    if weights_t is None:
        weights_t = np.ones(len(treated))
    if weights_c is None:
        weights_c = np.ones(len(control))
    if weights_t.sum() <= 0 or weights_c.sum() <= 0:
        return float("inf")
    mean_t = np.average(treated, weights=weights_t)
    mean_c = np.average(control, weights=weights_c)
    var_t = np.average((treated - mean_t) ** 2, weights=weights_t)
    var_c = np.average((control - mean_c) ** 2, weights=weights_c)
    scale = np.sqrt((var_t + var_c) / 2)
    if scale <= 1e-12:
        return 0.0 if abs(mean_t - mean_c) <= 1e-12 else float("inf")
    return float((mean_t - mean_c) / scale)


def fit_effect(table: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], *, draws: int = 300, seed: int = RANDOM_SEED) -> dict[str, object]:
    """IPW and matching estimates of the treatment difference, with a bootstrap interval.

    WHAT: A logistic regression estimates the propensity score from the confounders. Rows with a score outside 0.05 to 0.95 are trimmed. Stabilised IPW weights reweight the treated and untreated outcome means. Matching pairs each treated row with the nearest unused control on the logit score, inside a caliper of 0.2 standard deviations. The bootstrap resamples rows and refits both estimators.
    WHY: Two estimators that share a propensity model but not a formula are a check on each other. The interval is a case resample because the rows are the sample, and a normal interval would hide skew from a few extreme weights.
    WHY NOT: A regression of the outcome on the treatment alone. That is the association the confounders are here to challenge. Doubly robust estimation adds a second outcome model this sheet has not justified.
    ASSUMPTIONS: The treatment and the outcome are 0/1. Confounders are numeric. The seed is the project seed. Trimming keeps both groups at size at least 8 or the estimate is refused.
    COMPLEXITY: O(draws * n * p) for the logistic fits. Matching is O(n^2) once.
    FAILS WHEN: The treatment does not vary, or the logistic regression does not converge. The sentence says no effect was estimated. A confounder that is a copy of the treatment makes the score 0 or 1 and the trim removes everyone.
    """
    needed = [treatment, outcome, *confounders]
    if any(name not in table.columns for name in needed):
        return {"estimated": False, "sentence": "The treatment, outcome, or a confounder is not on this table."}
    design = table[needed].dropna().astype(float)
    treat = design[treatment].to_numpy()
    y_value = design[outcome].to_numpy()
    if len(set(treat.tolist())) < 2 or len(design) < 20:
        return {"estimated": False, "sentence": "The treatment needs both groups, and at least 20 rows, before an effect is estimated."}
    point = _once(design, treatment, outcome, confounders)
    if point is None:
        return {"estimated": False, "sentence": "Propensity scores piled up at 0 or 1, so the overlap trim removed a group."}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        picked = design.iloc[rng.integers(0, len(design), len(design))]
        estimated = _once(picked, treatment, outcome, confounders)
        if estimated is not None:
            samples.append(estimated["ipw"])
    low, high = (None, None)
    if len(samples) >= 30:
        low, high = np.quantile(samples, [0.025, 0.975])
    balance = _balance(design, treatment, confounders)
    return {
        "estimated": True,
        "sentence": f"The IPW difference is {point['ipw']:.3f}.",
        "ipw": round(float(point["ipw"]), 4),
        "ipw_low": None if low is None else round(float(low), 4),
        "ipw_high": None if high is None else round(float(high), 4),
        "matching": round(float(point["matching"]), 4),
        "matched_pairs": int(point["pairs"]),
        "kept": int(point["kept"]),
        "balance": balance,
        "overlap": point["overlap"],
        "propensity_treated": point["propensity_treated"],
        "propensity_control": point["propensity_control"],
    }


def _scores(design: pd.DataFrame, treatment: str, confounders: list[str]) -> np.ndarray | None:
    x_values = design[confounders].to_numpy(dtype=float) if confounders else np.ones((len(design), 1))
    y_values = design[treatment].to_numpy(dtype=int)
    if len(set(y_values.tolist())) < 2:
        return None
    model = LogisticRegression(max_iter=400, random_state=RANDOM_SEED)
    try:
        model.fit(x_values, y_values)
    except Exception:
        return None
    return model.predict_proba(x_values)[:, 1]


def _once(design: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict[str, object] | None:
    propensity = _scores(design, treatment, confounders)
    if propensity is None:
        return None
    treat = design[treatment].to_numpy(dtype=float)
    y_value = design[outcome].to_numpy(dtype=float)
    keep = (propensity >= 0.05) & (propensity <= 0.95)
    if int(((treat == 1) & keep).sum()) < 8 or int(((treat == 0) & keep).sum()) < 8:
        return None
    propensity, treat, y_value = propensity[keep], treat[keep], y_value[keep]
    portion = float(treat.mean())
    weights = np.where(treat == 1, portion / propensity, (1.0 - portion) / (1.0 - propensity))
    ipw = np.average(y_value[treat == 1], weights=weights[treat == 1]) - np.average(y_value[treat == 0], weights=weights[treat == 0])
    matching, pairs = _match(propensity, treat, y_value)
    return {
        "ipw": float(ipw),
        "matching": matching,
        "pairs": pairs,
        "kept": int(len(treat)),
        "propensity": propensity,
        "weights": weights,
        "overlap": {"trimmed": True, "low": 0.05, "high": 0.95},
        "propensity_treated": [round(float(value), 4) for value in propensity[treat == 1][:80]],
        "propensity_control": [round(float(value), 4) for value in propensity[treat == 0][:80]],
    }


def _match(propensity: np.ndarray, treat: np.ndarray, y_value: np.ndarray) -> tuple[float, int]:
    logit = np.log(propensity / (1.0 - propensity))
    scale = float(np.std(logit))
    caliper = 0.2 * scale if scale > 0 else 0.0
    treated = np.flatnonzero(treat == 1)
    control = list(np.flatnonzero(treat == 0))
    diffs = []
    for row in treated:
        if not control:
            break
        distances = [abs(logit[row] - logit[other]) for other in control]
        best = int(np.argmin(distances))
        if caliper and distances[best] > caliper:
            continue
        partner = control.pop(best)
        diffs.append(y_value[row] - y_value[partner])
    if not diffs:
        return 0.0, 0
    return float(np.mean(diffs)), len(diffs)


def _balance(design: pd.DataFrame, treatment: str, confounders: list[str]) -> list[dict[str, object]]:
    scores = _scores(design, treatment, confounders)
    if scores is None:
        return []
    treat = design[treatment].to_numpy(dtype=float)
    keep = (scores >= 0.05) & (scores <= 0.95)
    if int(keep.sum()) < 16:
        return []
    treat = treat[keep]
    scores = scores[keep]
    portion = float(treat.mean())
    weights = np.where(treat == 1, portion / scores, (1.0 - portion) / (1.0 - scores))
    rows = []
    for name in confounders:
        values = design[name].to_numpy(dtype=float)[keep]
        before = standardised_mean_difference(values[treat == 1], values[treat == 0])
        after = standardised_mean_difference(values[treat == 1], values[treat == 0], weights[treat == 1], weights[treat == 0])
        rows.append({"name": name, "before": round(float(before), 4), "after": round(float(after), 4), "balanced": abs(after) < 0.1})
    return rows
