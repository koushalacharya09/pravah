"""Refutations that have to pass before an effect is called causal."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pravah.causes.effect import fit_effect
from pravah.ingest.constants import RANDOM_SEED


def e_value(effect: float, baseline: float) -> float | None:
    """How strong an unseen confounder must be to explain away a risk difference.

    WHAT: Convert the treated and untreated outcome rates into a risk ratio, then ``RR + sqrt(RR * (RR - 1))``.
    WHY: A refutation that only shuffles columns already in the file cannot see a confounder that was never recorded. The E-value is the minimum association that missing confounder would need with both the treatment and the outcome.
    WHY NOT: A partial R-squared formula that needs a continuous outcome model. The outcomes here are yes or no.
    ASSUMPTIONS: ``effect`` is treated rate minus untreated rate. ``baseline`` is the untreated rate. Both rates are clipped into (0.01, 0.99). A risk ratio below 1 is flipped, so the value is the strength needed to erase the association in either direction.
    COMPLEXITY: O(1) time and space.
    FAILS WHEN: The baseline is outside (0, 1) after clipping is impossible, so this returns None only when the effect is not finite.
    """
    if not math.isfinite(effect) or not math.isfinite(baseline):
        return None
    control = min(0.99, max(0.01, baseline))
    treated = min(0.99, max(0.01, control + effect))
    ratio = treated / control
    if ratio < 1:
        ratio = 1.0 / ratio
    return float(ratio + math.sqrt(ratio * (ratio - 1.0)))


def refute(table: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], estimate: dict[str, object]) -> dict[str, object]:
    """Placebo, random cause, subset, and sensitivity checks.

    WHAT: A placebo shuffles the treatment and must produce an interval that contains 0. A random common cause is a standard normal column; the original effect must keep its sign and its interval must still exclude 0. A 75% subset must keep the sign. Sensitivity reports the E-value and passes only above 1.2. The verdict is likely causal only when balance holds and every check passes.
    WHY: Any one check can be lucky. The product rule stops a single passing placebo from upgrading an association.
    WHY NOT: Calling the IPW number causal because the diagram was drawn. The diagram is an assumption the user can edit. These checks are the part the file can actually fail.
    ASSUMPTIONS: The seed is the project seed. The subset is the first 75% of a shuffled copy. The E-value uses the IPW difference and the untreated outcome rate.
    COMPLEXITY: Three extra IPW fits, each with its own bootstrap of 40 draws.
    FAILS WHEN: The original interval includes 0. The verdict is association only, because a difference compatible with zero has nothing for a confounder to explain away.
    """
    checks: list[dict[str, object]] = []
    ipw = estimate.get("ipw")
    low = estimate.get("ipw_low")
    high = estimate.get("ipw_high")
    interval_excludes = ipw is not None and low is not None and high is not None and (float(low) > 0 or float(high) < 0)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = table.copy()
    shuffled[treatment] = rng.permutation(shuffled[treatment].to_numpy())
    placebo = fit_effect(shuffled, treatment, outcome, confounders, draws=40)
    placebo_ok = placebo.get("estimated") and placebo.get("ipw_low") is not None and float(placebo["ipw_low"]) <= 0 <= float(placebo["ipw_high"])
    checks.append({"name": "placebo treatment", "passed": bool(placebo_ok), "detail": placebo.get("sentence", "")})
    noisy = table.copy()
    noisy["random_cause"] = rng.normal(size=len(noisy))
    random_effect = fit_effect(noisy, treatment, outcome, [*confounders, "random_cause"], draws=40)
    random_ok = False
    if interval_excludes and random_effect.get("estimated") and random_effect.get("ipw_low") is not None:
        same_sign = float(random_effect["ipw"]) * float(ipw) > 0
        still = float(random_effect["ipw_low"]) > 0 or float(random_effect["ipw_high"]) < 0
        random_ok = same_sign and still
    checks.append({"name": "random common cause", "passed": bool(random_ok), "detail": random_effect.get("sentence", "")})
    subset = table.sample(frac=0.75, random_state=RANDOM_SEED)
    subset_effect = fit_effect(subset, treatment, outcome, confounders, draws=40)
    subset_ok = subset_effect.get("estimated") and ipw is not None and float(subset_effect["ipw"]) * float(ipw) > 0
    checks.append({"name": "data subset", "passed": bool(subset_ok), "detail": subset_effect.get("sentence", "")})
    baseline = float(table.loc[table[treatment] == 0, outcome].mean()) if treatment in table else 0.5
    strength = None if ipw is None else e_value(float(ipw), baseline)
    sensitivity_ok = strength is not None and strength >= 1.2 and interval_excludes
    detail = "No E-value." if strength is None else f"An unseen confounder would need an association of {strength:.2f} to erase the effect."
    checks.append({"name": "sensitivity analysis", "passed": bool(sensitivity_ok), "detail": detail})
    balance_ok = bool(estimate.get("balance")) and all(row.get("balanced") for row in estimate.get("balance", []))
    passed = interval_excludes and balance_ok and all(check["passed"] for check in checks)
    if passed:
        verdict = "likely causal"
        reason = "Balance is under 0.1 and every refutation test held."
    else:
        verdict = "association only"
        missing = [check["name"] for check in checks if not check["passed"]]
        if not interval_excludes:
            missing = ["the confidence interval includes 0", *missing]
        if not balance_ok:
            missing = ["covariate balance is above 0.1", *missing]
        reason = "Failed: " + ", ".join(missing) + "."
    return {"checks": checks, "verdict": verdict, "reason": reason, "e_value": None if strength is None else round(strength, 3)}
