"""Cox model via lifelines, kept only when the hazards stay proportional."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import MIN_COX_EVENTS


def fit_cox(table: pd.DataFrame, segment_names: list[str]) -> dict[str, object]:
    """Fit a Cox model and refuse it when the proportional-hazards check fails.

    WHAT: Segment columns become indicators, dropping the most common level. ``CoxPHFitter`` estimates a hazard ratio per indicator. Schoenfeld residuals, ranked in time, test whether that ratio stays flat. If any p-value is under 0.05, the ratios are not returned.
    WHY: A single hazard ratio is readable only when the gap between two groups keeps the same direction. A crossed pair of curves does not have one ratio. The stratified Kaplan–Meier curves are the picture that remains valid.
    WHY NOT: A penalised Cox that always returns a number. A ridge would hide a failed fit. A time-varying coefficient model is the next step after a failed check, and it is not this page. Writing the partial likelihood from scratch was rejected because the check, not the optimiser, is the claim, and lifelines is the allowed implementation.
    ASSUMPTIONS: Lifelines is installed. Indicators with no variation are dropped. The rank time transform is the Schoenfeld test's default. The 0.05 line is the same false-positive line used elsewhere, applied per covariate without a second correction. ``MIN_COX_EVENTS`` (40) is the floor before a fit is attempted.
    COMPLEXITY: The partial likelihood is iterative. Time is roughly O(iterations * n * p). Space O(n p).
    FAILS WHEN: Fewer than ``MIN_COX_EVENTS`` observed events, or the optimiser does not converge. The sentence names that floor. A small sample can also fail the Schoenfeld test by noise and hide a real constant ratio.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except ImportError:
        return {"fitted": False, "proportional_hazards": None, "sentence": "The Cox model is not available in this install.", "ratios": []}
    observed = int(table["event"].sum())
    if not segment_names:
        return {
            "fitted": False,
            "proportional_hazards": None,
            "sentence": "A Cox model needs a segment column with two groups.",
            "ratios": [],
        }
    if observed < MIN_COX_EVENTS:
        return {
            "fitted": False,
            "proportional_hazards": None,
            "sentence": (
                f"A Cox model needs at least {MIN_COX_EVENTS} observed events. "
                f"This file has {observed}, so no hazard ratio is shown."
            ),
            "ratios": [],
        }
    design = table[["duration", "event"]].copy()
    for name in segment_names:
        values = table[name].astype(str)
        reference = values.mode().iloc[0]
        for level in sorted(set(values) - {reference}):
            column = f"{name}={level}"
            design[column] = (values == level).astype(int)
    covariates = [column for column in design.columns if column not in {"duration", "event"}]
    covariates = [column for column in covariates if design[column].nunique() > 1][:6]
    if not covariates:
        return {"fitted": False, "proportional_hazards": None, "sentence": "No segment had a second level to compare.", "ratios": []}
    design = design[["duration", "event", *covariates]]
    model = CoxPHFitter()
    try:
        model.fit(design, duration_col="duration", event_col="event")
        check = proportional_hazard_test(model, design, time_transform="rank")
    except Exception:
        return {
            "fitted": False,
            "proportional_hazards": None,
            "sentence": "The Cox model did not converge, so no hazard ratio is shown.",
            "ratios": [],
        }
    p_values = [float(value) for value in check.summary["p"].tolist()]
    holds = all(value >= 0.05 for value in p_values)
    if not holds:
        return {
            "fitted": True,
            "proportional_hazards": False,
            "sentence": "The proportional-hazards assumption does not hold, so stratified Kaplan-Meier curves are shown instead of one hazard ratio.",
            "ratios": [],
        }
    ratios = []
    summary = model.summary
    for name in covariates:
        ratios.append(
            {
                "name": name,
                "hr": round(float(summary.loc[name, "exp(coef)"]), 3),
                "low": round(float(summary.loc[name, "exp(coef) lower 95%"]), 3),
                "high": round(float(summary.loc[name, "exp(coef) upper 95%"]), 3),
                "p": round(float(summary.loc[name, "p"]), 4),
            }
        )
    return {
        "fitted": True,
        "proportional_hazards": True,
        "sentence": "The proportional-hazards check did not find a change in the hazard ratio over time.",
        "ratios": ratios,
    }
