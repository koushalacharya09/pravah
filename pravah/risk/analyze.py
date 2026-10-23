"""Risk report: Markov absorption, a refused or fitted arena, and open cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.constants import EVENTS_PER_FEATURE
from pravah.ingest.models import SchemaResult
from pravah.risk.arena import feature_names, matrix, run_arena
from pravah.risk.markov import absorption_probabilities, transition_counts
from pravah.risk.prefix import build_prefixes
from pravah.risk.shapley import exact_shapley
from pravah.survival.events import prepare_cases


def _empty(sentence: str, markov: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "applicable": False,
        "verdict": sentence,
        "markov": markov or [],
        "arena": {"trained": False, "sentence": sentence, "models": []},
        "cases": [],
    }


def _public_arena(arena: dict[str, object]) -> dict[str, object]:
    def walk(value: object) -> object:
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items() if not str(key).startswith("_")}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(arena)  # type: ignore[return-value]


def analyze_risk(frame: pd.DataFrame, schema: SchemaResult, *, dropout_days: float = 14.0) -> dict[str, object]:
    """Score open cases only with facts already on the clock.

    WHAT: Build prefixes, fit the arena when the events-per-variable rule allows it, and list open cases by the calibrated probability of drop or SLA breach. Each listed case carries a conformal remaining-time interval and the three Shapley facts with the largest absolute push. The Markov absorption probabilities are reported even when the arena is refused.
    WHY: The events-per-variable rule is the same 10-events floor used for prediction readiness. Training below it produces a leaderboard nobody should trust. Shapley values are computed exactly against the training mean because the feature count is small enough to sum over every order.
    WHY NOT: Scoring a finished case as if it were open. Its outcome is already in the file. A global importance chart in place of a per-case reason would not answer which fact moved this student.
    ASSUMPTIONS: The completion activity is the one survival would choose, the modal last step. Dropout days match the survival default of 14. The SLA inside the arena is the training 75th percentile, not a number typed into the product.
    COMPLEXITY: Arena fitting dominates. Shapley is O(2^p) per listed case, and at most 12 cases are explained.
    FAILS WHEN: The sheet is not a process log. The verdict says risk needs a case, an activity, and a time. When the prefix count is under ``10 * features``, the arena is refused and the open-case list is empty.
    """
    if not (schema.case_id and schema.activity and schema.timestamp):
        return _empty("Risk prediction needs a case, an activity, and a time.")
    prepared = prepare_cases(frame, schema, event="completion", completion_activity=None, dropout_days=dropout_days)
    completion = None if prepared is None else str(prepared["completion_activity"])
    markov = absorption_probabilities(transition_counts(frame, schema, completion))
    prefixes = build_prefixes(frame, schema, completion=completion, dropout_days=dropout_days)
    if prefixes.empty:
        return _empty("No prefix could be built, so there is nothing to train on.", markov)
    activities = sorted(prefixes["last_activity"].astype(str).unique().tolist())[:6]
    names = feature_names(activities)
    needed = EVENTS_PER_FEATURE * len(names)
    finished = prefixes[prefixes["open"] == 0]
    if len(finished) < needed:
        sentence = (
            f"Only {len(finished)} finished events. The rule is {EVENTS_PER_FEATURE} events per feature, "
            f"and this model has {len(names)} features, so it needs {needed}. No model was trained."
        )
        return {
            "applicable": True,
            "trained": False,
            "verdict": sentence,
            "markov": markov,
            "arena": {"trained": False, "sentence": sentence, "models": []},
            "cases": [],
        }
    arena = run_arena(prefixes, activities)
    listed = _open_cases(prefixes, activities, names, arena)
    winner = arena.get("winner")
    verdict = arena.get("selection", "")
    if winner:
        verdict = f"The chosen classifier is {winner}. {verdict}"
    return {
        "applicable": True,
        "trained": bool(arena.get("trained")),
        "verdict": verdict,
        "markov": markov,
        "arena": _public_arena(arena),
        "cases": listed,
    }


def _open_cases(prefixes: pd.DataFrame, activities: list[str], names: list[str], arena: dict[str, object]) -> list[dict[str, object]]:
    if not arena.get("trained") or not arena.get("winner"):
        return []
    winner = str(arena["winner"])
    fitted = arena.get("_fitted", {})
    model = fitted.get(winner)
    if model is None:
        return []
    latest = prefixes[prefixes["open"] == 1].sort_values(["case", "events_so_far"]).groupby("case", sort=False).tail(1)
    if latest.empty:
        return []
    background = matrix(prefixes[prefixes["open"] == 0], activities).mean(axis=0)
    rows = []
    conformal = arena.get("conformal", {})
    half = float(conformal.get("quantile") or 0.0)
    reg = conformal.get("_model")
    for _, record in latest.head(12).iterrows():
        encoded = matrix(pd.DataFrame([record]), activities)[0]
        raw = float(model.predict_proba(encoded.reshape(1, -1))[0, 1])
        calibrator = arena.get("_calibrator")
        probability = float(calibrator.predict([raw])[0]) if calibrator is not None else raw
        remaining = float(reg.predict(encoded.reshape(1, -1))[0]) if reg is not None else float(record["remaining"])
        values = exact_shapley(lambda data: model.predict_proba(data)[:, 1], encoded, background)
        order = np.argsort(-np.abs(values))[:3]
        reasons = [
            f"SHAP value: {names[int(index)]} {'raises' if values[int(index)] >= 0 else 'lowers'} the risk by {abs(float(values[int(index)])):.2f}."
            for index in order
        ]
        rows.append(
            {
                "case": record["case"],
                "probability": round(probability, 4),
                "remaining_low": round(max(0.0, remaining - half), 3),
                "remaining_high": round(remaining + half, 3),
                "reasons": reasons,
            }
        )
    rows.sort(key=lambda row: -float(row["probability"]))
    return rows
