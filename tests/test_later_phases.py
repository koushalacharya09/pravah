"""Survival-adjacent checks for risk, causes, simulation, and the action plan."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.models import SchemaResult
from pravah.plan.knapsack import knapsack
from pravah.risk.markov import absorption_probabilities
from pravah.risk.prefix import build_prefixes, features_at
from pravah.risk.shapley import exact_shapley
from pravah.web.service import analyze_sample


def _schema() -> SchemaResult:
    return SchemaResult(
        case_id="case",
        activity="activity",
        timestamp="timestamp",
        resource=None,
        column_scores={},
        confidence=1.0,
        notes=[],
    )


def _log(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["case", "activity", "timestamp"])


def test_a_prefix_cannot_see_a_later_event() -> None:
    frame = _log(
        [
            ("c1", "enquiry", "2024-01-01"),
            ("c1", "secret", "2024-01-08"),
            ("c2", "enquiry", "2024-01-02"),
            ("c2", "secret", "2024-01-09"),
        ]
    )
    table = build_prefixes(frame, _schema(), completion="secret", dropout_days=14)
    assert set(table["last_activity"]) == {"enquiry"}
    assert int(table["events_so_far"].max()) == 1
    seen = features_at([{"time": 0.0, "activity": "enquiry", "resource_load": 1.0}])
    assert seen["last_activity"] == "enquiry"
    assert "secret" not in seen.values()


def test_shapley_matches_a_linear_model() -> None:
    weights = np.array([2.0, -1.0])

    def predict(data: np.ndarray) -> np.ndarray:
        return data @ weights

    values = exact_shapley(predict, np.array([3.0, 5.0]), np.array([1.0, 1.0]))
    assert np.allclose(values, [4.0, -4.0])


def test_absorption_follows_a_certain_path() -> None:
    rows = absorption_probabilities({("A", "B"): 10, ("B", "complete"): 10})
    by_name = {row["activity"]: row for row in rows}
    assert by_name["A"]["complete"] == 1.0
    assert by_name["A"]["drop"] == 0.0


def test_event_log_refuses_to_train() -> None:
    report = analyze_sample("event_log")["risk"]
    assert report["trained"] is False
    assert "events per feature" in report["verdict"]
    assert report["cases"] == []
    assert report["markov"]


def test_knapsack_beats_the_greedy_ratio() -> None:
    packed = knapsack([60, 100, 120], [10, 20, 30], 50)
    assert packed["value"] == 220
    assert packed["greedy_value"] == 160
    assert packed["differs"] is True


def test_known_treatment_effect_is_recovered() -> None:
    from pravah.causes.effect import fit_effect

    rng = np.random.default_rng(42)
    confounder = rng.normal(size=500)
    propensity = 1 / (1 + np.exp(-1.2 * confounder))
    treatment = rng.random(500) < propensity
    baseline = 1 / (1 + np.exp(-1.2 * confounder))
    treated = 1 / (1 + np.exp(-(1.6 + 1.2 * confounder)))
    truth = float((treated - baseline).mean())
    outcome = rng.random(500) < np.where(treatment, treated, baseline)
    table = pd.DataFrame({"t": treatment.astype(float), "y": outcome.astype(float), "c": confounder})
    estimate = fit_effect(table, "t", "y", ["c"], draws=60)
    assert estimate["estimated"] is True
    assert abs(float(estimate["ipw"]) - truth) < 0.12
    assert estimate["ipw_low"] < estimate["ipw"] < estimate["ipw_high"]
    assert all(row["balanced"] for row in estimate["balance"])


def test_two_step_log_validates_before_a_scenario() -> None:
    from pravah.sim.analyze import analyze_simulation

    rows = []
    for index in range(30):
        start = pd.Timestamp("2010-03-01") + pd.Timedelta(int(index), unit="D")
        rows.append((f"c{index}", "alpha", start.isoformat()))
        rows.append((f"c{index}", "beta", (start + pd.Timedelta(5, unit="D")).isoformat()))
    frame = pd.DataFrame(rows, columns=["case", "activity", "timestamp"])
    report = analyze_simulation(frame, _schema(), {"capacity": 2, "activity": "alpha", "volume": 1})
    assert report["validated"] is True
    assert report["comparison"]["relative_error"] <= 0.35
    assert report["scenario"]["low"] <= report["scenario"]["median"] <= report["scenario"]["high"]


def test_conformal_interval_names_its_coverage() -> None:
    from pravah.risk.arena import run_arena

    rng = np.random.default_rng(42)
    records = []
    for case in range(80):
        start = float(case)
        dropped = int(case % 5 == 0)
        for step, activity in enumerate(("alpha", "beta")):
            records.append(
                {
                    "case": f"c{case}",
                    "start": start,
                    "dropped": dropped,
                    "open": 0,
                    "duration": 10 + dropped * 8,
                    "remaining": 8 - step * 3 + rng.normal(0, 0.2),
                    "time_since_start": float(step),
                    "events_so_far": float(step + 1),
                    "time_since_last": float(step),
                    "rework_count": 0.0,
                    "resource_load": 1.0,
                    "day_of_week": float(case % 7),
                    "last_activity": activity,
                }
            )
    arena = run_arena(pd.DataFrame(records), ["alpha", "beta"])
    assert arena["trained"] is True
    assert arena["winner"]
    assert arena["conformal"]["target"] == 0.9
    assert arena["conformal"]["coverage"] >= 0.7
    assert "PR-AUC" in arena["selection"] or "ROC-AUC" in arena["selection"]
