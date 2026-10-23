"""Kaplan–Meier against a hand table and against lifelines, plus censoring rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from pravah.ingest.models import SchemaResult
from pravah.survival.analyze import analyze_survival
from pravah.survival.km import kaplan_meier
from pravah.survival.logrank import logrank
from pravah.web.service import analyze_sample


def _schema(**roles: str | None) -> SchemaResult:
    return SchemaResult(
        case_id=roles.get("case_id"),
        activity=roles.get("activity"),
        timestamp=roles.get("timestamp"),
        resource=roles.get("resource"),
        column_scores={},
        confidence=1.0,
        notes=[],
    )


def test_greenwood_matches_the_hand_product_and_lifelines() -> None:
    durations = np.array([1, 2, 3, 4], dtype=float)
    observed = np.array([1, 0, 1, 1])
    curve = {row["time"]: row for row in kaplan_meier(durations, observed)["curve"]}
    # t=1: 1/4 die, S=0.75. Greenwood = 1/(4*3).
    # t=3: 1 of the 2 still at risk die, S=0.375. Greenwood = 1/12 + 1/2.
    assert curve[1]["survival"] == 0.75
    assert curve[1]["greenwood"] == round(1 / 12, 8)
    assert curve[3]["survival"] == 0.375
    assert curve[3]["at_risk"] == 2
    fitted = KaplanMeierFitter()
    fitted.fit(durations, observed)
    ours = kaplan_meier(durations, observed)["curve"]
    for row in ours:
        if row["time"] == 0 or row["survival"] == 0:
            continue
        at = fitted.survival_function_at_times(row["time"]).iloc[0]
        assert row["survival"] == pytest.approx(float(at), abs=1e-6)
        band = fitted.confidence_interval_survival_function_.iloc[
            fitted.survival_function_.index.get_indexer([row["time"]], method="nearest")[0]
        ]
        assert row["low"] == pytest.approx(float(band.iloc[0]), abs=1e-4)
        assert row["high"] == pytest.approx(float(band.iloc[1]), abs=1e-4)


def test_an_open_case_is_not_counted_as_a_failure() -> None:
    durations = np.array([5, 5, 5, 9], dtype=float)
    censored = np.array([1, 1, 1, 0])
    failed = np.array([1, 1, 1, 1])
    open_curve = kaplan_meier(durations, censored)["curve"][-1]["survival"]
    closed_curve = kaplan_meier(durations, failed)["curve"][-1]["survival"]
    assert open_curve == 0.25
    assert closed_curve == 0.0


def test_logrank_statistic_matches_lifelines() -> None:
    durations = np.array([1, 2, 3, 4, 8, 9, 10, 11], dtype=float)
    observed = np.ones(8)
    groups = np.array(["early"] * 4 + ["late"] * 4)
    statistic, _p = logrank(durations, observed, groups)
    reference = logrank_test(durations[:4], durations[4:], event_observed_A=observed[:4], event_observed_B=observed[4:])
    assert statistic == pytest.approx(float(reference.test_statistic), rel=1e-6)


def test_completion_censors_cases_that_never_arrive_and_dropout_uses_the_threshold() -> None:
    frame = pd.DataFrame(
        {
            "case": ["A", "A", "B", "B", "C"],
            "step": ["start", "done", "start", "done", "start"],
            "when": ["2024-01-01", "2024-01-05", "2024-01-01", "2024-01-11", "2024-01-08"],
            "area": ["North", "North", "South", "South", "North"],
        }
    )
    schema = _schema(case_id="case", activity="step", timestamp="when")
    completion = analyze_survival(frame, schema, {"event": "completion", "completion_activity": "done"})
    assert completion["events"] == 2
    assert completion["censored"] == 1
    assert "censoring" in completion["censor_note"]
    assert completion["verdict"].startswith("Most completions happen")
    # Study ends 11 Jan. C's last event is 8 Jan, three days of silence.
    quiet = analyze_survival(frame, schema, {"event": "dropout", "dropout_days": 2, "completion_activity": "done"})
    assert quiet["events"] == 1
    assert quiet["verdict"].startswith("Most drop-offs happen")
    loud = analyze_survival(frame, schema, {"event": "dropout", "dropout_days": 14, "completion_activity": "done"})
    assert loud["events"] == 0
    assert loud["verdict"] == "No drop-off was observed."


def test_many_segments_use_benjamini_hochberg_and_a_crossed_hazard_is_refused() -> None:
    rows = []
    for index in range(40):
        area = "North" if index < 20 else "South"
        channel = "web" if index % 2 == 0 else "walk"
        if area == "North":
            done = "2024-01-03" if index < 10 else "2024-02-10"
        else:
            done = "2024-01-31" if index < 30 else "2024-01-09"
        rows.append((f"N{index}", "start", "2024-01-01", area, channel))
        rows.append((f"N{index}", "done", done, area, channel))
    frame = pd.DataFrame(rows, columns=["case", "step", "when", "area", "channel"])
    report = analyze_survival(frame, _schema(case_id="case", activity="step", timestamp="when"))
    assert "Benjamini-Hochberg" in report["segment_note"]
    assert report["cox"]["proportional_hazards"] is False
    assert "Kaplan-Meier" in report["cox"]["sentence"]
    assert report["cox"]["ratios"] == []
    assert any(segment["curves"] for segment in report["segments"])


def test_event_log_completes_at_attend_and_a_plain_table_refuses() -> None:
    event = analyze_sample("event_log")
    survival = event["survival"]
    assert survival["applicable"] is True
    assert survival["completion_activity"] == "attend"
    assert survival["censored"] == 0
    cox = survival["cox"]
    assert cox["fitted"] is False
    assert "40" in cox["sentence"]
    assert "observed events" in cox["sentence"]
    plain = analyze_sample("generic")
    assert plain["survival"]["applicable"] is False
