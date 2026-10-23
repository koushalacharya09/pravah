"""Baseline validation and what-if scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import SchemaResult
from pravah.risk.markov import transition_counts
from pravah.sim.engine import run_replication
from pravah.sim.forecast import seasonal_forecast, weekly_counts
from pravah.survival.events import prepare_cases

MEDIAN_TOLERANCE = 0.35
SHARE_TOLERANCE = 0.20
REPLICATIONS = 16


def _inputs(frame: pd.DataFrame, schema: SchemaResult) -> dict[str, object] | None:
    if not (schema.case_id and schema.activity and schema.timestamp):
        return None
    if schema.case_id not in frame.columns or schema.activity not in frame.columns or schema.timestamp not in frame.columns:
        return None
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    if working.empty:
        return None
    ordered = working.sort_values([schema.case_id, schema.timestamp], kind="mergesort")
    gaps: list[float] = []
    services: dict[str, list[float]] = {}
    firsts = []
    shares: dict[str, int] = {}
    durations = []
    for _case, group in ordered.groupby(schema.case_id, sort=False):
        stamps = list(group[schema.timestamp])
        labels = [str(value) for value in group[schema.activity]]
        firsts.append(stamps[0])
        durations.append((stamps[-1] - stamps[0]).total_seconds() / 86400.0)
        for label in labels:
            shares[label] = shares.get(label, 0) + 1
        for label, left, right in zip(labels, stamps, stamps[1:]):
            services.setdefault(label, []).append(max(0.05, (right - left).total_seconds() / 86400.0))
        if labels:
            services.setdefault(labels[-1], []).append(0.05)
    firsts.sort()
    for left, right in zip(firsts, firsts[1:]):
        gaps.append(max(0.05, (right - left).total_seconds() / 86400.0))
    total = sum(shares.values()) or 1
    prepared = prepare_cases(frame, schema, event="completion", completion_activity=None, dropout_days=14)
    completion = None if prepared is None else str(prepared["completion_activity"])
    counts = transition_counts(frame, schema, completion)
    activities = sorted(shares)
    routing: dict[str, dict[str, float]] = {}
    for activity in activities:
        row = {right: float(count) for (left, right), count in counts.items() if left == activity and right in activities}
        if not row:
            row = {"end": 1.0}
        routing[activity] = row
    starts = ordered.groupby(schema.case_id, sort=False).head(1)[schema.activity].astype(str)
    start = str(starts.mode().iloc[0]) if not starts.empty else activities[0]
    return {
        "interarrivals": np.array(gaps or [1.0]),
        "services": {name: np.array(values) for name, values in services.items()},
        "routing": routing,
        "start": start,
        "arrivals": len(firsts),
        "observed_median": float(np.median(durations)) if durations else 0.0,
        "observed_shares": {name: count / total for name, count in shares.items()},
        "activities": activities,
    }


def _band(samples: list[float]) -> dict[str, float]:
    array = np.array([value for value in samples if np.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"median": None, "low": None, "high": None}
    return {
        "median": round(float(np.median(array)), 4),
        "low": round(float(np.quantile(array, 0.05)), 4),
        "high": round(float(np.quantile(array, 0.95)), 4),
    }


def analyze_simulation(frame: pd.DataFrame, schema: SchemaResult, settings: dict[str, object] | None = None) -> dict[str, object]:
    """Validate a baseline, then run the requested scenario.

    WHAT: The baseline uses the empirical inter-arrival times, empirical service times, the Markov routing, and one server per activity. It must land within 35% of the observed median throughput time and within 0.20 of every stage share. A failed baseline blocks every what-if number. A passing baseline runs the scenario for 16 replications and reports the median throughput with a 90% interval.
    WHY: A simulator that cannot reproduce the file will happily produce a precise saving. The badge is the permission to read the scenario.
    WHY NOT: One long run. A single seed can sit in the tail. The interval is the replication distribution, not a confidence interval for a mean.
    ASSUMPTIONS: Capacity starts at 32 so a first run is not a one-server queue. When the request does not name a wait scale, the illustrated scenario cuts service time at the slowest step in half. The scenario may raise capacity, scale wait at one activity, skip one activity, or scale arrivals by the forecast. The tolerance numbers are the ones in this function.
    COMPLEXITY: O(replications * events log events).
    FAILS WHEN: The sheet is not a process log, or the baseline misses the tolerance. ``validated`` is false and ``scenarios`` is empty.
    """
    inputs = _inputs(frame, schema)
    if inputs is None:
        return {"applicable": False, "validated": False, "verdict": "What-if needs a case, an activity, and a time.", "forecast": None}
    capacity = {name: 32 for name in inputs["activities"]}
    baseline = [
        run_replication(
            inputs["interarrivals"], inputs["services"], inputs["routing"], capacity, inputs["start"],
            arrivals=int(inputs["arrivals"]), seed=42 + index,
        )
        for index in range(8)
    ]
    medians = _band([row["median"] for row in baseline])
    share_error = 0.0
    for name, observed in inputs["observed_shares"].items():
        simulated = np.mean([row["shares"].get(name, 0.0) for row in baseline])
        share_error = max(share_error, abs(float(simulated) - float(observed)))
    observed = float(inputs["observed_median"])
    simulated_median = float(medians["median"] or 0.0)
    relative = abs(simulated_median - observed) / observed if observed > 1e-6 else abs(simulated_median - observed)
    validated = relative <= MEDIAN_TOLERANCE and share_error <= SHARE_TOLERANCE
    forecast = seasonal_forecast(weekly_counts(frame, schema))
    comparison = {
        "observed_median": round(observed, 4),
        "simulated_median": medians["median"],
        "relative_error": round(float(relative), 4),
        "share_error": round(float(share_error), 4),
        "median_tolerance": MEDIAN_TOLERANCE,
        "share_tolerance": SHARE_TOLERANCE,
    }
    if not validated:
        return {
            "applicable": True,
            "validated": False,
            "verdict": (
                f"Baseline median {simulated_median:.2f} days versus observed {observed:.2f} "
                f"(relative {relative:.0%}), stage-share gap {share_error:.2f}. "
                "What-if is blocked because the simulator does not reproduce this file."
            ),
            "comparison": comparison,
            "forecast": forecast,
            "activities": inputs["activities"],
            "scenario": None,
        }
    settings = settings or {}
    slowest = max(
        inputs["activities"],
        key=lambda name: float(np.median(inputs["services"].get(name, np.array([0.05])))),
    )
    activity = str(settings.get("activity") or slowest)
    scenario_capacity = dict(capacity)
    scenario_capacity[activity] = max(1, int(settings.get("capacity") or capacity[activity]))
    if "wait_scale" in settings and settings["wait_scale"] is not None:
        wait_scale = float(settings["wait_scale"])
    else:
        wait_scale = 0.5
    wait = {activity: wait_scale}
    skip = str(settings.get("skip") or "") or None
    volume = float(settings.get("volume") or 1.0)
    if settings.get("use_forecast") and forecast.get("next"):
        weeks = weekly_counts(frame, schema)
        current = int(inputs["arrivals"]) / max(1, len(weeks))
        volume = max(0.2, float(np.mean(forecast["next"])) / max(0.1, current))
    scenario = [
        run_replication(
            inputs["interarrivals"], inputs["services"], inputs["routing"], scenario_capacity, inputs["start"],
            arrivals=int(inputs["arrivals"]), seed=100 + index, wait_scale=wait, skip=skip, volume=volume,
        )
        for index in range(REPLICATIONS)
    ]
    outcome = _band([row["median"] for row in scenario])
    return {
        "applicable": True,
        "validated": True,
        "verdict": f"Baseline is within tolerance. Scenario median throughput is {outcome['median']} days.",
        "comparison": comparison,
        "forecast": forecast,
        "activities": inputs["activities"],
        "scenario": {
            "median": outcome["median"],
            "low": outcome["low"],
            "high": outcome["high"],
            "baseline": medians["median"],
            "activity": activity,
            "wait_scale": wait_scale,
        },
    }
