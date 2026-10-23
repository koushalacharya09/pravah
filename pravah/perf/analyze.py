"""One bottleneck report for a detected event log."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.models import SchemaResult
from pravah.perf.littles import littles_check
from pravah.perf.resources import shrink_resources
from pravah.perf.rework import rework_table
from pravah.perf.score import bootstrap_shares
from pravah.perf.times import build_visits
from pravah.perf.trend import weekly_trend


def _empty(sentence: str) -> dict[str, object]:
    return {
        "applicable": False,
        "verdict": sentence,
        "mode": None,
        "mode_sentence": sentence,
        "activities": [],
        "littles": [],
        "resources": [],
        "trend": {"weeks": [], "change_point": None, "sentence": sentence},
        "rework": [],
        "cases_used": 0,
    }


def analyze_bottlenecks(
    frame: pd.DataFrame,
    schema: SchemaResult,
    *,
    draws: int = 1000,
) -> dict[str, object]:
    """Waiting time, Little's Law, resources, trend, and rework for one log.

    WHAT: Run only when a case, an activity, and a time are all detected. Rank activities by the share of throughput time spent waiting in front of them, with a bootstrap interval. Check Little's Law per activity. Shrink resource medians before ranking. Name one weekly slowdown when the series supports it. List repeated activities and their extra days.
    WHY: The readiness table only says whether a bottleneck rate is allowed. This is the rate. Each piece uses the same visits, so the page cannot show a wait the score did not compute.
    WHY NOT: A directly-follows graph as a prerequisite. The wait is defined on the ordered events of one case. Discovery can draw the map later without changing these numbers. A queueing simulation is a later phase; Little's Law here is a check, not a forecast.
    ASSUMPTIONS: The frame is the anonymised event table. Roles come from schema detection, including a reader's override. Draws default to 1000 and the seed is the project seed.
    COMPLEXITY: Dominated by the bootstrap, O(B n a), and the weekly scan, O(T^2).
    FAILS WHEN: The sheet is not a process log. The report says so and every list is empty. A single-event log has no throughput time, so the ranking is empty even though the roles exist.
    """
    if not (schema.case_id and schema.activity and schema.timestamp):
        return _empty("Bottlenecks need a case, an activity, and a time.")
    missing = [name for name in (schema.case_id, schema.activity, schema.timestamp) if name not in frame.columns]
    if missing or frame.empty:
        return _empty("Bottlenecks need a case, an activity, and a time.")
    mode, note, visits = build_visits(
        frame,
        case_id=schema.case_id,
        activity=schema.activity,
        timestamp=schema.timestamp,
        resource=schema.resource,
    )
    if not visits:
        return _empty("No event had a readable time, so there is no waiting time to rank.")
    activities = bootstrap_shares(visits, draws=draws)
    cases_used = len({visit.case for visit in visits})
    if not activities:
        verdict = "No case has a throughput time, so no bottleneck is ranked."
    else:
        top = activities[0]
        verdict = f"Most time is lost before {top['activity']}."
    return {
        "applicable": True,
        "verdict": verdict,
        "mode": mode,
        "mode_sentence": note,
        "activities": activities,
        "littles": littles_check(visits),
        "resources": shrink_resources(visits),
        "trend": weekly_trend(visits),
        "rework": rework_table(visits),
        "cases_used": cases_used,
    }
