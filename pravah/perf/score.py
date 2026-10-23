"""Bottleneck share of throughput time, with a case bootstrap."""

from __future__ import annotations

import numpy as np

from pravah.ingest.constants import RANDOM_SEED
from pravah.perf.times import Visit

DRAWS = 1000


def bootstrap_shares(
    visits: list[Visit],
    *,
    draws: int = DRAWS,
    seed: int = RANDOM_SEED,
) -> list[dict[str, object]]:
    """Rank activities by the share of throughput time spent waiting in front of them.

    WHAT: For each case, throughput time is the last timestamp minus the first. When a finish column exists, the last finish is used. Waiting time in front of an activity is summed inside the case. The score is that sum divided by the sum of throughput times. A percentile interval comes from resampling cases with replacement.
    WHY: A mean wait of three days on a step nobody reaches is not where the organisation loses time. The share says what fraction of the days students actually spend is sitting in front of that step.
    WHY NOT: Ranking by the mean wait alone. A rare step with one long gap would outrank the step every case waits on. A normal-approximation interval was also rejected: eight identical cases have a true interval of width zero, and the normal formula still prints a positive width.
    ASSUMPTIONS: Cases are exchangeable. The seed is ``RANDOM_SEED``. The interval is the 2.5 and 97.5 percentiles of ``draws`` resamples. Cases with no positive throughput time are left out of both the score and the resample.
    COMPLEXITY: O(B n a) time for ``B`` draws, ``n`` cases, and ``a`` activities. Space O(n a).
    FAILS WHEN: Every case is a single event, so throughput time is zero and no share is defined. Two activities with the same share keep alphabetical order, which is not a business ranking.
    """
    per_case: dict[str, dict[str, float]] = {}
    totals: dict[str, float] = {}
    activities: set[str] = set()
    bounds: dict[str, list] = {}
    for visit in visits:
        activities.add(visit.activity)
        per_case.setdefault(visit.case, {})
        per_case[visit.case][visit.activity] = per_case[visit.case].get(visit.activity, 0.0) + visit.wait_days
        bounds.setdefault(visit.case, []).append(visit.start)
        if visit.end is not None:
            bounds[visit.case].append(visit.end)
    for case, stamps in bounds.items():
        totals[case] = max(0.0, (max(stamps) - min(stamps)).total_seconds() / 86400.0)
    names = sorted(activities)
    cases = [case for case, total in totals.items() if total > 0]
    if not cases or not names:
        return []
    wait = np.array([[per_case[case].get(name, 0.0) for name in names] for case in cases], dtype=float)
    total = np.array([totals[case] for case in cases], dtype=float)
    point_den = float(total.sum())
    point = wait.sum(axis=0) / point_den
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(cases), size=(draws, len(cases)))
    sampled_wait = wait[picked].sum(axis=1)
    sampled_total = total[picked].sum(axis=1)
    shares = np.divide(
        sampled_wait,
        sampled_total[:, None],
        out=np.zeros_like(sampled_wait),
        where=sampled_total[:, None] > 0,
    )
    low = np.quantile(shares, 0.025, axis=0)
    high = np.quantile(shares, 0.975, axis=0)
    counts: dict[str, int] = {}
    processes: dict[str, list[float]] = {}
    for visit in visits:
        if visit.case not in totals or totals[visit.case] <= 0:
            continue
        counts[visit.activity] = counts.get(visit.activity, 0) + 1
        bucket = processes.setdefault(visit.activity, [])
        bucket.append(visit.wait_days)
        if visit.process_days is not None:
            processes.setdefault(visit.activity + "\0process", []).append(visit.process_days)
    rows = []
    for index, name in enumerate(names):
        waits = processes.get(name, [])
        done = processes.get(name + "\0process", [])
        rows.append(
            {
                "activity": name,
                "share": round(float(point[index]), 4),
                "share_low": round(float(low[index]), 4),
                "share_high": round(float(high[index]), 4),
                "median_wait_days": round(float(np.median(waits)), 2) if waits else 0.0,
                "median_process_days": None if not done else round(float(np.median(done)), 2),
                "visits": counts.get(name, 0),
            }
        )
    rows.sort(key=lambda row: (-float(row["share"]), str(row["activity"])))
    return rows
