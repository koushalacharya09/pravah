"""Discrete-event simulation of cases, servers, and routing."""

from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np


def run_replication(
    interarrivals: np.ndarray,
    services: dict[str, np.ndarray],
    routing: dict[str, dict[str, float]],
    capacity: dict[str, int],
    starts: str,
    *,
    arrivals: int,
    seed: int,
    wait_scale: dict[str, float] | None = None,
    skip: str | None = None,
    volume: float = 1.0,
) -> dict[str, float]:
    """One replication: arrivals, queues, service, and a next step.

    WHAT: A heap holds arrivals, service completions, and the moment a server frees. Inter-arrival times and service times are drawn by resampling the empirical lists. The next activity is drawn from the transition row. ``skip`` sends that activity's arrivals straight to its most common successor. ``volume`` scales how many cases arrive. ``wait_scale`` multiplies service time at that activity, which is the observed gap to the next step.
    WHY: The clock is an event list because the next thing that happens is the soonest scheduled event, which is what ``heapq`` returns. Resampling the observed times keeps the simulation inside the file instead of a fitted exponential.
    WHY NOT: A closed formula for a Jackson network. The routing here is an empirical matrix and the service times are not exponential, so the formula's assumptions are already false.
    ASSUMPTIONS: Capacity is at least 1. A missing service list uses one day. Volume below 1 keeps a fraction of arrivals. The seed is ``RANDOM_SEED`` plus the replication index, supplied by the caller.
    COMPLEXITY: O((arrivals + events) log events) time. Space O(events).
    FAILS WHEN: There is no start activity or no inter-arrival time. The caller does not run a replication and the validation badge stays off.
    """
    if interarrivals.size == 0 or not starts or starts not in routing:
        return {"median": float("nan"), "shares": {}}
    rng = np.random.default_rng(seed)
    count = max(1, int(round(arrivals * volume)))
    clock = 0.0
    sequence = 0
    queue: list[tuple[float, int, str, int, str]] = []
    busy: dict[str, int] = defaultdict(int)
    waiting: dict[str, list[tuple[int, float]]] = defaultdict(list)
    done: list[float] = []
    visits: dict[str, int] = defaultdict(int)

    def push(time: float, kind: str, case: int, activity: str) -> None:
        nonlocal sequence
        sequence += 1
        heapq.heappush(queue, (time, sequence, kind, case, activity))

    for case in range(count):
        clock += float(rng.choice(interarrivals))
        push(clock, "arrive", case, starts)
    started: dict[int, float] = {}
    steps: dict[int, int] = defaultdict(int)
    finished: set[int] = set()
    while queue and len(done) < count:
        time, _seq, kind, case, activity = heapq.heappop(queue)
        if case in finished:
            continue
        if steps[case] > 16:
            finished.add(case)
            done.append(time - started.get(case, time))
            continue
        if kind == "arrive":
            started.setdefault(case, time)
            steps[case] += 1
            _enter(activity, case, time, busy, waiting, capacity, services, rng, push, visits, wait_scale, skip, routing)
        elif kind == "finish":
            steps[case] += 1
            busy[activity] = max(0, busy[activity] - 1)
            nxt = _next(activity, routing, rng, skip)
            if nxt is None:
                finished.add(case)
                done.append(time - started.get(case, time))
            else:
                _enter(nxt, case, time, busy, waiting, capacity, services, rng, push, visits, wait_scale, skip, routing)
            if waiting[activity]:
                nxt_case, queued_at = waiting[activity].pop(0)
                scale = 1.0 if not wait_scale else float(wait_scale.get(activity, 1.0))
                delayed = time + max(0.0, (time - queued_at) * (scale - 1.0))
                _begin(activity, nxt_case, delayed, busy, services, rng, push, visits, wait_scale)
        if time > 1e7:
            break
    total = sum(visits.values()) or 1
    shares = {name: count_ / total for name, count_ in visits.items()}
    median = float(np.median(done)) if done else float("nan")
    return {"median": median, "shares": shares, "finished": float(len(done))}


def _next(activity: str, routing: dict[str, dict[str, float]], rng: np.random.Generator, skip: str | None) -> str | None:
    row = routing.get(activity, {})
    if skip and skip in row and len(row) > 1:
        row = {name: weight for name, weight in row.items() if name != skip}
    names = [name for name, weight in row.items() if weight > 0 and name != "end"]
    if not names:
        return None
    weights = np.array([row[name] for name in names], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


def _enter(activity, case, time, busy, waiting, capacity, services, rng, push, visits, wait_scale, skip, routing) -> None:
    if skip == activity:
        nxt = _next(activity, routing, rng, skip)
        if nxt is None:
            push(time, "finish", case, activity)
        else:
            _enter(nxt, case, time, busy, waiting, capacity, services, rng, push, visits, wait_scale, skip, routing)
        return
    limit = max(1, int(capacity.get(activity, 1)))
    if busy[activity] >= limit:
        waiting[activity].append((case, time))
        return
    _begin(activity, case, time, busy, services, rng, push, visits, wait_scale)


def _begin(activity, case, time, busy, services, rng, push, visits, wait_scale=None) -> None:
    busy[activity] += 1
    visits[activity] += 1
    pool = services.get(activity)
    span = float(rng.choice(pool)) if pool is not None and len(pool) else 1.0
    scale = 1.0 if not wait_scale else float(wait_scale.get(activity, 1.0))
    push(time + max(0.01, span * max(0.05, scale)), "finish", case, activity)
