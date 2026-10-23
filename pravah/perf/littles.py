"""Little's Law against the observed work-in-progress of one activity."""

from __future__ import annotations

from pravah.perf.times import Visit

# A 25% gap between L and λW is larger than rounding. It is the flag, not a test.
STEADY_GAP = 0.25
MIN_COMPLETED = 5


def _sojourn(visits: list[Visit], index: int, case_visits: list[Visit]) -> tuple[float, float] | None:
    """Arrival day and departure day for one completed visit, or None if censored."""
    visit = case_visits[index]
    arrival = visit.start
    if visit.end is not None and visit.process_days is not None:
        return arrival.value / 1e9 / 86400.0, visit.end.value / 1e9 / 86400.0
    if index + 1 >= len(case_visits):
        return None
    departure = case_visits[index + 1].start
    if departure <= arrival:
        return None
    return arrival.value / 1e9 / 86400.0, departure.value / 1e9 / 86400.0


def _wip(intervals: list[tuple[float, float]], start: float, end: float) -> tuple[float, float, float]:
    """Time-average level over the whole window and over each half."""
    points: list[tuple[float, int]] = []
    for arrival, departure in intervals:
        points.append((arrival, 1))
        points.append((departure, -1))
    points.sort(key=lambda item: (item[0], -item[1]))
    cuts = [start, (start + end) / 2, end]
    areas = [0.0, 0.0]
    level = 0
    previous = start
    cursor = 0
    for moment, delta in points:
        while cursor < 2 and moment > cuts[cursor + 1]:
            areas[cursor] += level * (cuts[cursor + 1] - previous)
            previous = cuts[cursor + 1]
            cursor += 1
        if cursor < 2 and moment > previous:
            areas[cursor] += level * (moment - previous)
            previous = moment
        level += delta
    while cursor < 2:
        areas[cursor] += level * (cuts[cursor + 1] - previous)
        previous = cuts[cursor + 1]
        cursor += 1
    half = (end - start) / 2
    whole = end - start
    return (areas[0] + areas[1]) / whole, areas[0] / half, areas[1] / half


def littles_check(visits: list[Visit]) -> list[dict[str, object]]:
    """Compare λW with the time-average number of cases inside each activity.

    WHAT: For each activity, λ is completed arrivals divided by the window from the first arrival to the last departure. W is the mean completed time in the activity. L = λW. The observed L is the time-average of how many cases are between arrival and departure. Over a window that contains every arrival and its departure, those two numbers match. The steady-state flag compares the time-average in the first half of the window with the second half. A relative gap above 0.25 means the queue is not stable.
    WHY: Little's Law is an identity for a stable system, and it is also an identity for a finite sample that starts and ends empty. The average can match while the queue is still filling. The two halves show that. A bottleneck share from a filling queue should not be read as a permanent capacity problem.
    WHY NOT: A formal stationarity test on the arrival series. Eight cases have no power for that test. Comparing λW with the same rows' time-average cannot fail, so it is reported as the check and the halves are the flag.
    ASSUMPTIONS: The last event of a case has no observed departure, so it is excluded from both W and the work-in-progress. In start/end mode the time in the activity is the processing time. In transition mode it is the gap until the next event, which is the sojourn of this activity, not the waiting time in front of the next one.
    COMPLEXITY: O(n log n) time per activity for the sweep. Space O(n).
    FAILS WHEN: Fewer than 5 completed visits. The row is reported as not checked. A window shorter than an hour is also not checked, because a rate per day is noise.
    """
    by_case: dict[str, list[Visit]] = {}
    for visit in visits:
        by_case.setdefault(visit.case, []).append(visit)
    activities = sorted({visit.activity for visit in visits})
    rows = []
    for activity in activities:
        intervals: list[tuple[float, float]] = []
        for case_visits in by_case.values():
            for index, visit in enumerate(case_visits):
                if visit.activity != activity:
                    continue
                span = _sojourn(visits, index, case_visits)
                if span is not None and span[1] > span[0]:
                    intervals.append(span)
        if len(intervals) < MIN_COMPLETED:
            rows.append(
                {
                    "activity": activity,
                    "checked": False,
                    "steady": None,
                    "sentence": (
                        f"{activity} has {len(intervals)} completed visits. "
                        f"Little's Law needs {MIN_COMPLETED} before a steady-state check."
                    ),
                }
            )
            continue
        start = min(item[0] for item in intervals)
        end = max(item[1] for item in intervals)
        window = end - start
        if window <= 1.0 / 24.0:
            rows.append(
                {
                    "activity": activity,
                    "checked": False,
                    "steady": None,
                    "sentence": f"{activity} fits inside one hour, so a daily rate is not meaningful.",
                }
            )
            continue
        mean_w = sum(item[1] - item[0] for item in intervals) / len(intervals)
        lam = len(intervals) / window
        formula = lam * mean_w
        observed, first_half, second_half = _wip(intervals, start, end)
        # Over a window that starts and ends empty, λW equals the time-average
        # exactly. A growing queue still matches on average, so the two halves
        # are what show that the process is not in steady state.
        half_gap = abs(second_half - first_half) / max(first_half, second_half, 1e-9)
        steady = half_gap <= STEADY_GAP
        state = "in steady state" if steady else "not in steady state"
        rows.append(
            {
                "activity": activity,
                "checked": True,
                "arrivals": len(intervals),
                "lambda_per_day": round(lam, 4),
                "mean_days": round(mean_w, 3),
                "L_formula": round(formula, 3),
                "L_observed": round(observed, 3),
                "relative_gap": round(half_gap, 3),
                "steady": steady,
                "sentence": (
                    f"{activity}: L = λW gives {formula:.2f}, observed work-in-progress is {observed:.2f}. "
                    f"The second half of the window holds {second_half:.2f} against {first_half:.2f} in the first half, "
                    f"so this step is {state}."
                ),
            }
        )
    return rows
