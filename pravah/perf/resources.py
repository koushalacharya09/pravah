"""Resource ranking after Empirical Bayes shrinkage."""

from __future__ import annotations

from pravah.perf.times import Visit

# A resource with fewer events than this is pulled halfway or more toward the
# overall median, and it is not allowed to outrank a larger resource.
PRIOR_EVENTS = 5


def shrink_resources(visits: list[Visit]) -> list[dict[str, object]]:
    """Rank resources by shrunk median duration, not by the raw median.

    WHAT: Workload is the number of events. The raw time is the median processing time when a finish column exists, otherwise the median transition time in front of the event. The shrunk time is ``n / (n + 5)`` times the group median plus ``5 / (n + 5)`` times the overall median. Resources with fewer than 5 events are listed after every resource that has at least 5.
    WHY: One slow case makes a new tutor the "worst" resource if the ranking uses the raw median. The weight ``n / (n + n0)`` is the normal-normal posterior weight when the prior is worth ``n0`` observations. Five events is the same floor used before a proportion is trusted elsewhere in this project.
    WHY NOT: Estimating the prior variance from the group means. One extreme group of size 1 inflates that variance and then barely shrinks, which is the failure this rule is here to prevent. A mixed-effects model would estimate the same weight with more machinery than a median needs.
    ASSUMPTIONS: The overall median is the median of event durations, not the median of group medians. Longer shrunk time is worse. Hashed resource ids are grouped as they stand.
    COMPLEXITY: O(n log n) time to median-sort each group. Space O(n).
    FAILS WHEN: There is no resource column, or every event lacks a duration. The caller then gets an empty list. A resource that is genuinely much slower, but has only four events, is kept off the top of the list until a fifth event arrives.
    """
    durations: dict[str, list[float]] = {}
    for visit in visits:
        if not visit.resource:
            continue
        value = visit.process_days if visit.process_days is not None else visit.wait_days
        if value is None:
            continue
        durations.setdefault(visit.resource, []).append(float(value))
    if not durations:
        return []
    pooled = sorted(value for values in durations.values() for value in values)
    mid = len(pooled) // 2
    overall = pooled[mid] if len(pooled) % 2 else (pooled[mid - 1] + pooled[mid]) / 2
    rows = []
    for name, values in durations.items():
        ordered = sorted(values)
        count = len(ordered)
        half = count // 2
        median = ordered[half] if count % 2 else (ordered[half - 1] + ordered[half]) / 2
        weight = count / (count + PRIOR_EVENTS)
        shrunk = weight * median + (1.0 - weight) * overall
        rows.append(
            {
                "resource": name,
                "events": count,
                "median_days": round(median, 2),
                "shrunk_days": round(shrunk, 2),
                "small_sample": count < PRIOR_EVENTS,
            }
        )
    rows.sort(key=lambda row: (row["small_sample"], -float(row["shrunk_days"]), row["resource"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
