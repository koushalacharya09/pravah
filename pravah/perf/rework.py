"""Repeated activities and the extra days they occupy."""

from __future__ import annotations

from pravah.perf.times import Visit


def rework_table(visits: list[Visit]) -> list[dict[str, object]]:
    """Cases that visit an activity more than once, and the days between those visits.

    WHAT: Inside one case, an activity that occurs at least twice is a rework loop. The extra days are the last occurrence minus the first, in days. The table sums those days and reports the median across cases.
    WHY: A repeated step is time that would not exist on the straight path. The span from the first try to the last try is the calendar cost, including the waits between the repeats.
    WHY NOT: Counting only the gaps between consecutive repeats and ignoring the activities in between. Those in-between steps are part of the loop the case actually walked. Charging the whole case throughput time would blame the loop for the first pass as well.
    ASSUMPTIONS: The same activity label means the same step. Two events at the same timestamp add a loop with zero extra days and are still counted. Order follows the visit list, which is time order.
    COMPLEXITY: O(n) time. Space O(n).
    FAILS WHEN: A loop that uses two different labels, such as "demo" and "demo retry", is invisible. A swapped timestamp can also look like a repeat of an earlier step; the date notes from ingestion are the place that possibility is already described.
    """
    seen: dict[tuple[str, str], list[Visit]] = {}
    for visit in visits:
        seen.setdefault((visit.case, visit.activity), []).append(visit)
    extra: dict[str, list[float]] = {}
    for (_case, activity), group in seen.items():
        if len(group) < 2:
            continue
        first = group[0].start
        last = group[-1].end if group[-1].end is not None else group[-1].start
        extra.setdefault(activity, []).append(max(0.0, (last - first).total_seconds() / 86400.0))
    rows = []
    for activity, values in extra.items():
        ordered = sorted(values)
        count = len(ordered)
        half = count // 2
        median = ordered[half] if count % 2 else (ordered[half - 1] + ordered[half]) / 2
        rows.append(
            {
                "activity": activity,
                "cases": count,
                "extra_days": round(sum(values), 2),
                "median_extra_days": round(median, 2),
            }
        )
    rows.sort(key=lambda row: (-float(row["extra_days"]), str(row["activity"])))
    return rows
