"""Weekly median throughput time and one slowdown change-point."""

from __future__ import annotations

import pandas as pd

from pravah.perf.times import Visit

MIN_SEGMENT = 2
# The later regime must explain at least this fraction of the one-mean residual
# sum of squares, and it must be slower, before a date is named.
MIN_RSS_DROP = 0.20


def _rss(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values)


def slowdown_index(values: list[float]) -> int | None:
    """Index of the first week of a slower regime, or None.

    WHAT: Binary segmentation with one split. Each candidate keeps at least two weeks on either side. The cost is the residual sum of squares around a constant mean. The split is kept only when the later mean is higher and the cost falls by at least 20% against a single mean.
    WHY: A founder asks when the process got slower, not for a full piecewise model. One split answers that with a date. Residual sum of squares prefers a split that actually moves the level, not a split that only shortens one noisy week.
    WHY NOT: CUSUM on the cumulative sum. CUSUM is built to detect a sustained shift and needs a threshold in standard-error units that a four-week series cannot estimate. PELT would return several changes this page would then have to pick among.
    ASSUMPTIONS: The series is a weekly median, already in time order. A speedup is not reported by this function. The 20% drop is a practical effect size, not a significance test.
    COMPLEXITY: O(T^2) time for ``T`` weeks. Space O(T).
    FAILS WHEN: Fewer than four weeks. A slow drift with no level change can miss the 20% bar and return None even though the last week is the slowest.
    """
    if len(values) < MIN_SEGMENT * 2:
        return None
    base = _rss(values)
    if base <= 0:
        return None
    best: tuple[float, int] | None = None
    for split in range(MIN_SEGMENT, len(values) - MIN_SEGMENT + 1):
        earlier = values[:split]
        later = values[split:]
        if sum(later) / len(later) <= sum(earlier) / len(earlier):
            continue
        cost = _rss(earlier) + _rss(later)
        drop = (base - cost) / base
        if drop < MIN_RSS_DROP:
            continue
        if best is None or cost < best[0]:
            best = (cost, split)
    return None if best is None else best[1]


def weekly_trend(visits: list[Visit]) -> dict[str, object]:
    """Median throughput time by the week the case ended.

    WHAT: A case contributes one throughput time, dated on the Monday of its last timestamp. The change-point sentence names that Monday when ``slowdown_index`` finds a split.
    WHY: Weekly bins match how an operator looks back. The case is dated when it finished, because that is when the throughput time became known.
    WHY NOT: Dating the case on its start. A case that started in a fast week and finished in a slow one would be blamed on the start. Calendar months hide a change that falls mid-month.
    ASSUMPTIONS: Throughput time uses the same start and end as the bottleneck score. Weeks with no finished case are absent, not filled with zero.
    COMPLEXITY: O(n) time plus the O(T^2) scan. Space O(n).
    FAILS WHEN: Fewer than four distinct weeks. The series is still returned, with a sentence that no date was named.
    """
    bounds: dict[str, list[pd.Timestamp]] = {}
    for visit in visits:
        stamp = visit.end if visit.end is not None else visit.start
        bounds.setdefault(visit.case, []).append(stamp)
    weeks: dict[pd.Timestamp, list[float]] = {}
    for stamps in bounds.values():
        if len(stamps) < 2:
            continue
        total = (max(stamps) - min(stamps)).total_seconds() / 86400.0
        if total <= 0:
            continue
        end = pd.Timestamp(max(stamps))
        monday = (end - pd.Timedelta(int(end.dayofweek), unit="D")).normalize()
        weeks.setdefault(monday, []).append(total)
    ordered = sorted(weeks)
    series = []
    medians: list[float] = []
    for monday in ordered:
        values = sorted(weeks[monday])
        count = len(values)
        half = count // 2
        median = values[half] if count % 2 else (values[half - 1] + values[half]) / 2
        medians.append(median)
        series.append(
            {"week": monday.date().isoformat(), "median_days": round(median, 2), "cases": count}
        )
    split = slowdown_index(medians)
    if split is None:
        sentence = (
            f"{len(series)} week(s) of throughput time. "
            "No slowdown stood out, or there are fewer than 4 weeks."
        )
        change = None
    else:
        change = series[split]["week"]
        sentence = f"Throughput time slowed down around {change}."
    return {"weeks": series, "change_point": change, "sentence": sentence}
