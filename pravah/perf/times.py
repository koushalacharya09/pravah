"""Waiting time and processing time inside each case."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pravah.ingest.datetime_parse import parse_datetimes

_DAY = 86400.0


@dataclass(frozen=True)
class Visit:
    """One activity occurrence inside one case."""

    case: str
    activity: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    resource: str | None
    wait_days: float
    process_days: float | None


def _days(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return float((later - earlier).total_seconds()) / _DAY


def find_end_column(frame: pd.DataFrame, timestamp: str) -> str | None:
    """A second clock that is at or after the event time on most rows.

    WHAT: Scan columns other than the event timestamp. Keep a column whose values parse as dates on at least 80% of rows and fall at or after the event time on at least 80% of the rows where both exist. If several qualify, keep the one with the smallest positive median gap.
    WHY: Processing time is the gap from start to finish of the same step. A sheet with only one clock cannot show that gap, and inventing it from the next step would mix the wait for the next activity into this activity's work.
    WHY NOT: Matching column names such as "end" or "finish". A fee date is not a name. A numeric case id can look like an Excel serial, so columns that parse as numbers and not as text dates are skipped by ``parse_datetimes``.
    ASSUMPTIONS: The detected timestamp is the start when a later clock exists. The later clock belongs to the same event, not to the next stage. Wide-sheet stage dates are the event timestamp itself, so they are not a second clock.
    COMPLEXITY: O(n c) time for ``n`` rows and ``c`` columns. Space O(n).
    FAILS WHEN: The finish time is stored on the next row instead of the same row. That column is a different event and is not used. A finish time that is earlier than the start on more than 20% of rows is also refused.
    """
    start = parse_datetimes(frame[timestamp])
    best: tuple[float, str] | None = None
    for column in frame.columns:
        if str(column) == timestamp:
            continue
        parsed = parse_datetimes(frame[column])
        if float(parsed.notna().mean()) < 0.8:
            continue
        both = parsed.notna() & start.notna()
        if int(both.sum()) < 5:
            continue
        if float((parsed[both] >= start[both]).mean()) < 0.8:
            continue
        gap = (parsed[both] - start[both]).dt.total_seconds() / _DAY
        positive = gap[gap > 0]
        if positive.empty:
            continue
        median = float(positive.median())
        if best is None or median < best[0]:
            best = (median, str(column))
    return None if best is None else best[1]


def build_visits(
    frame: pd.DataFrame,
    *,
    case_id: str,
    activity: str,
    timestamp: str,
    resource: str | None,
) -> tuple[str, str, list[Visit]]:
    """Order each case by time and measure the gap in front of each step.

    WHAT: Return ``"start_end"`` when a finish column exists, otherwise ``"transition"``. In start/end mode the wait is the idle gap from the previous finish to this start, and processing is finish minus start. In transition mode the only number is the gap from the previous event time, stored as waiting time, and processing is null.
    WHY: A founder needs to know whether a long gap is idle time or work. Calling a single timestamp a processing time would invent work the file does not contain.
    WHY NOT: Treating the gap to the next event as this activity's processing time. That gap is exactly the wait in front of the next activity. Using it twice would double-count the same days in the bottleneck share.
    ASSUMPTIONS: Events inside a case are ordered by the parsed timestamp, not by file order. A negative gap is clipped to zero. The first event in a case has no wait.
    COMPLEXITY: O(n log n) time to sort, O(n) space.
    FAILS WHEN: Two events in one case share a timestamp. Their order follows the original row order after the time sort (mergesort is stable). A finish column that actually belongs to another stage is still used if it passes the 80% tests.
    """
    end_column = find_end_column(frame, timestamp)
    mode = "start_end" if end_column else "transition"
    note = (
        "Start and end timestamps were both found, so waiting time and processing time are separate."
        if end_column
        else "Only one timestamp per event, so this is transition time, not a split of waiting time and processing time."
    )
    working = frame.dropna(subset=[case_id, activity, timestamp]).copy()
    working[timestamp] = parse_datetimes(working[timestamp])
    working = working.dropna(subset=[timestamp])
    if end_column:
        working["__end"] = parse_datetimes(frame.loc[working.index, end_column])
    working = working.sort_values([case_id, timestamp], kind="mergesort")
    visits: list[Visit] = []
    for case, group in working.groupby(case_id, sort=False):
        previous_end: pd.Timestamp | None = None
        for index, row in group.iterrows():
            start = row[timestamp]
            raw_end = row["__end"] if end_column else pd.NaT
            end = None if pd.isna(raw_end) else raw_end
            if previous_end is None:
                wait = 0.0
            else:
                wait = max(0.0, _days(start, previous_end))
            process = None if end is None else max(0.0, _days(end, start))
            if not end_column:
                previous = group[timestamp].shift(1).loc[index]
                if pd.notna(previous):
                    wait = max(0.0, _days(start, previous))
            resource_value = None
            if resource and resource in group.columns and pd.notna(row[resource]):
                resource_value = str(row[resource])
            visits.append(
                Visit(
                    case=str(case),
                    activity=str(row[activity]),
                    start=start,
                    end=end if end_column else None,
                    resource=resource_value,
                    wait_days=wait,
                    process_days=process,
                )
            )
            previous_end = end if end is not None else start
    return mode, note, visits
