"""Prefix features that can only see events up to one moment."""

from __future__ import annotations

import math

import pandas as pd

from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import SchemaResult

_DAY = 86400.0


def features_at(prefix: list[dict[str, object]]) -> dict[str, float]:
    """Numbers known at the end of this prefix, and nothing after it.

    WHAT: Time since the first event, events so far, time since the previous event, how many activities have already repeated, the resource's past load, and the day of week of the latest event. The latest activity is returned separately for the caller to encode.
    WHY: A risk score for an open case is only fair if it uses what a clerk could have seen at that moment. A feature that peeks at the next step is an answer key.
    WHY NOT: Aggregates computed on the finished case, such as the total number of events or the final activity. Those are constant across prefixes and would let the model recognise a finished story.
    ASSUMPTIONS: ``prefix`` is in time order. ``resource_load`` is supplied by the caller and already counts only events at or before this moment. An empty prefix is rejected by the caller.
    COMPLEXITY: O(k) time for a prefix of length k. Space O(1).
    FAILS WHEN: The caller puts a future event into ``prefix`` or into ``resource_load``. This function cannot see a list it was not given.
    """
    first = prefix[0]
    last = prefix[-1]
    start = float(first["time"])
    now = float(last["time"])
    previous = float(prefix[-2]["time"]) if len(prefix) > 1 else now
    seen: dict[str, int] = {}
    for item in prefix:
        name = str(item["activity"])
        seen[name] = seen.get(name, 0) + 1
    rework = sum(count - 1 for count in seen.values() if count > 1)
    moment = pd.Timestamp(now, unit="s")
    return {
        "time_since_start": (now - start) / _DAY,
        "events_so_far": float(len(prefix)),
        "time_since_last": (now - previous) / _DAY,
        "rework_count": float(rework),
        "resource_load": float(last.get("resource_load", 0.0)),
        "day_of_week": float(moment.dayofweek),
        "last_activity": str(last["activity"]),
    }


def _load_at(history: list[tuple[float, str]], moment: float, resource: str) -> float:
    return float(sum(1 for time, name in history if name == resource and time <= moment + 1e-6))


def build_prefixes(frame: pd.DataFrame, schema: SchemaResult, *, completion: str | None, dropout_days: float) -> pd.DataFrame:
    """One training row per event that is not yet the outcome.

    WHAT: Walk each case in time order. A prefix ends at an event strictly before completion. The case label is stored beside the features: whether the case later drops or runs longer than a threshold the caller will set inside each training fold. Remaining time is the outcome clock minus this clock.
    WHY: The label is the future. The features are the past. Keeping them in separate columns is what the leakage test checks.
    WHY NOT: One row per case using the whole log. That row knows the ending, so it cannot score a case that has not ended.
    ASSUMPTIONS: Completion is the supplied activity. Dropout is silence longer than ``dropout_days`` before the file's last timestamp, and completion censors dropout. Resource load counts every event of that resource at or before the prefix, including other cases.
    COMPLEXITY: O(n log n) time to sort, O(n) space.
    FAILS WHEN: The roles are missing. The frame comes back empty. A completion activity that is the first step of every case leaves no prefix.
    """
    if not (schema.case_id and schema.activity and schema.timestamp):
        return pd.DataFrame()
    if schema.case_id not in frame.columns or schema.activity not in frame.columns or schema.timestamp not in frame.columns:
        return pd.DataFrame()
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    if working.empty:
        return pd.DataFrame()
    resource = schema.resource if schema.resource in working.columns else None
    ordered = working.sort_values([schema.timestamp, schema.case_id], kind="mergesort")
    history: list[tuple[float, str]] = []
    for _, row in ordered.iterrows():
        if resource and pd.notna(row[resource]):
            history.append((row[schema.timestamp].timestamp(), str(row[resource])))
    study_end = ordered[schema.timestamp].max()
    records: list[dict[str, object]] = []
    grouped = working.sort_values([schema.case_id, schema.timestamp], kind="mergesort")
    for case, group in grouped.groupby(schema.case_id, sort=False):
        stamps = list(group[schema.timestamp])
        labels = [str(value) for value in group[schema.activity]]
        resources = [str(value) if resource and pd.notna(value) else "" for value in (group[resource] if resource else [""] * len(group))]
        origin = stamps[0]
        completion_at = None
        if completion:
            for stamp, label in zip(stamps, labels):
                if label == completion:
                    completion_at = stamp
                    break
        if completion_at is not None:
            outcome = completion_at
            dropped = 0
            open_case = 0
            duration = (completion_at - origin).total_seconds() / _DAY
        else:
            silence = (study_end - stamps[-1]).total_seconds() / _DAY
            duration = (stamps[-1] - origin).total_seconds() / _DAY
            if silence > dropout_days:
                outcome = stamps[-1] + pd.Timedelta(float(dropout_days), unit="D")
                dropped = 1
                open_case = 0
                duration = (outcome - origin).total_seconds() / _DAY
            else:
                outcome = study_end
                dropped = 0
                open_case = 1
        prefix: list[dict[str, object]] = []
        for stamp, label, who in zip(stamps, labels, resources):
            if completion_at is not None and stamp >= completion_at:
                break
            moment = float(stamp.timestamp())
            prefix.append(
                {
                    "time": moment,
                    "activity": label,
                    "resource_load": _load_at(history, moment, who) if who else 0.0,
                }
            )
            numbers = features_at(prefix)
            records.append(
                {
                    "case": str(case),
                    "start": origin.timestamp(),
                    "dropped": dropped,
                    "open": open_case,
                    "duration": max(0.0, float(duration)),
                    "remaining": max(0.0, (outcome - stamp).total_seconds() / _DAY),
                    **{key: value for key, value in numbers.items()},
                }
            )
    return pd.DataFrame(records)
