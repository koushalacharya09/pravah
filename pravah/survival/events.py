"""Turn the event table into one duration and one censoring flag per case."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import SchemaResult

DEFAULT_DROPOUT_DAYS = 14.0


def _modal_last_activity(frame: pd.DataFrame, case_id: str, activity: str, timestamp: str) -> str | None:
    ordered = frame.sort_values([case_id, timestamp], kind="mergesort")
    last = ordered.groupby(case_id, sort=False).tail(1)
    if last.empty:
        return None
    mode = last[activity].astype(str).mode()
    return None if mode.empty else str(mode.iloc[0])


def prepare_cases(
    frame: pd.DataFrame,
    schema: SchemaResult,
    *,
    event: str,
    completion_activity: str | None,
    dropout_days: float,
) -> dict[str, object] | None:
    """One row per case: days at risk, whether the chosen event happened, and segments.

    WHAT: Completion time is the first time the chosen activity appears. A case that never reaches it is censored at its last event. Dropout is silence longer than ``dropout_days`` before the latest timestamp in the file. The clock starts at the case's first event. Reaching the completion activity censors a dropout, because the case finished instead of disappearing.
    WHY: The same log answers two different questions. Time to enrol is not time to give up. A fixed threshold is a decision the reader can change, not a fact hidden in the estimator.
    WHY NOT: Calling every last event a failure. That deletes censoring, which is the reason to use survival rather than a regression on the observed days. Using today's wall clock as the end of the study was rejected because the file, not the server clock, is the observation window.
    ASSUMPTIONS: The completion activity defaults to the activity that most often ends a case. Dropout days default to 14 and must sit in ``(0, 3650]``. A segment value is the most common value of that column inside the case. Columns with 2 to 6 distinct values are segments. The case, activity, and time columns are not segments.
    COMPLEXITY: O(n log n) time to sort, O(n) space.
    FAILS WHEN: No case has both an id and a readable time. The caller then reports that survival does not apply. A completion activity that is also the first step gives a pile of zero durations.
    """
    if not (schema.case_id and schema.activity and schema.timestamp):
        return None
    if schema.case_id not in frame.columns or schema.activity not in frame.columns or schema.timestamp not in frame.columns:
        return None
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    if working.empty:
        return None
    activities = sorted(working[schema.activity].astype(str).unique().tolist())
    chosen = completion_activity if completion_activity in activities else _modal_last_activity(
        working, schema.case_id, schema.activity, schema.timestamp
    )
    days = dropout_days if 0 < dropout_days <= 3650 else DEFAULT_DROPOUT_DAYS
    study_end = working[schema.timestamp].max()
    blocked = {schema.case_id, schema.activity, schema.timestamp}
    segment_names = []
    if schema.resource and schema.resource in working.columns:
        segment_names.append(schema.resource)
    for column in working.columns:
        if column in blocked or column in segment_names:
            continue
        values = working[column].dropna()
        if values.empty:
            continue
        levels = values.astype(str).nunique()
        if 2 <= levels <= 6:
            segment_names.append(str(column))
    rows: list[dict[str, object]] = []
    ordered = working.sort_values([schema.case_id, schema.timestamp], kind="mergesort")
    for case, group in ordered.groupby(schema.case_id, sort=False):
        stamps = group[schema.timestamp]
        origin = stamps.iloc[0]
        last = stamps.iloc[-1]
        labels = group[schema.activity].astype(str)
        completion_at = stamps[labels == chosen].min() if chosen else pd.NaT
        if event == "dropout":
            if pd.notna(completion_at):
                duration = (completion_at - origin).total_seconds() / 86400.0
                happened = 0
            else:
                silence = (study_end - last).total_seconds() / 86400.0
                if silence > days:
                    duration = (last - origin).total_seconds() / 86400.0 + days
                    happened = 1
                else:
                    duration = (study_end - origin).total_seconds() / 86400.0
                    happened = 0
        else:
            if pd.notna(completion_at):
                duration = (completion_at - origin).total_seconds() / 86400.0
                happened = 1
            else:
                duration = (last - origin).total_seconds() / 86400.0
                happened = 0
        record: dict[str, object] = {"duration": max(0.0, float(duration)), "event": int(happened)}
        for name in segment_names:
            values = group[name].dropna().astype(str)
            record[name] = values.mode().iloc[0] if not values.empty else ""
        rows.append(record)
    return {
        "table": pd.DataFrame(rows),
        "activities": activities,
        "completion_activity": chosen,
        "dropout_days": days,
        "event": "dropout" if event == "dropout" else "completion",
        "segments": segment_names[:4],
    }
