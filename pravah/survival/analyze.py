"""Survival report: Kaplan–Meier, log-rank segments, and a checked Cox model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.models import SchemaResult
from pravah.survival.cox import fit_cox
from pravah.survival.events import prepare_cases
from pravah.survival.km import kaplan_meier, window_sentence
from pravah.survival.logrank import compare_segments


def _curve_for(table: pd.DataFrame) -> list[dict[str, float]]:
    fitted = kaplan_meier(table["duration"].to_numpy(), table["event"].to_numpy())
    return fitted["curve"]  # type: ignore[return-value]


def _empty(sentence: str) -> dict[str, object]:
    return {
        "applicable": False,
        "verdict": sentence,
        "event": None,
        "activities": [],
        "curve": [],
        "segments": [],
        "cox": {"fitted": False, "proportional_hazards": None, "sentence": sentence, "ratios": []},
    }


def analyze_survival(
    frame: pd.DataFrame,
    schema: SchemaResult,
    settings: dict[str, object] | None = None,
) -> dict[str, object]:
    """Estimate how long cases stay unfinished, with open cases kept in the curve.

    WHAT: Build one duration per case for either completion or dropout. Fit Kaplan–Meier, name the day range with the highest hazard, compare segment levels with a log-rank test and Benjamini–Hochberg, and fit a Cox model only as far as the proportional-hazards check allows.
    WHY: A regression on the observed number of days treats an open case as a finished one. Survival is the estimator that does not. The page has to say which event the clock is measuring, because completion and dropout are different failures.
    WHY NOT: Declaring the event from an activity name baked into the code. The last-step majority is the default, and the reader can switch it. A parametric Weibull curve was rejected because the Kaplan–Meier curve does not assume a shape the file has not earned.
    ASSUMPTIONS: Settings may set ``event`` to ``completion`` or ``dropout``, ``completion_activity`` to a detected activity, and ``dropout_days`` to a positive number of days. The default silence is 14 days.
    COMPLEXITY: Kaplan–Meier and the log-rank tests are O(n log n). The Cox fit is iterative in the number of segment indicators.
    FAILS WHEN: The sheet has no case and no time. The report says survival does not apply. A file where every case ends on the same step still produces a curve, with no censored cases, and the sentence says so.
    """
    if not (schema.case_id and schema.timestamp and schema.activity):
        return _empty("Survival needs a case, an activity, and a time. Open cases are what censoring is for.")
    chosen_event = "completion"
    chosen_activity = None
    dropout_days = 14.0
    if isinstance(settings, dict):
        if settings.get("event") in {"completion", "dropout"}:
            chosen_event = str(settings["event"])
        if isinstance(settings.get("completion_activity"), str):
            chosen_activity = str(settings["completion_activity"])
        try:
            dropout_days = float(settings.get("dropout_days", 14))
        except (TypeError, ValueError):
            dropout_days = 14.0
    prepared = prepare_cases(
        frame,
        schema,
        event=chosen_event,
        completion_activity=chosen_activity,
        dropout_days=dropout_days,
    )
    if prepared is None:
        return _empty("No case had a readable time, so there is no survival curve.")
    table: pd.DataFrame = prepared["table"]  # type: ignore[assignment]
    durations = table["duration"].to_numpy(dtype=float)
    observed = table["event"].to_numpy(dtype=int)
    fitted = kaplan_meier(durations, observed)
    dropout = prepared["event"] == "dropout"
    window = fitted["window"]
    sentence = window_sentence(window if isinstance(window, dict) else None, dropout=dropout, events=int(fitted["events"]))
    columns = {name: table[name].to_numpy() for name in prepared["segments"]}  # type: ignore[index]
    segments = compare_segments(durations, observed, columns)
    present = {str(segment["column"]) for segment in segments}
    for name in prepared["segments"]:  # type: ignore[union-attr]
        if str(name) in present:
            continue
        segments.append({"column": str(name), "comparisons": [], "curves": []})
    for segment in segments:
        levels = sorted(set(table[str(segment["column"])].astype(str)))
        segment["curves"] = [
            {"level": level, "curve": _curve_for(table[table[str(segment["column"])].astype(str) == level])}
            for level in levels
        ]
    cox = fit_cox(table, list(prepared["segments"]))  # type: ignore[arg-type]
    censored = int(fitted["censored"])
    if censored:
        censor_note = f"{censored} cases are still open, so censoring keeps them in the curve."
    else:
        censor_note = "No case was still open. Every case reached the event, so censoring had nothing to hold."
    if dropout:
        definition = (
            f"Dropout is no event for longer than {prepared['dropout_days']:g} days. "
            "Reaching the completion activity censors the case instead."
        )
    else:
        definition = f"Completion is the first time a case reaches {prepared['completion_activity']}."
    tests = sum(len(segment["comparisons"]) for segment in segments)
    if tests > 1:
        segment_note = f"{tests} segment comparisons were corrected with Benjamini-Hochberg."
    elif tests == 1:
        segment_note = "One segment comparison was made, so no multiple-test correction was applied."
    elif prepared["segments"]:
        segment_note = (
            "Segment columns are drawn, but a level had fewer than 3 cases or the log-rank variance was 0, "
            "so no difference is claimed."
        )
    else:
        segment_note = "No segment column had two groups large enough for a log-rank test."
    return {
        "applicable": True,
        "verdict": sentence,
        "definition": definition,
        "censor_note": censor_note,
        "segment_note": segment_note,
        "event": prepared["event"],
        "completion_activity": prepared["completion_activity"],
        "dropout_days": prepared["dropout_days"],
        "activities": prepared["activities"],
        "events": int(fitted["events"]),
        "censored": censored,
        "curve": fitted["curve"],
        "segments": segments,
        "cox": cox,
    }
