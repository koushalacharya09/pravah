"""Candidate treatments and the causal report for one choice."""

from __future__ import annotations

import pandas as pd

from pravah.causes.effect import fit_effect
from pravah.causes.refute import refute
from pravah.ingest.constants import MIN_CASES_FOR_RATE
from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import SchemaResult


def candidate_treatments(frame: pd.DataFrame, schema: SchemaResult) -> list[dict[str, str]]:
    """Treatments the columns support, named from the values that are actually there.

    WHAT: For each activity, offer "reached this activity within the median time from the case start". For each other 0/1 column, offer that column as a treatment. Outcomes are "reached this activity".
    WHY: A coaching file and a hospital file do not share a step called demo. The candidates have to come from the sheet.
    WHY NOT: A fixed menu of demo, discount, and call. Those words are not in every log, and inventing them would ask the model to estimate a treatment nobody recorded.
    ASSUMPTIONS: An activity treatment is 1 when the first time that activity appears is within the median gap from the case start, among cases that do reach it. Cases that never reach it are untreated.
    COMPLEXITY: O(n log n) time. Space O(cases * activities).
    FAILS WHEN: The roles are missing. The list is empty.
    """
    if not (schema.case_id and schema.activity and schema.timestamp):
        return []
    if schema.case_id not in frame.columns or schema.activity not in frame.columns or schema.timestamp not in frame.columns:
        return []
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    offers = []
    for activity in sorted(set(working[schema.activity].astype(str))):
        offers.append({"id": f"within:{activity}", "label": f"Reached {activity} within the median time", "kind": "treatment"})
        offers.append({"id": f"reached:{activity}", "label": f"Reached {activity}", "kind": "outcome"})
    return offers


def build_table(frame: pd.DataFrame, schema: SchemaResult, treatment: str, outcome: str, confounders: list[str]) -> pd.DataFrame:
    """One row per case for the chosen treatment and outcome."""
    if not (schema.case_id and schema.activity and schema.timestamp):
        return pd.DataFrame()
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    rows = []
    grouped = working.sort_values([schema.case_id, schema.timestamp], kind="mergesort")
    activity_name = treatment.split(":", 1)[1] if treatment.startswith("within:") else None
    outcome_name = outcome.split(":", 1)[1] if outcome.startswith("reached:") else None
    gaps = []
    records = []
    for case, group in grouped.groupby(schema.case_id, sort=False):
        start = group[schema.timestamp].iloc[0]
        labels = {str(value): stamp for stamp, value in zip(group[schema.timestamp], group[schema.activity])}
        record: dict[str, object] = {"case": str(case)}
        if activity_name:
            if activity_name in labels:
                gap = (labels[activity_name] - start).total_seconds() / 86400.0
                gaps.append(gap)
                record["gap"] = gap
            else:
                record["gap"] = None
        if outcome_name:
            record[outcome] = 1.0 if outcome_name in labels else 0.0
        for name in confounders:
            if name in group.columns:
                numeric = pd.to_numeric(group[name], errors="coerce")
                if numeric.notna().any():
                    record[name] = float(numeric.median())
                else:
                    record[name] = str(group[name].iloc[0])
        records.append(record)
    median_gap = float(pd.Series(gaps).median()) if gaps else 0.0
    for record in records:
        if activity_name:
            gap = record.pop("gap")
            record[treatment] = 1.0 if gap is not None and float(gap) <= median_gap else 0.0
        rows.append(record)
    table = pd.DataFrame(rows)
    for name in confounders:
        if name not in table.columns:
            continue
        if pd.api.types.is_numeric_dtype(table[name]):
            table[name] = pd.to_numeric(table[name], errors="coerce")
        else:
            table[name] = pd.factorize(table[name].astype(str))[0].astype(float)
    return table


def analyze_causes(frame: pd.DataFrame, schema: SchemaResult, settings: dict[str, object] | None = None) -> dict[str, object]:
    """Estimate one treatment effect and say whether the refutations allow a causal reading.

    WHAT: Offer every activity as a treatment and as an outcome. When the user has chosen both, fit IPW and matching, draw the balance, and run the four refutations. The verdict is likely causal only when all of them pass and every standardised difference after weighting is under 0.1.
    WHY: The diagram is editable because the file cannot know which columns are confounders. The verdict quotes the check that failed so a passing number cannot be read as a cause on its own.
    WHY NOT: Estimating every treatment at once. Each estimate is a bootstrap, and a page full of unrequested effects would look like a search for a significant one.
    ASSUMPTIONS: With no choice on a small sheet, the report lists candidates and does not estimate. On a sheet with at least ``MIN_CASES_FOR_RATE`` cases, the usual second step within its median time is the treatment and reaching the usual last step of the longest cases is the outcome. Confounders then include the resource and other 2–6 level columns unless the request names them.
    COMPLEXITY: Dominated by the bootstrap inside ``fit_effect``.
    FAILS WHEN: The sheet is not a process log, or the chosen treatment has no overlap. The sentence says no effect was estimated.
    """
    offers = candidate_treatments(frame, schema)
    if not offers:
        return {"applicable": False, "verdict": "Causes need a case, an activity, and a time.", "candidates": [], "estimate": None, "refutation": None}
    settings = settings or {}
    treatment = str(settings.get("treatment") or "")
    outcome = str(settings.get("outcome") or "")
    confounders = [str(name) for name in settings.get("confounders", []) if str(name) in frame.columns]
    n_cases = int(frame[schema.case_id].nunique()) if schema.case_id in frame.columns else 0
    if (not treatment or not outcome) and n_cases >= MIN_CASES_FOR_RATE:
        treatment, outcome = _default_ids(frame, schema)
        if not confounders:
            confounders = _default_confounders(frame, schema)
    if not treatment or not outcome:
        return {
            "applicable": True,
            "verdict": "Pick a treatment and an outcome. Nothing has been estimated yet.",
            "candidates": offers,
            "confounders": confounders,
            "estimate": None,
            "refutation": None,
        }
    table = build_table(frame, schema, treatment, outcome, confounders)
    if table.empty or treatment not in table.columns or outcome not in table.columns:
        return {"applicable": True, "verdict": "That treatment and outcome could not be built from this sheet.", "candidates": offers, "estimate": None, "refutation": None}
    estimate = fit_effect(table, treatment, outcome, confounders, draws=120)
    if not estimate.get("estimated"):
        return {
            "applicable": True,
            "verdict": str(estimate.get("sentence")),
            "candidates": offers,
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "estimate": estimate,
            "refutation": None,
        }
    refutation = refute(table, treatment, outcome, confounders, estimate)
    return {
        "applicable": True,
        "verdict": f"{refutation['verdict']}: {refutation['reason']}",
        "candidates": offers,
        "treatment": treatment,
        "outcome": outcome,
        "confounders": confounders,
        "estimate": estimate,
        "refutation": refutation,
    }


def _default_ids(frame: pd.DataFrame, schema: SchemaResult) -> tuple[str, str]:
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    ordered = working.sort_values([schema.case_id, schema.timestamp], kind="mergesort")
    seconds: dict[str, int] = {}
    long_last: dict[str, int] = {}
    longest = 0
    paths: list[list[str]] = []
    for _case, group in ordered.groupby(schema.case_id, sort=False):
        labels = [str(value) for value in group[schema.activity]]
        paths.append(labels)
        longest = max(longest, len(labels))
        if len(labels) >= 2:
            seconds[labels[1]] = seconds.get(labels[1], 0) + 1
    for labels in paths:
        if labels and len(labels) == longest:
            long_last[labels[-1]] = long_last.get(labels[-1], 0) + 1
    treat_act = max(seconds, key=seconds.get) if seconds else ""
    out_act = max(long_last, key=long_last.get) if long_last else ""
    treatment = f"within:{treat_act}" if treat_act else ""
    outcome = f"reached:{out_act}" if out_act else ""
    return treatment, outcome


def _default_confounders(frame: pd.DataFrame, schema: SchemaResult) -> list[str]:
    blocked = {schema.case_id, schema.activity, schema.timestamp}
    names: list[str] = []
    if schema.resource and schema.resource in frame.columns:
        names.append(schema.resource)
    for column in frame.columns:
        name = str(column)
        if name in blocked or name in names:
            continue
        distinct = int(frame[name].nunique(dropna=True))
        if 2 <= distinct <= 6:
            names.append(name)
    return names
