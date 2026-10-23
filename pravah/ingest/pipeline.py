"""Run the ingestion steps in an order that cannot leak personal data or future events."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pravah.ingest.anonymise import anonymise_frame
from pravah.ingest.constants import DATETIME_PARSE_RATE, DEFAULT_PII_SALT, PRECEDENCE_THRESHOLD
from pravah.ingest.date_format import resolve_date_formats
from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.duplicates import find_fuzzy_duplicates
from pravah.ingest.loader import load_bytes, load_table
from pravah.ingest.models import IngestResult, SchemaResult
from pravah.ingest.narrative import render_narrative
from pravah.ingest.profile import profile_frame
from pravah.ingest.profile_deep import build_profile
from pravah.ingest.reconstruct import reconstruct_wide
from pravah.ingest.schema import detect_schema
from pravah.ingest.shape import detect_shape
from pravah.ingest.validate import validate_event_log

logger = logging.getLogger("pravah.ingest")


def _with_parsed_timestamp(frame: pd.DataFrame, schema: SchemaResult) -> pd.DataFrame:
    if schema.timestamp is None or schema.timestamp not in frame.columns:
        return frame
    parsed = frame.copy()
    parsed[schema.timestamp] = parse_datetimes(parsed[schema.timestamp])
    return parsed


def _chronological(frame: pd.DataFrame, schema: SchemaResult, stage_order: list[str]) -> pd.DataFrame:
    if schema.case_id is None or schema.timestamp is None:
        return frame.reset_index(drop=True)
    ordered = frame.copy()
    rank = {stage: position for position, stage in enumerate(stage_order)}
    if schema.activity is not None and rank:
        ordered["__stage_rank"] = ordered[schema.activity].map(lambda value: rank.get(value, 10**6))
        sort_by = [schema.case_id, schema.timestamp, "__stage_rank"]
    else:
        ordered["__stage_rank"] = 0
        sort_by = [schema.case_id, schema.timestamp]
    ordered = ordered.sort_values(sort_by, kind="mergesort").drop(columns="__stage_rank")
    return ordered.reset_index(drop=True)


def _apply_role(schema: SchemaResult, frame: pd.DataFrame, overrides: dict[str, object] | None, role: str) -> None:
    if not overrides or role not in overrides:
        return
    chosen = overrides[role]
    if chosen in (None, ""):
        setattr(schema, role, None)
    elif isinstance(chosen, str) and chosen in frame.columns:
        setattr(schema, role, chosen)


def ingest(
    path: str | Path | None = None,
    *,
    content: bytes | None = None,
    filename: str | None = None,
    pii_salt: str = DEFAULT_PII_SALT,
    precedence_threshold: float = PRECEDENCE_THRESHOLD,
    sheet: str | int | None = None,
    overrides: dict[str, object] | None = None,
    reporter=None,
    survival: dict[str, object] | None = None,
    modules: dict[str, object] | None = None,
) -> IngestResult:
    """Load one file and return an anonymised, scored event table.

    Personal columns are hashed before shape detection, so a student's name
    cannot be chosen as the activity. Near-duplicate names are compared first,
    while the raw text still exists. Wide sheets are validated in stage order
    (so a backdated cell counts as out of order) and the returned frame is then
    sorted by time for later phases. Long logs keep their file order; the health
    score is the judgement of that order.
    """
    stages: list[dict[str, str]] = []

    def report(stage_id: str, label: str, detail: str) -> None:
        event = {"id": stage_id, "label": label, "detail": detail}
        stages.append(event)
        if reporter is not None:
            reporter(event)

    if content is not None:
        frame, loaded = load_bytes(content, filename or "upload.csv", sheet=sheet)
    else:
        frame, loaded = load_table(path if path is not None else "", sheet=sheet)
    report("read", "Reading file", f"{loaded.n_rows} rows, {loaded.n_columns} columns")
    duplicates = find_fuzzy_duplicates(frame)
    resolved, date_decisions = resolve_date_formats(frame)
    date_detail = date_decisions[0].sentence if date_decisions else "No date columns found"
    report("dates", "Resolving dates", date_detail)
    anonymised, anonymised_columns = anonymise_frame(resolved, salt=pii_salt)
    hashed = ", ".join(anonymised_columns) if anonymised_columns else "none"
    report("hash", "Hashing personal data", f"Hashed {len(anonymised_columns)} column(s): {hashed}")
    shape = detect_shape(anonymised)
    report("shape", "Detecting shape", shape.reasons[0])
    warnings: list[str] = []
    reconstruction = None

    if shape.shape == "wide_sheet":
        wide_schema = detect_schema(
            anonymised, shape="wide_sheet", anonymised_columns=anonymised_columns
        )
        _apply_role(wide_schema, anonymised, overrides, "case_id")
        date_columns = [
            profile.name
            for profile in profile_frame(anonymised)
            if profile.parse_rate >= DATETIME_PARSE_RATE and profile.name != wide_schema.case_id
        ]
        forced = None
        if overrides and isinstance(overrides.get("stage_order"), list):
            forced = [str(item) for item in overrides["stage_order"]]
        if wide_schema.case_id is None or len(date_columns) < 2:
            warnings.append(
                "The sheet looked wide, but it has no case id or fewer than two date columns, "
                "so it was not rebuilt."
            )
            schema = detect_schema(
                anonymised, shape="generic_table", anonymised_columns=anonymised_columns
            )
            working = anonymised
            delivered = _with_parsed_timestamp(working, schema)
        else:
            reconstruction = reconstruct_wide(
                anonymised,
                case_id=wide_schema.case_id,
                date_columns=date_columns,
                threshold=precedence_threshold,
                forced_order=forced,
            )
            schema = detect_schema(
                reconstruction.frame,
                shape="event_log",
                anonymised_columns=anonymised_columns,
            )
            working = reconstruction.frame
            delivered = _chronological(
                _with_parsed_timestamp(working, schema), schema, reconstruction.order
            )
    else:
        schema = detect_schema(
            anonymised,
            shape="event_log" if shape.shape == "event_log" else "generic_table",
            anonymised_columns=anonymised_columns,
        )
        working = anonymised
        delivered = _with_parsed_timestamp(working, schema)

    for role in ("case_id", "activity", "timestamp", "resource"):
        _apply_role(schema, working, overrides, role)
    validation = validate_event_log(working, schema, source=anonymised)
    report(
        "quality",
        "Checking quality",
        f"Data quality {round(validation.data_quality_score, 1)}",
    )
    readiness = "not applicable" if validation.process_readiness_score is None else str(round(validation.process_readiness_score, 1))
    report("readiness", "Building readiness", f"Process readiness {readiness}")
    from pravah.perf.analyze import analyze_bottlenecks
    from pravah.survival.analyze import analyze_survival

    bottlenecks = analyze_bottlenecks(delivered, schema)
    report("bottlenecks", "Finding bottlenecks", str(bottlenecks["verdict"]))
    survival_report = analyze_survival(delivered, schema, survival)
    report("survival", "Estimating survival", str(survival_report.get("verdict", "not applicable")))
    from pravah.causes.analyze import analyze_causes
    from pravah.plan.analyze import build_actions, choose_plan
    from pravah.risk.analyze import analyze_risk
    from pravah.sim.analyze import analyze_simulation

    questions = modules or {}
    risk_report = analyze_risk(delivered, schema)
    report("risk", "Scoring open cases", str(risk_report.get("verdict", "not applicable")))
    causes_report = analyze_causes(delivered, schema, questions.get("causes") if isinstance(questions.get("causes"), dict) else None)
    report("causes", "Estimating causes", str(causes_report.get("verdict", "not applicable")))
    simulation_report = analyze_simulation(delivered, schema, questions.get("simulation") if isinstance(questions.get("simulation"), dict) else None)
    report("whatif", "Simulating what-if", str(simulation_report.get("verdict", "not applicable")))
    plan_settings = questions.get("plan") if isinstance(questions.get("plan"), dict) else {}
    costs = plan_settings.get("costs") if isinstance(plan_settings.get("costs"), dict) else {}
    budget = int(plan_settings.get("budget") or 3)
    actions = build_actions(bottlenecks, causes_report, risk_report, simulation_report)
    plan_report = choose_plan(actions, {str(key): int(value) for key, value in costs.items()}, budget, risk_adjusted=bool(plan_settings.get("risk_adjusted")))
    plan_report["actions"] = actions
    report("plan", "Choosing actions", plan_report["lines"][0] if plan_report.get("lines") else "No action has a computed gain.")
    result = IngestResult(
        frame=delivered,
        load=loaded,
        shape=shape,
        schema=schema,
        reconstruction=reconstruction,
        validation=validation,
        anonymised_columns=anonymised_columns,
        narrative="",
        warnings=warnings,
        stages=stages,
        bottlenecks=bottlenecks,
        survival=survival_report,
        risk=risk_report,
        causes=causes_report,
        simulation=simulation_report,
        plan=plan_report,
    )
    result.profile = build_profile(
        anonymised,
        delivered,
        shape=shape,
        schema=schema,
        reconstruction=reconstruction,
        validation=validation,
        dates=date_decisions,
        duplicates=duplicates,
    )
    result.narrative = render_narrative(result)
    logger.info(
        "Ingested %s as %s with health %.1f",
        loaded.source_name,
        shape.shape,
        validation.quality_score,
    )
    return result
