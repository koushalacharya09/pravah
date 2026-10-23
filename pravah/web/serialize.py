"""Turn an ingestion result into plain JSON for the live page."""

from __future__ import annotations

import math

import pandas as pd

from pravah.ingest.constants import DATA_QUALITY_WEIGHTS, QUALITY_WEIGHTS
from pravah.ingest.models import IngestResult, RateReport

_SHAPE_LABELS = {
    "event_log": "Event log",
    "wide_sheet": "Wide sheet",
    "generic_table": "Not a process log",
}
_PREVIEW_ROWS = 12


def _plain(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _components(components: dict[str, RateReport], weights: dict[str, float]) -> list[dict[str, object]]:
    active = {
        name: weights[name]
        for name, report in components.items()
        if report.applicable and name in weights
    }
    total = sum(active.values()) or 1.0
    rows = []
    for name, report in components.items():
        lost = 0.0 if not report.applicable else active.get(name, 0.0) * (100.0 - report.score) / total
        rows.append(
            {
                "name": name,
                "score": None if not report.applicable else round(report.score, 1),
                "detail": report.detail,
                "technical": report.technical,
                "applicable": report.applicable,
                "rough": report.rough,
                "points_lost": round(lost, 2),
            }
        )
    return rows


def _synthetic_columns(result: IngestResult) -> list[str]:
    rebuild = result.reconstruction
    if rebuild is None:
        return []
    return [rebuild.activity_column, rebuild.timestamp_column]


def _source_columns(result: IngestResult, synthetic: list[str]) -> list[str]:
    minted = set(synthetic)
    kept = [str(column) for column in result.frame.columns if str(column) not in minted]
    if result.reconstruction is None:
        return kept
    return kept + [name for name in result.reconstruction.order if name not in kept]


def _stage_links(result: IngestResult) -> list[dict[str, object]]:
    rebuild = result.reconstruction
    if rebuild is None:
        return []
    links = []
    for source, target in zip(rebuild.order, rebuild.order[1:]):
        pair = next(
            (item for item in rebuild.pairs if {item.left, item.right} == {source, target}),
            None,
        )
        if pair is None:
            continue
        share = pair.p_left_before if pair.left == source else pair.p_right_before
        links.append(
            {
                "source": source,
                "target": target,
                "share": round(float(share), 4),
                "support": int(pair.support),
            }
        )
    return links


def to_public_dict(result: IngestResult) -> dict[str, object]:
    """JSON-safe summary. Preview cells are already anonymised by ingestion."""
    frame = result.frame
    preview = frame.head(_PREVIEW_ROWS)
    rows = [
        {str(column): _plain(preview.at[index, column]) for column in preview.columns}
        for index in preview.index
    ]
    order = result.reconstruction.order if result.reconstruction is not None else []
    synthetic = _synthetic_columns(result)
    data_score = round(result.validation.data_quality_score, 1)
    observed = result.validation.n_cases or result.validation.n_rows
    process_score = result.validation.process_readiness_score
    return {
        "narrative": result.narrative,
        "shape": result.shape.shape,
        "shape_label": _SHAPE_LABELS.get(result.shape.shape, result.shape.shape),
        "confidence": round(result.shape.confidence, 4),
        "schema": {
            "case_id": result.schema.case_id,
            "activity": result.schema.activity,
            "timestamp": result.schema.timestamp,
            "resource": result.schema.resource,
        },
        "quality_score": data_score,
        "data_quality_score": data_score,
        "process_readiness_score": None if process_score is None else round(process_score, 1),
        "n_cases": result.validation.n_cases,
        "n_events": result.validation.n_rows,
        "sufficient_for_inference": result.validation.sufficient_for_inference,
        "sample_sentence": result.validation.sample_sentence,
        "sample_observed": observed,
        "sample_required": result.validation.minimum_rows_recommended,
        "sample_unit": "cases" if result.validation.n_cases else "rows",
        "summary": result.validation.summary,
        "components": _components(result.validation.data_components, DATA_QUALITY_WEIGHTS),
        "process_components": _components(result.validation.components, QUALITY_WEIGHTS),
        "stages": list(result.stages),
        "stage_order": order,
        "stage_links": _stage_links(result),
        "synthetic_columns": synthetic,
        "source_columns": _source_columns(result, synthetic),
        "date_column_count": len(order),
        "anonymised_columns": list(result.anonymised_columns),
        "warnings": list(result.warnings),
        "columns": [str(column) for column in frame.columns],
        "preview": rows,
        "n_rows": int(len(frame)),
        "n_preview": len(rows),
        "profile": result.profile or {},
        "bottlenecks": result.bottlenecks or {},
        "survival": result.survival or {},
        "risk": result.risk or {},
        "causes": result.causes or {},
        "simulation": result.simulation or {},
        "plan": result.plan or {},
    }
