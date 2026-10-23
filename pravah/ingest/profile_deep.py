"""Attach the phase-1.5 checks to one ingestion result."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from pravah.ingest.date_format import DateFormatDecision
from pravah.ingest.models import ReconstructionResult, SchemaResult, ShapeResult, ValidationResult
from pravah.ingest.readiness import analysis_verdict, readiness_matrix
from pravah.stats.benford import benford_report
from pravah.stats.missingness import missingness_report
from pravah.stats.outliers import box_summaries, outlier_report
from pravah.stats.relationships import relationship_matrix


def build_profile(
    source: pd.DataFrame,
    delivered: pd.DataFrame,
    *,
    shape: ShapeResult,
    schema: SchemaResult,
    reconstruction: ReconstructionResult | None,
    validation: ValidationResult,
    dates: list[DateFormatDecision],
    duplicates: list[dict[str, object]],
) -> dict[str, object]:
    """Run the deep checks on the sheet before it is melted into events.

    Missing stage dates only exist on that sheet. The melted log has already dropped them.
    """
    verdict = analysis_verdict(
        validation.data_quality_score,
        validation.sufficient_for_inference,
        process_ready=validation.process_readiness_score is not None,
    )
    date_rows = [asdict(decision) for decision in dates] or [
        {
            "column": None,
            "choice": "none",
            "confidence": None,
            "method": "none",
            "sentence": "No date columns found.",
        }
    ]
    return {
        "dates": date_rows,
        "duplicates": duplicates,
        "missingness": missingness_report(source),
        "outliers": outlier_report(source),
        "boxes": box_summaries(source),
        "missing_rates": [
            {"column": str(name), "missing": round(float(source[name].isna().mean()), 3)}
            for name in source.columns
        ],
        "relationships": relationship_matrix(source),
        "benford": benford_report(source),
        "readiness": readiness_matrix(
            delivered,
            shape=shape,
            schema=schema,
            reconstruction=reconstruction,
            validation=validation,
        ),
        "verdict": verdict,
    }
