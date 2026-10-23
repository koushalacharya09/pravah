"""Template sentences for an ingestion result. Every number is computed upstream."""

from __future__ import annotations

from pravah.ingest.models import IngestResult


def _schema_sentence(result: IngestResult) -> str:
    schema = result.schema
    parts: list[str] = []
    if schema.case_id:
        parts.append(f"{schema.case_id} as the case")
    if schema.activity:
        parts.append(f"{schema.activity} as the activity")
    if schema.timestamp:
        parts.append(f"{schema.timestamp} as the time")
    if schema.resource:
        parts.append(f"{schema.resource} as the resource")
    if not parts:
        return "No case, activity, or time column cleared the score cutoff."
    sentence = "I read " + ", ".join(parts) + "."
    if result.anonymised_columns:
        names = ", ".join(result.anonymised_columns)
        sentence += f" These columns were hashed before that read, because they look like personal data: {names}."
    return sentence


def _reconstruction_sentence(result: IngestResult) -> str | None:
    rebuild = result.reconstruction
    if rebuild is None:
        return None
    if not rebuild.order:
        return "The date columns could not be ordered."
    sequence = ", then ".join(rebuild.order)
    sentence = f"The stage order is {sequence}."
    if rebuild.dropped_edges:
        dropped = rebuild.dropped_edges[0]
        sentence += (
            f" One link was removed because the dates cycled: "
            f"{dropped.source} before {dropped.target} "
            f"({dropped.probability:.0%} of {dropped.support} comparable rows)."
        )
    provisional = [pair for pair in rebuild.pairs if pair.kept_direction and pair.provisional]
    if provisional:
        sentence += (
            f" {len(provisional)} kept link(s) rest on fewer than 73 rows with both dates filled, "
            "so the order is provisional."
        )
    return sentence


def render_narrative(result: IngestResult) -> str:
    """Plain-language account of one ingestion. No model writes these sentences."""
    lines = [
        result.shape.reasons[0],
        _schema_sentence(result),
    ]
    reconstruction = _reconstruction_sentence(result)
    if reconstruction:
        lines.append(reconstruction)
    lines.append(result.validation.summary)
    lines.append(result.validation.sample_sentence)
    profile = result.profile or {}
    verdict = profile.get("verdict")
    if isinstance(verdict, dict) and verdict.get("reason"):
        lines.append(f"Safe to analyse? {verdict.get('label')}. {verdict.get('reason')}")
    dates = profile.get("dates")
    if isinstance(dates, list) and dates:
        first = dates[0]
        if isinstance(first, dict) and first.get("sentence"):
            lines.append(str(first["sentence"]))
    for warning in result.warnings:
        lines.append(warning)
    return " ".join(lines)
