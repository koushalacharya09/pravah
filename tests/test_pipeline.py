"""End-to-end ingestion of the checked-in samples."""

from __future__ import annotations

from pathlib import Path

from pravah.ingest.pipeline import ingest

DATA = Path(__file__).resolve().parents[1] / "data"


def test_event_log_sample_is_anonymous_and_scored() -> None:
    result = ingest(DATA / "sample_event_log.csv")
    assert result.shape.shape == "event_log"
    assert result.schema.case_id == "student_id"
    assert result.schema.activity == "step"
    assert result.schema.timestamp == "event_time"
    assert result.schema.resource == "tutor"
    assert {"student_name", "phone", "tutor"} <= set(result.anonymised_columns)
    flat = " ".join(result.frame.astype(str).values.ravel())
    assert "9000000001" not in flat
    assert "Asha Rao" not in flat
    assert "9000000001" not in result.narrative
    assert result.validation.quality_score == 100.0
    assert result.validation.sufficient_for_inference is False
    assert "preview" in result.validation.sample_sentence
    assert result.frame["event_time"].isna().sum() == 0


def test_wide_sample_rebuilds_a_stage_order_and_keeps_the_swap_visible() -> None:
    result = ingest(DATA / "sample_wide_sheet.csv")
    assert result.shape.shape == "wide_sheet"
    assert result.reconstruction is not None
    assert result.reconstruction.order == [
        "enquiry_date",
        "demo_date",
        "joining_date",
        "fee_date",
    ]
    assert result.schema.case_id == "student_id"
    assert result.schema.activity == result.reconstruction.activity_column
    assert result.schema.timestamp == result.reconstruction.timestamp_column
    assert result.validation.components["out_of_order"].bad_count == 1
    assert len(result.frame) == 84
    assert "24 cases, 84 events" in result.validation.sample_sentence
    assert result.profile["dates"][0]["method"] == "stage_order"
    assert len(result.profile["dates"]) == 1
    assert "provisional" in result.narrative
    flat = " ".join(result.frame.astype(str).values.ravel())
    assert "9100000001" not in flat


def test_generic_sample_is_not_called_a_process() -> None:
    result = ingest(DATA / "sample_generic.csv")
    assert result.shape.shape == "generic_table"
    assert result.reconstruction is None
    assert result.validation.data_quality_score >= 80
    assert result.validation.process_readiness_score is None
    assert all(not item.applicable for item in result.validation.components.values())
