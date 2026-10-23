"""The live page returns the same ingestion result, with personal values already hashed."""

from __future__ import annotations

from pathlib import Path

import pytest

from pravah.web.service import PayloadError, analyze_sample, analyze_upload

DATA = Path(__file__).resolve().parents[1] / "data"


def test_sample_event_log_is_safe_to_show() -> None:
    body = analyze_sample("event_log")
    assert body["shape"] == "event_log"
    assert body["quality_score"] == 100.0
    assert body["schema"]["case_id"] == "student_id"
    flat = str(body["preview"])
    assert "9000000001" not in flat
    assert "Asha Rao" not in flat
    assert "anon:" in flat


def test_uploaded_csv_matches_the_sample() -> None:
    content = (DATA / "sample_wide_sheet.csv").read_bytes()
    body = analyze_upload("sheet.csv", content)
    assert body["shape"] == "wide_sheet"
    assert body["stage_order"] == [
        "enquiry_date",
        "demo_date",
        "joining_date",
        "fee_date",
    ]
    assert "Rahul" not in str(body)


def test_unknown_sample_and_bad_suffix_are_rejected() -> None:
    with pytest.raises(PayloadError):
        analyze_sample("../sample_event_log.csv")
    with pytest.raises(PayloadError):
        analyze_upload("notes.txt", b"hello")
    with pytest.raises(PayloadError):
        analyze_upload("empty.csv", b"")
