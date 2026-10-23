"""Regression tests for the phase-1.5 score, privacy, date, and unit bugs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pravah.ingest.anonymise import anonymise_frame
from pravah.ingest.date_format import resolve_date_formats
from pravah.ingest.pipeline import ingest
from pravah.stats.relationships import relationship_matrix
from pravah.web.service import analyze_upload
from tests.test_pipeline import DATA


def test_clean_plain_table_scores_high_and_skips_process_checks() -> None:
    result = ingest(DATA / "sample_generic.csv")
    assert result.validation.data_quality_score >= 80
    assert result.validation.process_readiness_score is None
    assert result.validation.quality_score == result.validation.data_quality_score
    for component in result.validation.components.values():
        assert component.applicable is False
        assert component.detail.startswith("Not applicable")
    assert "Data quality is" in result.validation.summary
    assert "100 out of 100" not in result.validation.summary or "100.0" in result.validation.summary


def test_names_emails_and_profile_urls_are_hashed() -> None:
    frame = pd.DataFrame(
        {
            "From": ["Rahul Sharma", "Meera Lal"],
            "To": ["Asha Rao", "Dev Shah"],
            "Profile URL": [
                "https://www.linkedin.com/in/rahul-sharma",
                "https://www.linkedin.com/in/meera-lal",
            ],
            "Email Address": ["rahul@example.com", "meera@example.com"],
            "Company": ["Northwind", "Contoso"],
        }
    )
    hashed, columns = anonymise_frame(frame, salt="test-salt")
    assert set(columns) == {"From", "To", "Profile URL", "Email Address"}
    flat = " ".join(hashed.astype(str).values.ravel())
    assert "Rahul" not in flat
    assert "linkedin.com/in" not in flat
    assert "rahul@example.com" not in flat
    assert hashed["Company"].tolist() == ["Northwind", "Contoso"]


def test_upload_is_not_written_with_path_write_bytes(monkeypatch) -> None:
    calls: list[Path] = []

    def trap(self: Path, data: bytes) -> int:
        calls.append(self)
        raise AssertionError("upload was written to disk")

    monkeypatch.setattr(Path, "write_bytes", trap)
    body = analyze_upload("items.csv", b"item,price,quantity\nPencil,10,3\nNotebook,45,2\n")
    assert calls == []
    assert body["data_quality_score"] >= 80
    assert "Pencil" in str(body["preview"])


def test_iso_timestamps_skip_day_month_logic() -> None:
    frame = pd.DataFrame({"connected_on": ["2010-08-15T02:26:00"] * 12})
    _resolved, decisions = resolve_date_formats(frame)
    assert len(decisions) == 1
    assert decisions[0].choice == "iso"
    assert decisions[0].confidence == 1.0
    assert "month/day" not in decisions[0].sentence
    assert "above 12" not in decisions[0].sentence


def test_hard_day_evidence_is_full_confidence_despite_ambiguous_rows() -> None:
    frame = pd.DataFrame({"when": ["20/01/2024", "05/06/2024", "03/04/2024", "21/02/2024"]})
    _resolved, decisions = resolve_date_formats(frame)
    assert decisions[0].method == "impossible_day"
    assert decisions[0].choice == "dmy"
    assert decisions[0].confidence == 1.0


def test_price_and_quantity_are_scored_with_a_small_sample_warning() -> None:
    frame = pd.read_csv(DATA / "sample_generic.csv")
    matrix = relationship_matrix(frame)
    pair = next(cell for cell in matrix["cells"] if {cell["left"], cell["right"]} == {"price", "quantity"})
    assert pair["measure"] == "spearman"
    assert pair["warning"] == "n=5, very low confidence"
    assert "No pair of columns varies" not in matrix["sentence"]
