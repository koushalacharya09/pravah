"""Presentation fields for the polish pass. Scores are unchanged."""

from __future__ import annotations

from pathlib import Path

from pravah.ingest.pipeline import ingest
from pravah.web.service import analyze_sample

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_identifier_columns_in_any_position_are_hashed() -> None:
    result = ingest(DATA / "sample_identifiers.csv")
    assert set(result.anonymised_columns) == {
        "mobile_no",
        "linkedin_url",
        "email_address",
        "student_name",
    }
    flat = " ".join(result.frame.astype(str).values.ravel())
    for secret in (
        "Asha Rao",
        "Dev Shah",
        "Meera Lal",
        "asha@example.com",
        "dev@example.com",
        "9000000011",
        "9000000022",
        "linkedin.com/in",
    ):
        assert secret not in flat
    assert result.frame["centre"].tolist() == ["Mansarovar", "Sanganer", "Vaishali"]
    assert result.frame["fee"].tolist() == [5000, 6500, 8000]


def test_sample_adequacy_uses_cases_and_the_formula_floor() -> None:
    wide = analyze_sample("wide_sheet")
    assert wide["sample_observed"] == 24
    assert wide["sample_required"] == 97
    assert wide["sample_unit"] == "cases"
    assert wide["data_quality_score"] == 97.5
    assert wide["process_readiness_score"] == 99.7
    plain = analyze_sample("generic")
    assert plain["sample_observed"] == 5
    assert plain["sample_required"] == 97
    assert plain["sample_unit"] == "rows"
    assert plain["process_readiness_score"] is None


def test_wide_sheet_hides_synthetic_columns_and_reports_vote_shares() -> None:
    body = analyze_sample("wide_sheet")
    assert body["synthetic_columns"] == ["activity", "timestamp"]
    assert "activity" not in body["source_columns"]
    assert "timestamp" not in body["source_columns"]
    assert body["date_column_count"] == 4
    pairs = [(link["source"], link["target"]) for link in body["stage_links"]]
    assert pairs == [
        ("enquiry_date", "demo_date"),
        ("demo_date", "joining_date"),
        ("joining_date", "fee_date"),
    ]
    assert all(0 < link["share"] <= 1 and link["support"] > 0 for link in body["stage_links"])


def test_page_shell_has_product_basics() -> None:
    page = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
    about = (ROOT / "public" / "about.html").read_text(encoding="utf-8")
    assert "How it works" in page
    for label in ("Upload", "Understand", "Find leaks", "Act"):
        assert label in page
    assert "pravah-tour" in page
    assert ">Skip<" in page
    assert "prefers-reduced-motion" in page
    assert "requestAnimationFrame" in page
    assert "Up to 3 MB" in page
    assert "Try a realistic dataset" in page
    assert 'data-sample="large"' in page
    assert 'rel="icon"' in page
    assert "og:image" in page
    assert 'href="/about"' in page
    assert 'href="/links"' in page
    assert "Project links" in page
    assert "playback" not in page
    assert "What it does now" in about
    assert "not built yet" in about
    favicon = ROOT / "public" / "favicon.svg"
    image = ROOT / "public" / "og.png"
    assert favicon.is_file() and image.stat().st_size > 5000
    assert image.read_bytes().startswith(b"\x89PNG")


def test_about_and_favicon_are_served() -> None:
    from app import app

    client = app.test_client()
    assert client.get("/about").status_code == 200
    assert client.get("/learn").status_code == 200
    links = client.get("/links")
    assert links.status_code == 200
    body = links.get_data(as_text=True)
    assert "PRAVAH — Koushal Acharya" in body
    assert "Try a realistic dataset" in body
    assert 'href="/favicon.svg"' in body
    assert "og.png" in body
    assert "prefers-reduced-motion" in body
    assert client.get("/favicon.svg").status_code == 200
    home = client.get("/").get_data(as_text=True)
    assert "og:image" in home
    assert "How it works" in home
