"""The glossary is the only definition of a term the page is allowed to use."""

from __future__ import annotations

import json
from pathlib import Path

from pravah.glossary import GLOSSARY, MODULES, glossary_payload, unglossed_mentions
from pravah.web.service import analyze_sample

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "case",
    "event",
    "activity",
    "variant",
    "directly-follows graph",
    "throughput time",
    "waiting time",
    "bottleneck",
    "Little's Law",
    "rework loop",
    "edit distance",
    "censoring",
    "Kaplan-Meier",
    "hazard",
    "Markov chain",
    "calibration",
    "conformal interval",
    "SHAP value",
    "confounder",
    "propensity score",
    "IPW",
    "refutation test",
    "discrete-event simulation",
    "knapsack",
    "MCAR",
    "Cramér's V",
    "Spearman",
    "Benford's Law",
    "confidence interval",
    "credible interval",
    "p-value",
    "Benjamini-Hochberg",
    "PSI",
    "sample adequacy",
)

# Spellings that show up in the page or in engine sentences. Each must resolve.
WATCH = REQUIRED + (
    "Cramer's V",
    "Benford",
    "Benford's law",
    "Wilson",
    "modified z-score",
    "correlation ratio",
    "mutual information",
    "SHAP",
    "cases",
    "events",
    "not MCAR",
    "likely MCAR",
    "Benjamini–Hochberg",
)


def test_every_required_term_has_the_four_fields() -> None:
    modules = {name for name, _label in MODULES}
    for phrase in REQUIRED:
        assert phrase in GLOSSARY, phrase
        term = GLOSSARY[phrase]
        assert term.plain and term.technical and term.why_used and term.example
        assert term.module in modules
    assert "Cramer's V" in GLOSSARY["Cramér's V"].aliases


def test_payload_keeps_one_row_per_term() -> None:
    rows = glossary_payload()
    assert len(rows) == len(GLOSSARY)
    assert {row["phrase"] for row in rows} == set(GLOSSARY)
    assert rows[0]["module"] == "process"


def test_terms_used_on_the_page_and_in_sample_copy_are_defined() -> None:
    parts = [
        (ROOT / "public" / "index.html").read_text(encoding="utf-8"),
        (ROOT / "public" / "learn.html").read_text(encoding="utf-8"),
    ]
    for sample in ("event_log", "wide_sheet", "generic"):
        parts.append(json.dumps(analyze_sample(sample)))
    missing = unglossed_mentions("\n".join(parts), WATCH)
    assert missing == []


def test_an_unknown_spelling_is_reported() -> None:
    assert unglossed_mentions("The sheet uses a Kaplan Meier curve.", ["Kaplan Meier"]) == ["Kaplan Meier"]


def test_learn_page_and_glossary_route() -> None:
    from app import app

    client = app.test_client()
    page = client.get("/learn")
    assert page.status_code == 200
    assert b"Learn the terms" in page.data
    body = client.get("/api/glossary")
    assert body.status_code == 200
    assert len(body.get_json()) == len(GLOSSARY)
