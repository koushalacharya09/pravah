"""Public pages and docs must not carry a product build or last-updated date."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".html", ".js", ".css", ".md", ".txt", ".svg", ".json"}
BUILD_DATE = re.compile(
    r"Last updated|last updated|Date of this report|"
    r"©\s*20\d{2}|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}\b|"
    r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+20\d{2}\b|"
    r"\b20(?:2[5-9]|[3-9]\d)-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


def test_public_and_docs_have_no_build_dates() -> None:
    hits: list[str] = []
    for folder in (ROOT / "public", ROOT / "docs"):
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for match in BUILD_DATE.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}:{match.group(0)}")
    assert hits == []


def test_sample_files_use_a_neutral_calendar() -> None:
    recent = re.compile(r"20(?:2[4-9]|[3-9]\d)")
    hits: list[str] = []
    for path in (ROOT / "data").glob("sample_*"):
        if path.suffix.lower() not in {".csv", ".xes"}:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if recent.search(line):
                hits.append(f"{path.name}:{line_no}")
    assert hits == []
