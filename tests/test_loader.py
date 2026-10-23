"""CSV, XLSX, and XES all land as dataframes. XES is checked by hand, not by a miner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pravah.ingest.loader import load_table

DATA = Path(__file__).resolve().parents[1] / "data"


def test_csv_round_trip_counts() -> None:
    frame, meta = load_table(DATA / "sample_event_log.csv")
    assert meta.source_format == "csv"
    assert meta.n_rows == 32
    assert "step" in frame.columns


def test_xlsx_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "sheet.xlsx"
    pd.DataFrame({"case_ref": ["C1", "C1"], "step": ["apply", "done"]}).to_excel(
        source, index=False
    )
    frame, meta = load_table(source)
    assert meta.source_format == "xlsx"
    assert frame["step"].tolist() == ["apply", "done"]


def test_builtin_xes_reader_extracts_the_two_traces() -> None:
    frame, meta = load_table(DATA / "sample_log.xes")
    assert meta.source_format == "xes"
    assert meta.reader in {"builtin-xes", "pm4py"}
    assert set(frame["case:concept:name"]) == {"case-1", "case-2"}
    assert set(frame["concept:name"]) == {"enquiry", "enrol", "trial"}
    assert len(frame) == 4


def test_unknown_extension_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("hello", encoding="utf-8")
    try:
        load_table(source)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")
