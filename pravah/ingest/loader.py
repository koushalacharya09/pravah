"""Read CSV, Excel, and XES into a dataframe. XES prefers pm4py when it is installed."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from pravah.ingest.models import LoadResult

logger = logging.getLogger("pravah.ingest.loader")


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for name in columns:
        seen[name] = seen.get(name, 0) + 1
        deduped.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return deduped


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = _dedupe_columns([str(column).strip() for column in cleaned.columns])
    for name in cleaned.columns:
        if cleaned[name].dtype == object:
            blank = cleaned[name].astype("string").str.strip().eq("")
            cleaned.loc[blank.fillna(False), name] = pd.NA
    return cleaned


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_xes_builtin(source: Path | io.BytesIO) -> pd.DataFrame:
    """Read the trace id and event attributes of a XES file.

    This is an XML reader, not a process-mining algorithm. pm4py is used when
    it is installed; this path keeps XES ingestion working without that stack.
    """
    root = ElementTree.parse(source).getroot()
    rows: list[dict[str, object]] = []
    for trace in root.iter():
        if _local_name(trace.tag) != "trace":
            continue
        case_id = None
        for child in list(trace):
            if _local_name(child.tag) == "event":
                continue
            if child.attrib.get("key") == "concept:name":
                case_id = child.attrib.get("value")
        for event in list(trace):
            if _local_name(event.tag) != "event":
                continue
            row: dict[str, object] = {"case:concept:name": case_id}
            for field in list(event):
                key = field.attrib.get("key")
                if key is None:
                    continue
                row[key] = field.attrib.get("value")
            rows.append(row)
    if not rows:
        name = getattr(source, "name", "upload.xes")
        raise ValueError(f"No events were found in XES file {name}.")
    return pd.DataFrame(rows)


def load_table(path: str | Path, *, sheet: str | int | None = None) -> tuple[pd.DataFrame, LoadResult]:
    """Load a CSV, XLSX, or XES file.

    WHAT: CSV is read with pandas, XLSX with openpyxl through pandas, and XES
    with pm4py when that package imports. Otherwise a small XML reader extracts
    ``case:concept:name`` and each event's attributes. Column names are stripped
    and de-duplicated. Blank strings become missing.

    WHY: The three formats are the ones founders actually export: a sheet, a
    workbook, and an event log from a process-mining tool. Keeping pm4py
    optional means the core install stays on NumPy and pandas, which is the
    stack the algorithms are written against.

    WHY NOT: Using pm4py for CSV as well is rejected because the spec allows it
    only for XES, and pandas already reads CSV and Excel faithfully. Writing
    only the builtin XES reader is rejected as the sole path because pm4py is
    the permitted reference for XES and is preferred when present.

    ASSUMPTIONS: The file extension is the format. XLSX uses the first sheet
    unless ``sheet`` is passed. XES event attributes are flat strings, which
    is the common export. A trace's case id is its ``concept:name``.

    COMPLEXITY: ``O(n m)`` time and space for the resulting table.

    FAILS WHEN: The extension is wrong, the XES file has no events, or an XLSX
    sheet name does not exist. Those raise ``ValueError``. A XES file that uses
    nested containers beyond one level of event attributes keeps only the first
    level. pm4py and the builtin reader can disagree on extra classifier columns;
    schema detection uses values, so a renamed helper column does not change roles.
    """
    source = Path(path)
    if not source.exists():
        raise ValueError(f"File not found: {source}")
    suffix = source.suffix.lower()
    reader = "pandas"
    if suffix == ".csv":
        frame = pd.read_csv(source)
        source_format = "csv"
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(source, sheet_name=0 if sheet is None else sheet)
        source_format = "xlsx"
    elif suffix == ".xes":
        source_format = "xes"
        try:
            import pm4py
        except ImportError:
            logger.info("pm4py is not installed; reading XES with the built-in parser")
            frame = _read_xes_builtin(source)
            reader = "builtin-xes"
        else:
            logger.info("Reading XES with pm4py")
            frame = pm4py.read_xes(str(source))
            reader = "pm4py"
    else:
        raise ValueError(
            f"Unsupported file type {suffix or '(none)'}. Use CSV, XLSX, or XES."
        )
    frame = _clean_frame(frame)
    meta = LoadResult(
        source_format=source_format,
        source_name=source.name,
        n_rows=int(len(frame)),
        n_columns=int(len(frame.columns)),
        reader=reader,
    )
    logger.info(
        "Loaded %s rows and %s columns from %s via %s",
        meta.n_rows,
        meta.n_columns,
        meta.source_name,
        meta.reader,
    )
    return frame, meta


def load_bytes(
    content: bytes,
    filename: str,
    *,
    sheet: str | int | None = None,
) -> tuple[pd.DataFrame, LoadResult]:
    """Load an upload from memory. The bytes are not written to disk."""
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(content)
    reader = "pandas"
    if suffix == ".csv":
        frame = pd.read_csv(buffer)
        source_format = "csv"
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(buffer, sheet_name=0 if sheet is None else sheet)
        source_format = "xlsx"
    elif suffix == ".xes":
        frame = _read_xes_builtin(buffer)
        source_format = "xes"
        reader = "builtin-xes"
    else:
        raise ValueError(f"Unsupported file type {suffix or '(none)'}. Use CSV, XLSX, or XES.")
    frame = _clean_frame(frame)
    meta = LoadResult(
        source_format=source_format,
        source_name=Path(filename).name,
        n_rows=int(len(frame)),
        n_columns=int(len(frame.columns)),
        reader=reader,
    )
    logger.info(
        "Loaded %s rows and %s columns from an in-memory %s",
        meta.n_rows,
        meta.n_columns,
        meta.source_format,
    )
    return frame, meta
