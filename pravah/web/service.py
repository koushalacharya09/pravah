"""Run ingestion for an upload or a built-in sample. Uploads stay in memory."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pravah.ingest.pipeline import ingest
from pravah.web.serialize import to_public_dict

_MAX_BYTES = 3 * 1024 * 1024
_ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xlsm", ".xes"}
_SAMPLES = {
    "event_log": "sample_event_log.csv",
    "wide_sheet": "sample_wide_sheet.csv",
    "generic": "sample_generic.csv",
    "large": "sample_large.csv",
}


class PayloadError(ValueError):
    """The upload itself is unusable. The message is safe to show."""


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def analyze_path(
    path: Path,
    overrides: dict[str, object] | None = None,
    survival: dict[str, object] | None = None,
    modules: dict[str, object] | None = None,
) -> dict[str, object]:
    return to_public_dict(ingest(path, overrides=overrides, survival=survival, modules=modules))


def analyze_sample(
    name: str,
    overrides: dict[str, object] | None = None,
    survival: dict[str, object] | None = None,
    modules: dict[str, object] | None = None,
) -> dict[str, object]:
    filename = _SAMPLES.get(name)
    if filename is None:
        raise PayloadError("That sample does not exist.")
    source = _data_dir() / filename
    if not source.is_file():
        raise PayloadError("The sample file is missing from this deployment.")
    return analyze_path(source, overrides, survival, modules)


def _checked_upload(filename: str, content: bytes) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise PayloadError("Use a CSV, Excel, or XES file.")
    if not content:
        raise PayloadError("The file is empty.")
    if len(content) > _MAX_BYTES:
        raise PayloadError("This file is over 3 MB. Export a smaller sheet and try again.")


def analyze_upload(
    filename: str,
    content: bytes,
    overrides: dict[str, object] | None = None,
    survival: dict[str, object] | None = None,
    modules: dict[str, object] | None = None,
) -> dict[str, object]:
    _checked_upload(filename, content)
    return to_public_dict(ingest(content=content, filename=filename, overrides=overrides, survival=survival, modules=modules))


def iter_events(
    *,
    sample: str | None = None,
    filename: str | None = None,
    content: bytes | None = None,
    overrides: dict[str, object] | None = None,
    survival: dict[str, object] | None = None,
    modules: dict[str, object] | None = None,
) -> Iterator[str]:
    """Yield one JSON line per real pipeline stage, then the finished result.

    Nothing here sleeps. A line is written when that stage has finished.
    """
    import queue
    import threading

    events: queue.Queue[dict[str, object] | None] = queue.Queue()

    def reporter(event: dict[str, str]) -> None:
        events.put({"type": "stage", **event})

    def run() -> None:
        try:
            if sample:
                filename_for_sample = _SAMPLES.get(sample)
                if filename_for_sample is None:
                    raise PayloadError("That sample does not exist.")
                result = ingest(_data_dir() / filename_for_sample, overrides=overrides, reporter=reporter, survival=survival, modules=modules)
            else:
                if content is None or not filename:
                    raise PayloadError("Choose a file first.")
                _checked_upload(filename, content)
                result = ingest(content=content, filename=filename, overrides=overrides, reporter=reporter, survival=survival, modules=modules)
            events.put({"type": "result", "result": to_public_dict(result)})
        except Exception as exc:
            message = str(exc) if isinstance(exc, (PayloadError, ValueError)) else "The file could not be read."
            events.put({"type": "error", "error": message})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = events.get()
        if item is None:
            break
        yield json.dumps(item) + "\n"
