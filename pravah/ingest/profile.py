"""Column summaries shared by shape detection and schema scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pravah.ingest.datetime_parse import is_numeric, is_stringy, parse_rate


@dataclass(frozen=True)
class ColumnProfile:
    """Cheap facts about one column. Parsing is not stored, to keep this small."""

    name: str
    n_rows: int
    n_non_null: int
    nunique: int
    null_fraction: float
    parse_rate: float
    stringy: bool
    numeric: bool
    mean_repeats: float
    cardinality_ratio: float
    freetext: bool


def _is_freetext(series: pd.Series) -> bool:
    if not is_stringy(series):
        return False
    text = series.dropna().astype(str)
    if text.empty:
        return False
    long_values = float(text.str.len().median()) > 60
    many_words = float(text.str.count(r"\s+").mean()) >= 3
    return long_values or many_words


def profile_frame(frame: pd.DataFrame) -> list[ColumnProfile]:
    """Profile every column. Datetime parse rate is computed once per column."""
    n_rows = len(frame)
    profiles: list[ColumnProfile] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        n_non_null = int(len(non_null))
        nunique = int(non_null.nunique()) if n_non_null else 0
        profiles.append(
            ColumnProfile(
                name=str(name),
                n_rows=n_rows,
                n_non_null=n_non_null,
                nunique=nunique,
                null_fraction=(1.0 if n_rows == 0 else 1.0 - n_non_null / n_rows),
                parse_rate=parse_rate(series) if n_rows else 0.0,
                stringy=is_stringy(series),
                numeric=is_numeric(series),
                mean_repeats=(0.0 if nunique == 0 else n_non_null / nunique),
                cardinality_ratio=(0.0 if n_rows == 0 else nunique / n_rows),
                freetext=_is_freetext(series),
            )
        )
    return profiles


def by_name(profiles: list[ColumnProfile]) -> dict[str, ColumnProfile]:
    """Index profiles by column name. Names are unique after the loader."""
    return {profile.name: profile for profile in profiles}
