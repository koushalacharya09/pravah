"""Parse datetimes without mistaking case ids and amounts for dates."""

from __future__ import annotations

import re

import pandas as pd

_SLASH_DATE = re.compile(r"^\s*(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})")
_ISO_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")
_YEAR_FIRST = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_MISSING_TEXT = {"", "nat", "none", "null", "nan", "na", "-"}


def is_stringy(series: pd.Series) -> bool:
    """True for object or string columns. Numbers are not strings."""
    return bool(
        pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    )


def is_numeric(series: pd.Series) -> bool:
    """True for bools and numbers. Those columns are not parsed as dates."""
    return bool(
        pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)
    )


def infer_dayfirst(series: pd.Series) -> str:
    """Decide day-month order from the unambiguous values in the column.

    Returns ``dmy``, ``mdy``, ``iso``, or ``conflict``.
    """
    if not is_stringy(series):
        return "iso"
    dmy_votes = 0
    mdy_votes = 0
    iso_votes = 0
    for raw in series.dropna():
        text = re.split(r"[T\s]", str(raw).strip(), maxsplit=1)[0]
        if _YEAR_FIRST.match(text) or _ISO_DATE.match(text):
            iso_votes += 1
            continue
        match = _SLASH_DATE.match(text)
        if match is None:
            continue
        first, second, _year = (int(part) for part in match.groups())
        if len(match.group(1)) == 4:
            iso_votes += 1
            continue
        if first > 12 and second <= 12:
            dmy_votes += 1
        elif second > 12 and first <= 12:
            mdy_votes += 1
    if dmy_votes and mdy_votes:
        return "conflict"
    if dmy_votes:
        return "dmy"
    if mdy_votes:
        return "mdy"
    return "iso"


def _empty_datetimes(series: pd.Series) -> pd.Series:
    return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")


def _parse_with_dayfirst(text: pd.Series, *, dayfirst: bool) -> pd.Series:
    try:
        return pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=dayfirst)
    except (ValueError, TypeError):
        return pd.to_datetime(
            text, errors="coerce", format="mixed", dayfirst=dayfirst, utc=True
        ).dt.tz_localize(None)


def _parse_conflict(text: pd.Series) -> pd.Series:
    """Parse unambiguous tokens. Slash dates with both parts <= 12 become missing."""

    def one(value: object) -> pd.Timestamp:
        if value is pd.NA or value is None:
            return pd.NaT
        raw = str(value).strip()
        if raw.lower() in _MISSING_TEXT:
            return pd.NaT
        if _ISO_DATE.match(raw):
            parsed = pd.to_datetime(raw, errors="coerce")
            return pd.NaT if pd.isna(parsed) else parsed
        match = _SLASH_DATE.match(raw)
        if match is None:
            parsed = pd.to_datetime(raw, errors="coerce")
            return pd.NaT if pd.isna(parsed) else parsed
        first, second, _year = (int(part) for part in match.groups())
        if len(match.group(1)) == 4:
            parsed = pd.to_datetime(raw, errors="coerce")
            return pd.NaT if pd.isna(parsed) else parsed
        if first > 12 and second <= 12:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        elif second > 12 and first <= 12:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=False)
        else:
            return pd.NaT
        return pd.NaT if pd.isna(parsed) else parsed

    return pd.to_datetime(text.map(one), errors="coerce")


def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse a column to datetimes, or return all-missing for non-dates.

    WHAT: A datetime column is returned coerced. A numeric column becomes all
    missing. A string column is parsed after a day-month vote: if any value
    has a day token above 12 and none has a month token above 12, parse
    day-first; the mirror image parses month-first; a mixture parses only the
    unambiguous values. With no vote, ISO and month-first numeric dates are used.

    WHY: Uploads from India often use day-first dates, and exports often use
    ISO. The vote reads the convention off the column instead of assuming one
    country. Numeric columns are refused so a case id or a price cannot become
    1970.

    WHY NOT: ``dayfirst=True`` globally would misread US exports and ISO-like
    strings that pandas already handles. Excel serial numbers are rejected
    rather than accepted: openpyxl already turns real Excel dates into
    datetimes, and an integer column of serials is indistinguishable from an
    id column once it is detached from the workbook.

    ASSUMPTIONS: A single column uses one convention, except in a conflict,
    where ambiguous rows are dropped rather than guessed. Values are dates a
    person could type, not Unix epochs stored as integers.

    COMPLEXITY: O(n) time, O(n) space for a column of length n.

    FAILS WHEN: Every date is ambiguous (``05/06/2024`` with no 13-31 anywhere)
    and the sheet is day-first. Those values are read month-first. Pure numeric
    Excel serials are left unparsed. Mixed time zones are stripped to naive
    timestamps after a UTC conversion.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            return parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        return parsed
    if is_numeric(series) or not is_stringy(series):
        return _empty_datetimes(series)

    text = series.astype("string").str.strip()
    mode = infer_dayfirst(text)
    if mode == "conflict":
        parsed = _parse_conflict(text)
    else:
        parsed = _parse_with_dayfirst(text, dayfirst=(mode == "dmy"))
    missing = text.isna() | text.str.lower().isin(_MISSING_TEXT)
    return parsed.mask(missing)


def parse_rate(series: pd.Series) -> float:
    """Fraction of non-missing values that parse as datetimes. Empty columns score 0."""
    if pd.api.types.is_datetime64_any_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return 0.0
        return 1.0
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    parsed = parse_datetimes(non_null)
    return float(parsed.notna().mean())
