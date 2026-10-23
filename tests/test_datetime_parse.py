"""Day-month inference and the refusal to parse numbers as dates."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.datetime_parse import infer_dayfirst, parse_datetimes, parse_rate


def test_iso_dates_parse() -> None:
    parsed = parse_datetimes(pd.Series(["2024-01-15", "2024-12-01"]))
    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2024-01-15")


def test_unambiguous_day_first_votes_dmy() -> None:
    column = pd.Series(["20/01/2024", "05/01/2024", "21/02/2024"])
    assert infer_dayfirst(column) == "dmy"
    parsed = parse_datetimes(column)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-20")
    assert parsed.iloc[1] == pd.Timestamp("2024-01-05")


def test_unambiguous_month_first_votes_mdy() -> None:
    column = pd.Series(["01/20/2024", "02/15/2024"])
    assert infer_dayfirst(column) == "mdy"
    assert parse_datetimes(column).iloc[0] == pd.Timestamp("2024-01-20")


def test_conflicting_conventions_drop_ambiguous_rows() -> None:
    column = pd.Series(["20/01/2024", "01/20/2024", "05/06/2024"])
    assert infer_dayfirst(column) == "conflict"
    parsed = parse_datetimes(column)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-20")
    assert parsed.iloc[1] == pd.Timestamp("2024-01-20")
    assert pd.isna(parsed.iloc[2])


def test_integers_are_not_dates() -> None:
    column = pd.Series([1, 2, 44927, 20240101])
    assert parse_rate(column) == 0.0
    assert parse_datetimes(column).isna().all()


def test_parse_rate_ignores_blank_rows() -> None:
    column = pd.Series(["2024-01-01", None, "not a date"])
    assert parse_rate(column) == 0.5
