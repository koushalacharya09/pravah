"""Shape heuristics on obvious logs, obvious wide sheets, and the two-date boundary."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.shape import detect_shape


def _event_log() -> pd.DataFrame:
    rows = []
    for case in range(6):
        for step_index, step in enumerate(["apply", "review", "offer"]):
            rows.append(
                {
                    "case_ref": f"C{case}",
                    "step": step,
                    "when": f"2024-02-{case + 1:02d}T0{step_index}:00:00",
                    "owner": "Ada" if step_index == 0 else "Ben",
                }
            )
    return pd.DataFrame(rows)


def test_classic_event_log() -> None:
    result = detect_shape(_event_log())
    assert result.shape == "event_log"
    assert result.confidence >= 0.8


def test_three_stage_dates_are_a_wide_sheet() -> None:
    frame = pd.DataFrame(
        {
            "student_id": [f"S{i:02d}" for i in range(10)],
            "enquiry_date": [f"2024-01-{i + 1:02d}" for i in range(10)],
            "trial_date": [f"2024-01-{i + 11:02d}" for i in range(10)],
            "enrol_date": [f"2024-02-{i + 1:02d}" for i in range(10)],
            "centre": ["North", "South"] * 5,
        }
    )
    result = detect_shape(frame)
    assert result.shape == "wide_sheet"
    assert result.confidence >= 0.8


def test_price_list_is_generic() -> None:
    frame = pd.DataFrame({"item": ["Pencil", "Notebook"], "price": [10, 45], "quantity": [3, 2]})
    result = detect_shape(frame)
    assert result.shape == "generic_table"
    assert result.confidence >= 0.8


def test_start_and_end_on_the_same_event_stays_a_long_log() -> None:
    rows = []
    for case in range(4):
        for step_index, step in enumerate(["apply", "review", "offer"]):
            start = pd.Timestamp("2010-03-01") + pd.Timedelta(int(case * 20 + step_index * 5), unit="D")
            rows.append(
                {
                    "event_id": f"E{case}-{step_index}",
                    "case_ref": f"C{case}",
                    "step": step,
                    "started": start.strftime("%Y-%m-%dT09:00:00"),
                    "ended": (start + pd.Timedelta(1, unit="h")).strftime("%Y-%m-%dT10:00:00"),
                }
            )
    result = detect_shape(pd.DataFrame(rows))
    assert result.shape == "event_log"
    assert result.signals["within_row_gap_days"] < 1
    assert result.signals["between_row_gap_days"] > 2


def test_two_stage_dates_with_a_status_column_stay_wide() -> None:
    rows = []
    for index in range(8):
        enquiry = pd.Timestamp("2010-04-01") + pd.Timedelta(int(index // 2), unit="D")
        rows.append(
            {
                "student_id": f"S{index}",
                "centre": "North" if index < 4 else "South",
                "status": ["new", "trial", "active", "new"][index % 4],
                "enquiry_date": enquiry.strftime("%Y-%m-%d"),
                "enrol_date": (enquiry + pd.Timedelta(10, unit="D")).strftime("%Y-%m-%d"),
            }
        )
    result = detect_shape(pd.DataFrame(rows))
    assert result.shape == "wide_sheet"
    assert result.signals["within_row_gap_days"] >= 2
