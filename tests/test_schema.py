"""Schema scoring must not depend on English column names."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.schema import detect_schema


def _log() -> pd.DataFrame:
    rows = []
    for case in range(8):
        for step_index, step in enumerate(["enquiry", "trial", "enrol", "attend"]):
            rows.append(
                {
                    "learner": f"L{case}",
                    "milestone": step,
                    "happened": f"2024-05-{case + 1:02d}T0{step_index}:00:00",
                    "guide": "Asha" if step_index < 2 else "Rohit",
                    "campus": "North" if case % 2 == 0 else "South",
                }
            )
    return pd.DataFrame(rows)


def test_roles_follow_structure_not_names() -> None:
    result = detect_schema(_log(), shape="event_log")
    assert result.case_id == "learner"
    assert result.activity == "milestone"
    assert result.timestamp == "happened"
    assert result.resource == "guide"
    scores = result.column_scores
    assert scores["learner"]["case_id"] > scores["milestone"]["case_id"]
    assert scores["milestone"]["activity"] > scores["campus"]["activity"]
    assert scores["guide"]["resource"] > scores["campus"]["resource"]


def test_wide_sheet_picks_the_code_not_the_person_name() -> None:
    frame = pd.DataFrame(
        {
            "full_name": ["Asha Rao", "Dev Shah", "Meera Iyer", "Kabir Khan", "Nisha Paul", "Omar Seth"],
            "code": [f"S{i}" for i in range(6)],
            "enquiry_date": [f"2024-01-0{i + 1}" for i in range(6)],
            "enrol_date": [f"2024-02-0{i + 1}" for i in range(6)],
        }
    )
    result = detect_schema(frame, shape="wide_sheet")
    assert result.case_id == "code"
    assert result.activity is None
    assert result.timestamp is None


def test_anonymised_columns_cannot_become_the_case_or_the_activity() -> None:
    frame = _log()
    frame["roll_no"] = frame["learner"]
    frame["learner"] = "anon:" + frame["learner"].astype(str)
    result = detect_schema(frame, shape="event_log", anonymised_columns=["learner"])
    assert result.case_id == "roll_no"
    assert result.activity == "milestone"
