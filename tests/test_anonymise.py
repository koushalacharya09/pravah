"""Personal data is hashed; case ids and activity labels are not."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.anonymise import anonymise_frame, hash_value
from pravah.ingest.constants import ANON_PREFIX


def test_phone_email_and_name_are_hashed_and_stable() -> None:
    frame = pd.DataFrame(
        {
            "student_id": ["S1", "S1", "S2"],
            "student_name": ["Asha Rao", "Asha Rao", "Dev Shah"],
            "phone": ["9000000001", "9000000001", "9000000002"],
            "email": ["asha@example.com", "asha@example.com", "dev@example.com"],
            "step": ["enquiry", "trial", "enquiry"],
            "centre": ["North", "North", "South"],
        }
    )
    hashed, columns = anonymise_frame(frame, salt="test-salt")
    assert set(columns) == {"student_name", "phone", "email"}
    assert hashed["student_id"].tolist() == ["S1", "S1", "S2"]
    assert hashed["step"].tolist() == ["enquiry", "trial", "enquiry"]
    assert hashed["centre"].tolist() == ["North", "North", "South"]
    assert hashed["phone"].iloc[0] == hashed["phone"].iloc[1]
    assert hashed["phone"].iloc[0] != hashed["phone"].iloc[2]
    assert str(hashed["phone"].iloc[0]).startswith(ANON_PREFIX)
    assert "9000000001" not in set(hashed.astype(str).values.ravel())
    assert hashed["student_name"].iloc[0] == hash_value("Asha Rao", salt="test-salt", column="student_name")


def test_single_given_name_without_a_name_header_is_kept() -> None:
    frame = pd.DataFrame({"owner": ["Anita", "Rahul", "Anita"]})
    hashed, columns = anonymise_frame(frame, salt="test-salt")
    assert columns == []
    assert hashed["owner"].tolist() == ["Anita", "Rahul", "Anita"]


def test_tutor_header_hashes_a_personal_name() -> None:
    frame = pd.DataFrame({"tutor": ["Anita Rao", "Rahul Das"]})
    _hashed, columns = anonymise_frame(frame, salt="test-salt")
    assert columns == ["tutor"]
