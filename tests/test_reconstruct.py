"""Pairwise precedence, including the small-sample refusal, and a hand cycle."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.reconstruct import pairwise_precedence, reconstruct_wide


def test_hand_counted_precedence_and_ties() -> None:
    frame = pd.DataFrame(
        {
            "enquiry": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-09", None],
            "trial": ["2024-01-02", "2024-01-03", "2024-01-01", "2024-01-04", "2024-01-10", "2024-01-01"],
        }
    )
    # Five rows have both dates: enquiry is earlier on rows 0, 1, and 4 (3),
    # later on row 2 (1), tied on row 3 (1). The blank enquiry row is ignored.
    pairs, edges = pairwise_precedence(frame, ["enquiry", "trial"])
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.support, pair.left_before, pair.right_before, pair.ties) == (5, 3, 1, 1)
    assert pair.p_left_before == 3 / 5
    assert edges == []
    assert pair.kept_direction is None


def test_four_of_five_does_not_become_an_edge_and_five_of_five_does() -> None:
    shaky = pd.DataFrame(
        {
            "a": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06"],
            "b": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-05"],
        }
    )
    # 4 of 5 have a < b. Point estimate 0.80, Wilson lower bound under 0.50.
    _pairs, edges = pairwise_precedence(shaky, ["a", "b"])
    assert edges == []

    solid = pd.DataFrame(
        {
            "a": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "b": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        }
    )
    pairs, edges = pairwise_precedence(solid, ["a", "b"])
    assert len(edges) == 1
    assert (edges[0].source, edges[0].target) == ("a", "b")
    assert pairs[0].provisional is True


def test_support_below_five_never_votes() -> None:
    frame = pd.DataFrame({"a": ["2024-01-01"] * 4, "b": ["2024-01-02"] * 4})
    pairs, edges = pairwise_precedence(frame, ["a", "b"])
    assert edges == []
    assert "4" in pairs[0].reject_reason


def test_reconstruction_emits_one_event_per_filled_stage_in_topo_order() -> None:
    frame = pd.DataFrame(
        {
            "student_id": [f"S{i}" for i in range(8)],
            "region": ["North", "South"] * 4,
            "enquiry_date": [f"2024-01-{day:02d}" for day in range(1, 9)],
            "trial_date": [f"2024-01-{day:02d}" for day in range(5, 13)],
            "enrol_date": [f"2024-02-{day:02d}" for day in range(1, 9)],
        }
    )
    # One swapped trial, still a clear majority: 7/8 enquiry before trial.
    frame.loc[0, "trial_date"] = "2023-12-01"
    result = reconstruct_wide(frame, case_id="student_id")
    assert result.order == ["enquiry_date", "trial_date", "enrol_date"]
    assert result.dropped_edges == []
    assert len(result.frame) == 24
    first_case = result.frame[result.frame["student_id"] == "S0"][result.activity_column].tolist()
    assert first_case == ["enquiry_date", "trial_date", "enrol_date"]
    assert set(result.frame["region"]) == {"North", "South"}
