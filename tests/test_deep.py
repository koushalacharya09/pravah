"""Phase 1.5 checks: dates, duplicates, missingness, outliers, association, Benford."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.date_format import resolve_date_formats
from pravah.ingest.duplicates import find_fuzzy_duplicates, levenshtein_distance
from pravah.ingest.pipeline import ingest
from pravah.stats.inference import benjamini_hochberg, chi2_sf, mann_whitney_p
from pravah.stats.missingness import missingness_report
from pravah.stats.outliers import modified_z_scores
from pravah.stats.relationships import cramers_v, spearman
from tests.test_pipeline import DATA


def test_kitten_to_sitting_takes_three_edits() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3


def test_ambiguous_slash_dates_follow_stage_order() -> None:
    frame = pd.DataFrame(
        {
            "enquiry_date": ["05/03/2024"] * 8,
            "demo_date": ["03/04/2024"] * 8,
        }
    )
    resolved, decisions = resolve_date_formats(frame)
    assert {item.choice for item in decisions} == {"dmy"}
    assert decisions[0].method == "stage_order"
    assert resolved["enquiry_date"].iloc[0] == pd.Timestamp("2024-03-05")
    assert resolved["demo_date"].iloc[0] == pd.Timestamp("2024-04-03")


def test_impossible_day_beats_the_stage_order_guess() -> None:
    frame = pd.DataFrame({"when": ["20/01/2024", "05/06/2024", "21/02/2024"]})
    _resolved, decisions = resolve_date_formats(frame)
    assert decisions[0].choice == "dmy"
    assert decisions[0].method == "impossible_day"


def test_fuzzy_pair_is_reported_without_the_name() -> None:
    frame = pd.DataFrame(
        {
            "student_name": ["Rahul Sharma", "Rahul Sharmaa", "Meera Lal"],
            "phone": ["9198000001", "9198000001", "9198000003"],
        }
    )
    pairs = find_fuzzy_duplicates(frame)
    assert pairs[0]["similarity"] >= 0.9
    assert "Rahul" not in pairs[0]["sentence"]
    assert "same phone" in pairs[0]["sentence"]


def test_modified_z_ignores_the_mean() -> None:
    values = [float(item) for item in range(1, 21)] + [10_000.0]
    scores = modified_z_scores(values)
    assert abs(scores[-1]) > 3.5
    assert all(abs(score) < 3.5 for score in scores[:-1])


def test_spearman_sees_a_curve_and_cramers_v_sees_a_split() -> None:
    rank = spearman([1, 2, 3, 4, 5, 6, 7, 8], [1, 4, 9, 16, 25, 36, 49, 64])
    assert rank is not None and rank > 0.99
    association = cramers_v(
        ["North"] * 10 + ["South"] * 10,
        ["stay"] * 10 + ["leave"] * 10,
    )
    assert association is not None and association > 0.8


def test_missingness_flags_a_centre_pattern() -> None:
    frame = pd.DataFrame(
        {
            "demo_date": [pd.Timestamp("2024-04-02")] * 12 + [pd.NaT] * 12,
            "centre": ["Mansarovar"] * 12 + ["Sanganer"] * 12,
        }
    )
    report = missingness_report(frame)
    assert report[0]["label"] == "not MCAR"
    assert "Sanganer" in report[0]["sentence"]


def test_benjamini_hochberg_rejects_only_the_small_p_values() -> None:
    flags = benjamini_hochberg([0.001, 0.04, 0.2, 0.8])
    assert flags[0] is True
    assert flags[-1] is False


def test_rank_test_separates_two_groups() -> None:
    p_value = mann_whitney_p([1, 2, 3, 4, 5, 6, 7, 8], [20, 21, 22, 23, 24, 25, 26, 27])
    assert p_value is not None and p_value < 0.01


def test_chi_square_tail_is_small_for_a_large_statistic() -> None:
    assert chi2_sf(30, 8) < 0.01
    assert chi2_sf(0, 8) == 1.0


def test_wide_sample_recovers_enquiry_demo_joining_fee() -> None:
    result = ingest(DATA / "sample_wide_sheet.csv")
    assert result.reconstruction is not None
    assert result.reconstruction.order == [
        "enquiry_date",
        "demo_date",
        "joining_date",
        "fee_date",
    ]
    assert result.validation.components["out_of_order"].bad_count == 1
    profile = result.profile or {}
    dates = profile["dates"]
    assert dates[0]["method"] == "stage_order"
    assert dates[0]["choice"] == "dmy"
    missing = profile["missingness"]
    assert any(item["label"] == "not MCAR" for item in missing)
    assert any("not shown" in item["sentence"] for item in profile["duplicates"])
    assert "Rahul" not in " ".join(item["sentence"] for item in profile["duplicates"])
    assert any(item["tested"] is False for item in profile["benford"])
    assert any(item["count"] > 0 for item in profile["outliers"])
    assert result.profile["verdict"]["label"] in {"Yes", "Caution", "No"}
