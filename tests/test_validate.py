"""Health score against a hand-counted seven-row log."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.models import SchemaResult
from pravah.ingest.validate import validate_event_log

_SCHEMA = SchemaResult(
    case_id="case_id",
    activity="activity",
    timestamp="timestamp",
    resource=None,
    column_scores={},
    confidence=1.0,
    notes=[],
)


def _messy() -> pd.DataFrame:
    # Fractions, counted by hand:
    # missing required: row 6 only -> 1/7
    # unparseable timestamps: every non-null time parses -> 0/6
    # consecutive pairs: case A has (1 Jan, 3 Jan), (3 Jan, 3 Jan), (3 Jan, 2 Jan)
    #   so 1 inversion out of 3 pairs. Cases B and C add no pairs.
    # duplicate keys: the second (A, review, 3 Jan) is the only extra -> 1/7
    # single-event cases: B only -> 1/3
    return pd.DataFrame(
        [
            {"case_id": "A", "activity": "apply", "timestamp": "2024-01-01"},
            {"case_id": "A", "activity": "review", "timestamp": "2024-01-03"},
            {"case_id": "A", "activity": "review", "timestamp": "2024-01-03"},
            {"case_id": "A", "activity": "offer", "timestamp": "2024-01-02"},
            {"case_id": "B", "activity": "apply", "timestamp": "2024-01-01"},
            {"case_id": "C", "activity": "apply", "timestamp": None},
            {"case_id": "C", "activity": "review", "timestamp": "2024-01-05"},
        ]
    )


def test_quality_score_matches_the_weighted_hand_calculation() -> None:
    result = validate_event_log(_messy(), _SCHEMA)
    missing = 100.0 * (1.0 - 1.0 / 7.0)
    unparseable = 100.0
    backwards = 100.0 * (1.0 - 1.0 / 3.0)
    duplicate = 100.0 * (1.0 - 1.0 / 7.0)
    single = 100.0 * (1.0 - 1.0 / 3.0)
    expected = (
        25.0 * missing + 25.0 * unparseable + 20.0 * backwards + 15.0 * duplicate + 15.0 * single
    ) / 100.0
    assert result.components["missing_required"].bad_count == 1
    assert result.components["unparseable_timestamps"].bad_count == 0
    assert result.components["out_of_order"].bad_count == 1
    assert result.components["out_of_order"].total == 3
    assert result.components["duplicate_events"].bad_count == 1
    assert result.components["single_event_cases"].bad_count == 1
    assert result.process_readiness_score == expected
    assert result.sufficient_for_inference is False
    assert result.minimum_rows_recommended == 97
    assert "97" in result.sample_sentence
    assert "0.10²" in result.sample_sentence or "0.10^2" in result.sample_sentence or "0.10²" in result.sample_sentence


def test_clean_log_scores_one_hundred_and_still_warns_when_small() -> None:
    frame = pd.DataFrame(
        [
            {"case_id": "A", "activity": "apply", "timestamp": "2024-01-01"},
            {"case_id": "A", "activity": "done", "timestamp": "2024-01-02"},
        ]
    )
    result = validate_event_log(frame, _SCHEMA)
    assert result.data_quality_score == 100.0
    assert result.process_readiness_score == 100.0
    assert "Data quality is 100.0 out of 100" in result.summary
    assert "Process readiness is 100.0 out of 100" in result.summary
    assert result.sufficient_for_inference is False
