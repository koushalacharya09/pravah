"""Hand-checked bottleneck shares, Little's Law, shrinkage, and the slowdown date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.models import SchemaResult
from pravah.perf.analyze import analyze_bottlenecks
from pravah.perf.littles import littles_check
from pravah.perf.score import bootstrap_shares
from pravah.perf.times import Visit, build_visits
from pravah.perf.trend import slowdown_index
from pravah.web.service import analyze_sample


def _schema(**roles: str | None) -> SchemaResult:
    return SchemaResult(
        case_id=roles.get("case_id"),
        activity=roles.get("activity"),
        timestamp=roles.get("timestamp"),
        resource=roles.get("resource"),
        column_scores={},
        confidence=1.0,
        notes=[],
    )


def test_waiting_share_matches_the_hand_total() -> None:
    frame = pd.DataFrame(
        {
            "case": ["A", "A", "A", "B", "B", "B"],
            "step": ["enquiry", "trial", "enrol", "enquiry", "trial", "enrol"],
            "when": [
                "2024-01-01",
                "2024-01-04",
                "2024-01-05",
                "2024-01-01",
                "2024-01-02",
                "2024-01-06",
            ],
        }
    )
    report = analyze_bottlenecks(frame, _schema(case_id="case", activity="step", timestamp="when"), draws=200)
    # A waits 3 days before trial and 1 before enrol, over 4 days.
    # B waits 1 day before trial and 4 before enrol, over 5 days.
    # trial (3+1)/9, enrol (1+4)/9.
    by_name = {row["activity"]: row for row in report["activities"]}
    assert by_name["enrol"]["share"] == round(5 / 9, 4)
    assert by_name["trial"]["share"] == round(4 / 9, 4)
    assert by_name["enquiry"]["share"] == 0.0
    assert report["verdict"].startswith("Most time is lost before enrol.")
    assert report["mode"] == "transition"
    assert "waiting time" in report["mode_sentence"]
    assert by_name["enrol"]["share_low"] <= by_name["enrol"]["share"] <= by_name["enrol"]["share_high"]


def test_bootstrap_interval_matches_an_independent_resample() -> None:
    frame = pd.DataFrame(
        {
            "case": ["A", "A", "B", "B", "C", "C"],
            "step": ["one", "two", "one", "two", "one", "two"],
            "when": ["2024-01-01", "2024-01-03", "2024-01-01", "2024-01-06", "2024-01-01", "2024-01-02"],
        }
    )
    _mode, _note, visits = build_visits(frame, case_id="case", activity="step", timestamp="when", resource=None)
    rows = bootstrap_shares(visits, draws=50, seed=42)
    wait = np.array([[0.0, 2.0], [0.0, 5.0], [0.0, 1.0]])
    total = np.array([2.0, 5.0, 1.0])
    rng = np.random.default_rng(42)
    picked = rng.integers(0, 3, size=(50, 3))
    shares = wait[picked].sum(axis=1) / total[picked].sum(axis=1)[:, None]
    two = next(row for row in rows if row["activity"] == "two")
    assert two["share_low"] == round(float(np.quantile(shares[:, 1], 0.025)), 4)
    assert two["share_high"] == round(float(np.quantile(shares[:, 0 if False else 1], 0.975)), 4)


def test_start_and_end_split_waiting_from_processing() -> None:
    rows = []
    for case, day in (("A", 1), ("B", 2), ("C", 3)):
        rows.append((case, "cut", f"2024-01-0{day} 09:00", f"2024-01-0{day} 12:00"))
        rows.append((case, "sew", f"2024-01-0{day + 1} 09:00", f"2024-01-0{day + 1} 11:00"))
    frame = pd.DataFrame(rows, columns=["case", "step", "start", "finish"])
    report = analyze_bottlenecks(frame, _schema(case_id="case", activity="step", timestamp="start"))
    assert report["mode"] == "start_end"
    sew = next(row for row in report["activities"] if row["activity"] == "sew")
    # 21 idle hours over a 26-hour throughput, for every case.
    assert sew["share"] == round(21 / 26, 4)
    assert sew["median_process_days"] == round(2 / 24, 2)
    assert "processing time" in report["mode_sentence"]


def test_steady_queue_passes_littles_law_and_a_growing_wait_does_not() -> None:
    steady_rows = []
    for index in range(8):
        start = pd.Timestamp("2010-01-01") + pd.Timedelta(int(index), unit="D")
        steady_rows.append(("C" + str(index), "hold", start))
        steady_rows.append(("C" + str(index), "done", start + pd.Timedelta(2, unit="D")))
    steady = pd.DataFrame(steady_rows, columns=["case", "step", "when"])
    _mode, _note, visits = build_visits(steady, case_id="case", activity="step", timestamp="when", resource=None)
    hold = next(row for row in littles_check(visits) if row["activity"] == "hold")
    assert hold["steady"] is True

    growing = []
    for index in range(6):
        start = pd.Timestamp("2010-03-01") + pd.Timedelta(int(index * 2), unit="D")
        growing.append(("G" + str(index), "hold", start))
        growing.append(("G" + str(index), "done", start + pd.Timedelta(1, unit="D")))
    burst = pd.Timestamp("2010-03-20")
    for index in range(6):
        growing.append(("B" + str(index), "hold", burst))
        growing.append(("B" + str(index), "done", burst + pd.Timedelta(5, unit="D")))
    grown = pd.DataFrame(growing, columns=["case", "step", "when"])
    _mode, _note, grown_visits = build_visits(grown, case_id="case", activity="step", timestamp="when", resource=None)
    grown_hold = next(row for row in littles_check(grown_visits) if row["activity"] == "hold")
    assert grown_hold["steady"] is False
    assert "not in steady state" in grown_hold["sentence"]


def test_a_single_slow_event_cannot_outrank_a_larger_resource() -> None:
    rows = [("S" + str(index), "step", "2024-01-01", "2024-01-03", "regular") for index in range(12)]
    rows.append(("Z", "step", "2024-02-01", "2024-04-01", "newcomer"))
    frame = pd.DataFrame(rows, columns=["case", "step", "start", "end", "who"])
    # Two timestamps on one row is start and end only when the second is a column
    # on the same event. These rows are single events, so the gap is zero and the
    # end column is processing. Add a following marker so the sheet has an event time
    # and a finish. The resource duration is the processing time.
    report = analyze_bottlenecks(
        frame, _schema(case_id="case", activity="step", timestamp="start", resource="who")
    )
    assert report["mode"] == "start_end"
    ranks = {row["resource"]: row["rank"] for row in report["resources"]}
    assert ranks["newcomer"] > ranks["regular"]
    newcomer = next(row for row in report["resources"] if row["resource"] == "newcomer")
    assert newcomer["small_sample"] is True
    assert newcomer["shrunk_days"] < newcomer["median_days"]


def test_binary_segmentation_names_the_slower_week() -> None:
    assert slowdown_index([2, 2, 2, 10, 10, 10]) == 3
    assert slowdown_index([2, 2, 2, 2]) is None
    rows = []
    for week, days in enumerate([2, 2, 2, 10, 10, 10]):
        end = pd.Timestamp("2010-03-01") + pd.Timedelta(int(week), unit="W")
        start = end - pd.Timedelta(int(days), unit="D")
        rows.append(("W" + str(week), "start", start.date().isoformat()))
        rows.append(("W" + str(week), "end", end.date().isoformat()))
    frame = pd.DataFrame(rows, columns=["case", "step", "when"])
    report = analyze_bottlenecks(frame, _schema(case_id="case", activity="step", timestamp="when"))
    assert report["trend"]["change_point"] == "2010-03-22"
    assert report["trend"]["sentence"] == "Throughput time slowed down around 2010-03-22."


def test_rework_extra_days_are_the_span_from_first_to_last_repeat() -> None:
    frame = pd.DataFrame(
        {
            "case": ["A", "A", "A"],
            "step": ["demo", "join", "demo"],
            "when": ["2024-01-01", "2024-01-03", "2024-01-06"],
        }
    )
    report = analyze_bottlenecks(frame, _schema(case_id="case", activity="step", timestamp="when"))
    assert report["rework"] == [
        {"activity": "demo", "cases": 1, "extra_days": 5.0, "median_extra_days": 5.0}
    ]


def test_plain_table_refuses_and_the_event_log_names_attend() -> None:
    plain = analyze_sample("generic")
    assert plain["bottlenecks"]["applicable"] is False
    event = analyze_sample("event_log")
    assert event["bottlenecks"]["verdict"].startswith("Most time is lost before attend.")
    attend = next(row for row in event["bottlenecks"]["activities"] if row["activity"] == "attend")
    assert attend["share"] == 0.5
    resources = event["bottlenecks"]["resources"]
    assert resources[0]["resource"] != "Meera Lal"
