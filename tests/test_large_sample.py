"""The 400-case coaching log must unlock later phases with a real result."""

from __future__ import annotations

from pathlib import Path

from pravah.web.service import analyze_sample

ROOT = Path(__file__).resolve().parents[1]


def test_causes_readiness_asks_for_a_treatment_when_roles_exist() -> None:
    event = analyze_sample("event_log")
    row = next(item for item in event["profile"]["readiness"] if item["analysis"] == "Causes")
    assert row["status"] != "no"
    assert "Choose a treatment to unlock" in row["reason"]
    plain = analyze_sample("generic")
    refused = next(item for item in plain["profile"]["readiness"] if item["analysis"] == "Causes")
    assert refused["status"] == "no"


def test_realistic_sample_gives_nontrivial_later_phases() -> None:
    source = ROOT / "data" / "sample_large.csv"
    assert source.is_file()
    text = source.read_text(encoding="utf-8")
    assert "2010-" in text
    assert "2024-" not in text
    assert "2025-" not in text
    assert "2026-" not in text
    body = analyze_sample("large")
    assert body["n_cases"] == 400
    survival = body["survival"]
    assert survival["applicable"] is True
    assert int(survival["events"]) >= 40
    assert len(survival["curve"]) >= 2
    risk = body["risk"]
    assert risk["trained"] is True
    assert risk["cases"]
    causes = body["causes"]
    estimate = causes.get("estimate") or {}
    assert estimate.get("estimated") is True
    assert causes.get("refutation")
    assert estimate.get("ipw") not in (None, 0, 0.0)
    simulation = body["simulation"]
    assert simulation["validated"] is True
    scenario = simulation["scenario"]
    assert scenario["low"] <= scenario["median"] <= scenario["high"]
    assert scenario["median"] != scenario["baseline"]
    plan = body["plan"]
    gain = plan.get("value")
    assert gain not in (None, 0, 0.0)
    assert plan.get("chosen")


def test_requirements_pins_every_runtime_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = []
    capturing = False
    for line in pyproject.splitlines():
        if line.strip() == "dependencies = [":
            capturing = True
            continue
        if capturing:
            if line.strip() == "]":
                break
            package = line.strip().strip(",").strip('"').split(">")[0].split("=")[0].split("<")[0].strip()
            if package:
                names.append(package)
    assert "lifelines" in names
    assert "scikit-learn" in names
    for name in names:
        assert f"{name}==" in requirements, name
