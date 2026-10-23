"""Dump facts for the project report. Not a product module."""

from __future__ import annotations

import json
import sys

import flask
import lifelines
import numpy as np
import pandas as pd
import sklearn

from pravah.ingest.constants import MIN_CASES_FOR_RATE, EVENTS_PER_FEATURE, RANDOM_SEED, Z_95
from pravah.web.service import analyze_sample

print("python", sys.version.split()[0])
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("flask", flask.__version__)
print("sklearn", sklearn.__version__)
print("lifelines", lifelines.__version__)
print("MIN_CASES_FOR_RATE", MIN_CASES_FOR_RATE)
print("EVENTS_PER_FEATURE", EVENTS_PER_FEATURE)
print("RANDOM_SEED", RANDOM_SEED)
print("Z_95", Z_95)

for name in ("event_log", "wide_sheet", "generic"):
    data = analyze_sample(name)
    keys = [
        "shape_label",
        "data_quality_score",
        "process_readiness_score",
        "n_cases",
        "n_events",
        "sample_observed",
        "sample_required",
        "sample_unit",
        "stage_order",
        "anonymised_columns",
    ]
    snippet = {k: data.get(k) for k in keys}
    snippet["bottleneck_verdict"] = (data.get("bottlenecks") or {}).get("verdict")
    snippet["survival_verdict"] = (data.get("survival") or {}).get("verdict")
    snippet["survival_applicable"] = (data.get("survival") or {}).get("applicable")
    snippet["risk_trained"] = (data.get("risk") or {}).get("trained")
    snippet["risk_verdict"] = (data.get("risk") or {}).get("verdict")
    snippet["causes_verdict"] = (data.get("causes") or {}).get("verdict")
    snippet["sim_validated"] = (data.get("simulation") or {}).get("validated")
    snippet["sim_verdict"] = (data.get("simulation") or {}).get("verdict")
    snippet["plan_line"] = ((data.get("plan") or {}).get("lines") or [None])[0]
    if data.get("bottlenecks") and data["bottlenecks"].get("activities"):
        snippet["top_share"] = data["bottlenecks"]["activities"][0]
    print("---", name)
    print(json.dumps(snippet, default=str, indent=2))
