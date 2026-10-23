"""Actions taken only from numbers earlier modules already computed."""

from __future__ import annotations

from pravah.plan.knapsack import knapsack


def build_actions(bottlenecks: dict | None, causes: dict | None, risk: dict | None, simulation: dict | None) -> list[dict[str, object]]:
    """Candidate actions whose gain, if any, is a simulated interval.

    WHAT: A bottleneck action exists when a stage holds the largest share. A causal action exists only when the verdict is likely causal. A risk action exists when an open case is listed. A rework action exists when a loop was found. The gain interval is copied from the validated scenario. If the simulator did not validate, the gain is null and the action cannot be selected.
    WHY: An action plan that invents a saving has nothing to show when someone asks where the number came from. Copying the interval keeps every sentence tied to a module that already ran.
    WHY NOT: A default 10% improvement. That number is not in the file.
    ASSUMPTIONS: The scenario interval is a throughput change in days, baseline minus scenario, so a positive gain is a shorter case. Costs are filled later by the user.
    COMPLEXITY: O(actions) time.
    FAILS WHEN: No earlier module produced a finding. The list is empty and the plan says there is nothing to choose.
    """
    actions: list[dict[str, object]] = []
    gain = None
    low = None
    high = None
    if simulation and simulation.get("validated") and simulation.get("scenario"):
        scenario = simulation["scenario"]
        base = float(scenario["baseline"] or 0)
        mid = float(scenario["median"] or 0)
        gain = round(base - mid, 4)
        low = round(base - float(scenario["high"] or mid), 4)
        high = round(base - float(scenario["low"] or mid), 4)
    ranked = (bottlenecks or {}).get("activities") or []
    if bottlenecks and bottlenecks.get("applicable") and ranked:
        actions.append(
            {
                "id": "bottleneck",
                "text": f"Add capacity before {ranked[0]['activity']}.",
                "evidence": bottlenecks.get("verdict"),
                "gain": gain,
                "low": low,
                "high": high,
                "confidence": "from the bottleneck share",
                "source": "bottlenecks",
            }
        )
    if causes and causes.get("refutation") and causes["refutation"].get("verdict") == "likely causal":
        actions.append(
            {
                "id": "cause",
                "text": "Keep the treatment that passed every refutation test.",
                "evidence": causes.get("verdict"),
                "gain": gain,
                "low": low,
                "high": high,
                "confidence": "likely causal",
                "source": "causes",
            }
        )
    if risk and risk.get("cases"):
        top = risk["cases"][0]
        actions.append(
            {
                "id": "risk",
                "text": f"Review {top['case']} first. Its drop or breach probability is {top['probability']}.",
                "evidence": risk.get("verdict"),
                "gain": gain,
                "low": low,
                "high": high,
                "confidence": "from the risk model",
                "source": "risk",
            }
        )
    loops = (bottlenecks or {}).get("rework") or []
    if loops:
        actions.append(
            {
                "id": "rework",
                "text": f"Cut the rework loop at {loops[0]['activity']}.",
                "evidence": f"Extra span {loops[0].get('extra_days')} days.",
                "gain": gain,
                "low": low,
                "high": high,
                "confidence": "from the rework span",
                "source": "rework",
            }
        )
    return actions


def choose_plan(actions: list[dict[str, object]], costs: dict[str, int], budget: int, *, risk_adjusted: bool = False) -> dict[str, object]:
    """Pick actions with the knapsack, using the lower bound when asked.

    WHAT: Each action needs a gain interval and a positive cost. The objective is the gain, or its lower bound when ``risk_adjusted`` is true. The dynamic program returns the subset. The report lines quote the evidence, the interval, the cost, and the confidence already stored on the action.
    WHY: A lower bound is the pessimistic reading of the same interval. It is still a computed number. A sentence is not written unless that number exists.
    WHY NOT: Ranking by confidence words. Words do not add up to a budget.
    ASSUMPTIONS: Costs are positive integers supplied by the user. Missing costs default to 1 so the program can still show a subset. An action with a null gain is listed and not eligible.
    COMPLEXITY: O(actions * budget) inside the knapsack.
    FAILS WHEN: No action has a gain. The plan says the simulator did not validate, so no impact is claimed.
    """
    eligible = []
    lines = []
    for action in actions:
        if action.get("gain") is None:
            lines.append(f"{action['text']} No simulated gain is available, so this action is not costed.")
            continue
        value = float(action["low"] if risk_adjusted else action["gain"])
        cost = int(costs.get(str(action["id"]), 1))
        if value <= 0:
            lines.append(f"{action['text']} The simulated gain is {value} days, so it is not selected.")
            continue
        eligible.append({**action, "value": value, "cost": cost})
    if not eligible:
        return {"chosen": [], "value": None, "low": None, "high": None, "lines": lines or ["No action has a computed gain."], "differs": False}
    packed = knapsack([float(item["value"]) for item in eligible], [int(item["cost"]) for item in eligible], budget)
    chosen = [eligible[index] for index in packed["chosen"]]
    for item in chosen:
        lines.append(
            f"{item['text']} Evidence: {item['evidence']} Gain {item['gain']} days "
            f"(interval {item['low']} to {item['high']}). Cost {item['cost']}. Confidence: {item['confidence']}."
        )
    if packed["differs"]:
        lines.append(
            f"Greedy would keep value {packed['greedy_value']}. The knapsack keeps {packed['value']}."
        )
    low = sum(float(item["low"] or 0) for item in chosen)
    high = sum(float(item["high"] or 0) for item in chosen)
    return {
        "chosen": [{"id": item["id"], "text": item["text"], "cost": item["cost"], "gain": item["gain"], "low": item["low"], "high": item["high"], "evidence": item["evidence"]} for item in chosen],
        "value": packed["value"],
        "low": round(low, 4),
        "high": round(high, 4),
        "lines": lines,
        "differs": packed["differs"],
        "greedy_value": packed["greedy_value"],
    }
