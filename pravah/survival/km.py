"""Product-limit survival curve and Greenwood bands."""

from __future__ import annotations

import math

import numpy as np

from pravah.ingest.constants import Z_95


def _day(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def kaplan_meier(durations: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    """Right-censored Kaplan–Meier curve with a Greenwood log-log band.

    WHAT: At each time with at least one event, survival is multiplied by ``1 - deaths / at risk``. Cases with ``observed`` 0 stay in the risk set until their time and are not counted as events. The Greenwood sum is ``deaths / (at_risk * (at_risk - deaths))``. The 95% band is the log-log transform of that variance, using ``z = 1.96``.
    WHY: An open case has only a lower bound on its time. Dropping it makes the process look faster. Treating its last day as a failure makes the process look slower. The product-limit keeps it in the risk set and then lets it leave.
    WHY NOT: A plain ``S ± 1.96 se`` band. Greenwood's standard error is on the probability scale and that band falls outside 0 and 1. The log-log band is the one ``lifelines`` prints, so the two implementations can be compared cell by cell. A life table with fixed bins was rejected because the event times are already the bins.
    ASSUMPTIONS: Times are days from each case's own first event. Ties are one jump. A case censored at the same time as an event is still at risk for that event. ``Z_95`` is 1.96.
    COMPLEXITY: O(n log n) time to sort, O(n) space.
    FAILS WHEN: Every case is censored. The curve stays at 1 and no hazard window is named. When the last event empties the risk set, survival hits 0 and the Greenwood denominator is zero, so the band at that point is closed at 0.
    """
    times = np.asarray(durations, dtype=float)
    flags = np.asarray(observed, dtype=int)
    if len(times) == 0:
        return {"curve": [], "window": None, "events": 0, "censored": 0}
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    flags = flags[order]
    total = len(times)
    curve: list[dict[str, float]] = [{"time": 0.0, "survival": 1.0, "low": 1.0, "high": 1.0, "greenwood": 0.0, "at_risk": total, "deaths": 0}]
    survival = 1.0
    greenwood = 0.0
    cursor = 0
    peak: tuple[float, float, float] | None = None
    previous_time = 0.0
    while cursor < total:
        moment = times[cursor]
        at_risk = total - cursor
        deaths = 0
        while cursor < total and times[cursor] == moment:
            deaths += int(flags[cursor] == 1)
            cursor += 1
        if deaths and at_risk:
            hazard = deaths / at_risk
            if at_risk > deaths:
                greenwood += deaths / (at_risk * (at_risk - deaths))
            survival *= 1.0 - hazard
            low, high = _band(survival, greenwood)
            if peak is None or hazard > peak[0]:
                peak = (hazard, previous_time, float(moment))
            previous_time = float(moment)
            curve.append(
                {
                    "time": round(float(moment), 4),
                    "survival": round(survival, 6),
                    "low": round(low, 6),
                    "high": round(high, 6),
                    "greenwood": round(greenwood, 8),
                    "at_risk": int(at_risk),
                    "deaths": int(deaths),
                }
            )
    window = None
    if peak is not None:
        window = {"start": round(peak[1], 4), "end": round(peak[2], 4), "hazard": round(peak[0], 4)}
    return {
        "curve": curve,
        "window": window,
        "events": int(flags.sum()),
        "censored": int(total - flags.sum()),
    }


def _band(survival: float, greenwood: float) -> tuple[float, float]:
    if survival <= 0.0 or survival >= 1.0 or greenwood <= 0.0:
        return survival, survival
    log_s = math.log(survival)
    if log_s == 0.0:
        return survival, survival
    centre = math.log(-log_s)
    step = Z_95 * math.sqrt(greenwood) / abs(log_s)
    low = math.exp(-math.exp(centre + step))
    high = math.exp(-math.exp(centre - step))
    return max(0.0, min(low, high)), min(1.0, max(low, high))


def window_sentence(window: dict[str, float] | None, *, dropout: bool, events: int) -> str:
    """Name the day range that holds the highest hazard."""
    if events <= 0 or window is None:
        return "No drop-off was observed." if dropout else "No completion was observed."
    start, end = _day(float(window["start"])), _day(float(window["end"]))
    if start == end:
        if dropout:
            return f"Most drop-offs happen on day {end}."
        return f"Most completions happen on day {end}."
    if dropout:
        return f"Most drop-offs happen between day {start} and day {end}."
    return f"Most completions happen between day {start} and day {end}."
