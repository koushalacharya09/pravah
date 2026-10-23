"""Benford's law for amount-like columns, with an explicit skip when it does not apply."""

from __future__ import annotations

import math

import pandas as pd

from pravah.stats.inference import chi2_sf

_MIN_VALUES = 30
_DIGIT_PROBABILITY = {digit: math.log10(1 + 1 / digit) for digit in range(1, 10)}


def _first_digit(value: float) -> int | None:
    if value <= 0 or value != value or value in {float("inf"), float("-inf")}:
        return None
    text = f"{value:.12e}"
    digit = text[0]
    if digit == "-":
        digit = text[1]
    if not digit.isdigit() or digit == "0":
        return None
    return int(digit)


def benford_report(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Test first digits of amount-like columns against Benford's law.

    WHAT: Expected share of first digit ``d`` is ``log10(1 + 1/d)``. The chi-square statistic compares observed counts with ``n`` times that share, on 8 degrees of freedom.
    WHY: Naturally varying amounts, from invoices to fees that are not a menu of fixed prices, put more numbers under 1, 2, and 3 than under 9. A large departure is a reason to ask whether the column was typed by hand or copied from a rule.
    WHY NOT: A z-score on the raw amounts does not know what a first-digit law looks like. Applying Benford to case ids or to a price list of three SKUs accuses clean data.
    ASSUMPTIONS: The column is positive, has at least 30 values, at least 9 distinct values, and the largest value is at least 10 times the smallest. Rows are independent.
    COMPLEXITY: O(n) time per column, O(1) extra space.
    FAILS WHEN: The amounts are assigned, such as a flat fee. The function then skips the column and says why. It also fails when the sample is under 30, because the chi-square tail is then a poor description of the counts.
    """
    reports: list[dict[str, object]] = []
    numeric_names = [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]
    if not numeric_names:
        return [
            {
                "column": None,
                "tested": False,
                "sentence": "No numeric amount column was found, so Benford's law was not used.",
            }
        ]
    for name in numeric_names:
        series = frame[name].dropna()
        positive = [float(value) for value in series.tolist() if float(value) > 0]
        distinct = len(set(positive))
        if len(positive) < _MIN_VALUES:
            reports.append(
                {
                    "column": str(name),
                    "tested": False,
                    "sentence": (
                        f"{name} was not checked against Benford's law: only {len(positive)} positive values, "
                        f"and { _MIN_VALUES } are needed."
                    ),
                }
            )
            continue
        if distinct <= 8:
            reports.append(
                {
                    "column": str(name),
                    "tested": False,
                    "sentence": (
                        f"{name} was not checked against Benford's law: only {distinct} distinct amounts. "
                        "Fixed fees do not follow that law."
                    ),
                }
            )
            continue
        low = min(positive)
        high = max(positive)
        if low == 0 or high / low < 10:
            reports.append(
                {
                    "column": str(name),
                    "tested": False,
                    "sentence": (
                        f"{name} was not checked against Benford's law: the values do not span "
                        "a factor of ten, so the first-digit law does not apply."
                    ),
                }
            )
            continue
        counts = {digit: 0 for digit in range(1, 10)}
        for value in positive:
            digit = _first_digit(value)
            if digit is not None:
                counts[digit] += 1
        n = sum(counts.values())
        statistic = 0.0
        for digit in range(1, 10):
            expected = n * _DIGIT_PROBABILITY[digit]
            statistic += (counts[digit] - expected) ** 2 / expected
        p_value = chi2_sf(statistic, 8)
        if p_value < 0.01:
            verdict = (
                f"{name} departs from Benford's law (chi-square {statistic:.1f}, approximate p {p_value:.3f}). "
                "The first digits do not look like a naturally varying amount. This can mean a fixed menu, hand typing, or a formula."
            )
        else:
            verdict = (
                f"{name} is compatible with Benford's law (chi-square {statistic:.1f}, approximate p {p_value:.3f})."
            )
        reports.append(
            {
                "column": str(name),
                "tested": True,
                "statistic": round(statistic, 2),
                "p_value": round(p_value, 4),
                "digits": [
                    {
                        "digit": digit,
                        "actual": counts[digit],
                        "expected": round(n * _DIGIT_PROBABILITY[digit], 2),
                    }
                    for digit in range(1, 10)
                ],
                "sentence": verdict,
            }
        )
    return reports
