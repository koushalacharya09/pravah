"""Modified z-scores from the median and the median absolute deviation."""

from __future__ import annotations

import pandas as pd

_MODIFIED_Z_CUTOFF = 3.5
_CONSISTENCY = 0.6745


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def modified_z_scores(values: list[float]) -> list[float]:
    """Iglewicz–Hoaglin modified z-score for every value.

    WHAT: ``0.6745 * (x - median) / MAD``, where ``MAD`` is the median of ``|x - median|``.
    WHY: Fees and durations are skewed. The median and MAD stay put when a few values are extreme, so the score still means "far from the bulk."
    WHY NOT: The ordinary z-score ``(x - mean) / sd`` is the default and is rejected because the outliers move the mean and the standard deviation toward themselves, which hides the next outlier.
    ASSUMPTIONS: The column is numeric and the middle half of the data is a useful description of a typical value. At least 8 finite numbers.
    COMPLEXITY: O(n log n) time for the sorts, O(n) space.
    FAILS WHEN: More than half the values are identical, so MAD is 0. The function then returns an empty list and the caller falls back to the IQR fence.
    """
    clean = [float(value) for value in values if value == value]
    if len(clean) < 8:
        return []
    centre = _median(clean)
    mad = _median([abs(value - centre) for value in clean])
    if mad == 0:
        return []
    return [_CONSISTENCY * (value - centre) / mad for value in clean]


def iqr_bounds(values: list[float]) -> tuple[float, float] | None:
    """Tukey fences at 1.5 IQR. Used only when MAD is zero."""
    clean = sorted(float(value) for value in values if value == value)
    if len(clean) < 8:
        return None
    def quartile(fraction: float) -> float:
        position = (len(clean) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(clean) - 1)
        weight = position - low
        return clean[low] * (1 - weight) + clean[high] * weight
    q1 = quartile(0.25)
    q3 = quartile(0.75)
    span = q3 - q1
    if span == 0:
        return None
    return q1 - 1.5 * span, q3 + 1.5 * span


def outlier_report(frame: pd.DataFrame) -> list[dict[str, object]]:
    """One plain-language row per numeric column that has extreme values."""
    reports: list[dict[str, object]] = []
    for name in frame.columns:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = [float(value) for value in series.dropna().tolist()]
        if len(values) < 8:
            continue
        scores = modified_z_scores(values)
        if scores:
            flags = [abs(score) > _MODIFIED_Z_CUTOFF for score in scores]
            method = "modified z-score"
        else:
            bounds = iqr_bounds(values)
            if bounds is None:
                continue
            low, high = bounds
            flags = [value < low or value > high for value in values]
            method = "IQR fence"
        count = sum(flags)
        if count == 0:
            continue
        fraction = count / len(values)
        reports.append(
            {
                "column": str(name),
                "count": count,
                "total": len(values),
                "method": method,
                "sentence": (
                    f"{count} of {len(values)} values in {name} sit far from the middle "
                    f"({method}). That is {fraction:.0%} of the column, so a plain average will be pulled."
                ),
            }
        )
    if not reports:
        reports.append(
            {
                "column": None,
                "count": 0,
                "total": 0,
                "method": "modified z-score",
                "sentence": "No numeric column has values far from its middle.",
            }
        )
    return reports


def box_summaries(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Five-number summary for each numeric column, for the page's box plots."""
    boxes = []
    for name in frame.columns:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        values = sorted(float(value) for value in series.dropna().tolist())
        if len(values) < 4:
            continue
        def at(fraction: float) -> float:
            position = (len(values) - 1) * fraction
            low = int(position)
            high = min(low + 1, len(values) - 1)
            weight = position - low
            return values[low] * (1 - weight) + values[high] * weight
        scores = modified_z_scores(values)
        flags = []
        if scores:
            flags = [values[index] for index, score in enumerate(scores) if abs(score) > _MODIFIED_Z_CUTOFF]
        boxes.append(
            {
                "column": str(name),
                "min": values[0],
                "q1": round(at(0.25), 4),
                "median": round(at(0.5), 4),
                "q3": round(at(0.75), 4),
                "max": values[-1],
                "flags": flags[:8],
            }
        )
    return boxes
