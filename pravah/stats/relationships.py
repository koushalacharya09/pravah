"""Pairwise association, with the measure chosen from the two column types."""

from __future__ import annotations

import math

import pandas as pd

from pravah.stats.inference import chi_square_independence

_MAX_COLUMNS = 8


def _is_identifier(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty or pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    return len(non_null) >= 20 and non_null.nunique() / max(len(series), 1) >= 0.9


def _as_float(value: object) -> float:
    if isinstance(value, pd.Timestamp):
        return float(value.timestamp())
    return float(value)


def _kind(series: pd.Series) -> str | None:
    non_null = series.dropna()
    if len(non_null) < 3 or non_null.nunique() <= 1:
        return None
    if pd.api.types.is_datetime64_any_dtype(series) or (
        pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    ):
        return "numeric"
    return "categorical" if non_null.nunique() <= 20 else None


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(values):
        end = cursor
        while end + 1 < len(values) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end + 1) / 2.0
        for position in range(cursor, end + 1):
            ranks[order[position]] = average
        cursor = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    n = len(left)
    if n < 3:
        return None
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return covariance / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float | None:
    """Spearman rank correlation.

    WHAT: Pearson correlation of the average ranks of the two columns.
    WHY: Fees and durations often rise together without being a straight line. Spearman still sees a monotone relationship.
    WHY NOT: Pearson on the raw numbers is the default and misses a curved but still ordered relationship, and one huge fee dominates it.
    ASSUMPTIONS: At least 3 paired finite values. Ties receive the average rank. Below 8 pairs the number is reported with a low-sample warning by the caller.
    COMPLEXITY: O(n log n) time, O(n) space.
    FAILS WHEN: Either column is constant. The function returns ``None``.
    """
    return _pearson(_ranks(left), _ranks(right))


def cramers_v(left: list[object], right: list[object]) -> float | None:
    """Cramér's V for two categories.

    WHAT: ``sqrt(chi^2 / (n * min(rows-1, cols-1)))``.
    WHY: Centre against stage is two labels. V is a correlation-sized number between 0 and 1.
    WHY NOT: The raw chi-square grows with the number of rows, so two sheets of different sizes cannot be compared. A plain match rate ignores associations that are not equality.
    ASSUMPTIONS: The chi-square expected counts are at least 5. Levels are the observed labels.
    COMPLEXITY: O(n + rows * cols) time.
    FAILS WHEN: A rare level makes an expected count drop below 5. The function returns ``None`` instead of a misleading V.
    """
    levels_left = list(dict.fromkeys(left))
    levels_right = list(dict.fromkeys(right))
    index_left = {level: i for i, level in enumerate(levels_left)}
    index_right = {level: i for i, level in enumerate(levels_right)}
    table = [[0 for _ in levels_right] for _ in levels_left]
    for a, b in zip(left, right):
        table[index_left[a]][index_right[b]] += 1
    tested = chi_square_independence(table)
    if tested is None:
        return None
    statistic, _p, _degrees = tested
    n = len(left)
    scale = min(len(levels_left) - 1, len(levels_right) - 1)
    if n == 0 or scale <= 0:
        return None
    return math.sqrt(statistic / (n * scale))


def correlation_ratio(categories: list[object], values: list[float]) -> float | None:
    """Correlation ratio eta of a number broken out by a category.

    WHAT: ``sqrt(SS_between / SS_total)``, the share of variance explained by group means.
    WHY: A fee can differ by centre without the centre labels having an order. Eta does not invent an order.
    WHY NOT: Spearman would require ranking the category, which is arbitrary for labels such as North and South.
    ASSUMPTIONS: At least two groups and a non-zero total variance. At least 3 pairs.
    COMPLEXITY: O(n) time, O(groups) space.
    FAILS WHEN: Every value is the same, or every row is its own category. The function returns ``None``.
    """
    if len(values) < 3:
        return None
    overall = sum(values) / len(values)
    total = sum((value - overall) ** 2 for value in values)
    if total == 0:
        return None
    groups: dict[object, list[float]] = {}
    for category, value in zip(categories, values):
        groups.setdefault(category, []).append(value)
    if len(groups) < 2 or len(groups) > 20:
        return None
    between = 0.0
    for group in groups.values():
        mean = sum(group) / len(group)
        between += len(group) * (mean - overall) ** 2
    return math.sqrt(between / total)


def mutual_information(left: list[object], right: list[object]) -> float | None:
    """Discrete mutual information in bits.

    WHAT: ``sum p(x,y) log2( p(x,y) / (p(x)p(y)) )``.
    WHY: It detects a relationship that is not monotone and not a difference of means. It is the common scale when the type-specific measure cannot be computed.
    WHY NOT: Using it as the only number would hide whether the relationship is an order, a group shift, or a pure dependence. Those are easier to explain one at a time.
    ASSUMPTIONS: The labels are already discrete. Continuous values must be binned by the caller. At least 3 pairs.
    COMPLEXITY: O(n) time, O(levels^2) space.
    FAILS WHEN: One side is constant. Information is then zero and the function returns 0, which is a real answer, not a failure. It fails as a description when the bins were chosen so coarsely that the dependence sits inside one bin.
    """
    if len(left) < 3:
        return None
    n = len(left)
    joint: dict[tuple[object, object], int] = {}
    left_counts: dict[object, int] = {}
    right_counts: dict[object, int] = {}
    for a, b in zip(left, right):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        left_counts[a] = left_counts.get(a, 0) + 1
        right_counts[b] = right_counts.get(b, 0) + 1
    info = 0.0
    for (a, b), count in joint.items():
        p_xy = count / n
        p_x = left_counts[a] / n
        p_y = right_counts[b] / n
        info += p_xy * math.log(p_xy / (p_x * p_y), 2)
    return info


def _bins(values: list[float], groups: int = 4) -> list[int]:
    ordered = sorted(values)
    if len(set(ordered)) <= groups:
        index = {value: i for i, value in enumerate(sorted(set(ordered)))}
        return [index[value] for value in values]
    edges = []
    for step in range(1, groups):
        position = int(round((len(ordered) - 1) * step / groups))
        edges.append(ordered[position])
    binned = []
    for value in values:
        bucket = 0
        while bucket < len(edges) and value > edges[bucket]:
            bucket += 1
        binned.append(bucket)
    return binned


def _pair_points(left: list[object], right: list[object], left_kind: str, right_kind: str) -> list[dict[str, float]]:
    if left_kind != "numeric" or right_kind != "numeric":
        return []
    points = []
    for left_value, right_value in list(zip(left, right))[:40]:
        points.append({"x": round(_as_float(left_value), 4), "y": round(_as_float(right_value), 4)})
    return points


def relationship_matrix(frame: pd.DataFrame) -> dict[str, object]:
    """Choose Spearman, Cramér's V, or eta for each eligible pair, and always record mutual information.

    Identifiers and free-text columns are left out. At most 8 columns are kept, lowest cardinality first, so the page stays readable.
    """
    kinds: list[tuple[str, str]] = []
    for name in frame.columns:
        series = frame[name]
        if _is_identifier(series):
            continue
        sample = series.dropna().astype(str).head(12)
        if len(sample) and sample.str.startswith("anon:").mean() >= 0.8:
            continue
        kind = _kind(series)
        if kind is not None:
            kinds.append((str(name), kind))
    kinds = sorted(kinds, key=lambda item: (frame[item[0]].nunique(dropna=True), item[0]))[:_MAX_COLUMNS]
    cells: list[dict[str, object]] = []
    for i, (left_name, left_kind) in enumerate(kinds):
        for right_name, right_kind in kinds[i + 1 :]:
            paired = frame[[left_name, right_name]].dropna()
            if len(paired) < 3 or paired[left_name].nunique() < 2 or paired[right_name].nunique() < 2:
                continue
            left_raw = paired[left_name].tolist()
            right_raw = paired[right_name].tolist()
            measure = "mutual_information"
            value: float | None
            if left_kind == "numeric" and right_kind == "numeric":
                measure = "spearman"
                left_numbers = [_as_float(v) for v in left_raw]
                right_numbers = [_as_float(v) for v in right_raw]
                value = spearman(left_numbers, right_numbers)
                info = mutual_information(_bins(left_numbers), _bins(right_numbers))
            elif left_kind == "categorical" and right_kind == "categorical":
                measure = "cramers_v"
                value = cramers_v(left_raw, right_raw)
                info = mutual_information(left_raw, right_raw)
            else:
                measure = "correlation_ratio"
                if left_kind == "numeric":
                    numbers = [_as_float(v) for v in left_raw]
                    value = correlation_ratio(right_raw, numbers)
                    info = mutual_information(_bins(numbers), right_raw)
                else:
                    numbers = [_as_float(v) for v in right_raw]
                    value = correlation_ratio(left_raw, numbers)
                    info = mutual_information(left_raw, _bins(numbers))
            if value is None and info is None:
                continue
            shown = value if value is not None else info
            cells.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "measure": measure if value is not None else "mutual_information",
                    "value": None if shown is None else round(float(shown), 3),
                    "mutual_information": None if info is None else round(float(info), 3),
                    "n": int(len(paired)),
                    "warning": None if len(paired) >= 8 else f"n={len(paired)}, very low confidence",
                    "points": _pair_points(left_raw, right_raw, left_kind, right_kind),
                }
            )
    return {
        "columns": [name for name, _kind in kinds],
        "cells": cells,
        "sentence": (
            "Each pair uses Spearman for two numbers, Cramér's V for two categories, "
            "and the correlation ratio when a number is split by a category. "
            "Mutual information is the backup when that measure cannot be computed."
            if cells
            else "No pair of columns varies, so there is no relationship to measure."
        ),
    }
