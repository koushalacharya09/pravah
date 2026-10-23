"""Distribution tails, chi-square, Mann–Whitney, and Benjamini–Hochberg."""

from __future__ import annotations

import math
from collections.abc import Sequence


def normal_sf(z: float) -> float:
    """Upper tail of a standard normal, via the complementary error function.

    WHAT: ``P(Z > z) = 0.5 * erfc(z / sqrt(2))``.
    WHY: Later tests need a tail probability, and ``erfc`` is in the standard library.
    WHY NOT: A z table is the textbook shortcut and is too coarse around 0.05.
    ASSUMPTIONS: ``z`` is a standard normal statistic.
    COMPLEXITY: O(1) time and space.
    FAILS WHEN: The statistic is not approximately normal.
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_sf(statistic: float, degrees: int) -> float:
    """Chi-square upper tail by the Wilson–Hilferty approximation.

    WHAT: ``(X/k)^(1/3)`` is treated as normal with mean ``1 - 2/(9k)`` and variance ``2/(9k)``. The result is an approximate ``P(X > statistic)``.
    WHY: Missingness and Benford need a p-value, and the core does not import SciPy.
    WHY NOT: SciPy's regularised gamma is the exact tail. Tests may compare against it. Shipping that continued fraction in the core is more code than this gate needs.
    ASSUMPTIONS: ``degrees >= 1`` and the chi-square approximation to the original test is already appropriate.
    COMPLEXITY: O(1) time and space.
    FAILS WHEN: Degrees of freedom is 1 and the statistic is huge. A borderline p-value can then flip. Callers still correct for multiple looks.
    """
    if degrees <= 0:
        raise ValueError("degrees must be positive")
    if statistic <= 0:
        return 1.0
    k = float(degrees)
    centre = 1.0 - 2.0 / (9.0 * k)
    scale = math.sqrt(2.0 / (9.0 * k))
    z = ((statistic / k) ** (1.0 / 3.0) - centre) / scale
    return min(1.0, max(0.0, normal_sf(z)))


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini–Hochberg rejections at level ``alpha``.

    WHAT: On sorted p-values, find the largest rank ``k`` with ``p_(k) <= (k/m) alpha`` and reject every p-value up to that rank.
    WHY: Profiling looks at many columns. An uncorrected 0.05 test would report false associations.
    WHY NOT: Bonferroni is simpler and too strict once a sheet has more than a handful of columns.
    ASSUMPTIONS: Each p-value is valid when its null is true.
    COMPLEXITY: O(m log m) time, O(m) space.
    FAILS WHEN: The p-values are biased downward by an approximation. Too many tests then survive.
    """
    count = len(p_values)
    rejected = [False] * count
    if count == 0:
        return rejected
    order = sorted(range(count), key=lambda index: p_values[index])
    cutoff = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= (rank / count) * alpha:
            cutoff = rank
    for rank, index in enumerate(order, start=1):
        if rank <= cutoff:
            rejected[index] = True
    return rejected


def chi_square_independence(table: Sequence[Sequence[int]]) -> tuple[float, float, int] | None:
    """Independence chi-square. Returns ``None`` if any expected count is under 5.

    WHAT: ``sum (observed - expected)^2 / expected``, with expected equal to ``row total * column total / n``.
    WHY: This is the right check for a yes/no missingness flag against a category.
    WHY NOT: A correlation of two dummy codes throws away levels that have more than two values.
    ASSUMPTIONS: One row of the original sheet contributes one count. Expected counts are at least 5.
    COMPLEXITY: O(rows * columns).
    FAILS WHEN: Events from the same case are treated as independent rows. The p-value is then too small.
    """
    if not table or not table[0]:
        return None
    row_count = len(table)
    col_count = len(table[0])
    if row_count < 2 or col_count < 2:
        return None
    row_totals = [sum(row) for row in table]
    total = sum(row_totals)
    if total == 0:
        return None
    col_totals = [sum(table[row][col] for row in range(row_count)) for col in range(col_count)]
    statistic = 0.0
    for row in range(row_count):
        for col in range(col_count):
            expected = row_totals[row] * col_totals[col] / total
            if expected < 5:
                return None
            statistic += (table[row][col] - expected) ** 2 / expected
    degrees = (row_count - 1) * (col_count - 1)
    return statistic, chi2_sf(statistic, degrees), degrees


def mann_whitney_p(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Two-sided Mann–Whitney p-value, normal approximation, ties averaged.

    WHAT: ``U = R1 - n1(n1+1)/2`` is compared with mean ``n1*n2/2``. Variance is ``n1*n2/12 * ((N+1) - sum(t^3-t)/(N(N-1)))``.
    WHY: Fee and duration columns are skewed. Ranks answer whether the missing rows sit elsewhere without using the mean.
    WHY NOT: A two-sample t-test uses the mean, which is the quantity an outlier is already distorting.
    ASSUMPTIONS: The groups are independent and each has at least 8 observations.
    COMPLEXITY: O(n log n) time, O(n) space.
    FAILS WHEN: Either group is smaller than 8, or the variance is 0 because of ties. The function then returns ``None``.
    """
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    n_left, n_right = len(left_values), len(right_values)
    if n_left < 8 or n_right < 8:
        return None
    pooled = left_values + right_values
    order = sorted(range(len(pooled)), key=lambda index: pooled[index])
    ranks = [0.0] * len(pooled)
    tie_cube = 0.0
    cursor = 0
    while cursor < len(pooled):
        end = cursor
        while end + 1 < len(pooled) and pooled[order[end + 1]] == pooled[order[cursor]]:
            end += 1
        average = (cursor + 1 + end + 1) / 2.0
        size = end - cursor + 1
        tie_cube += size**3 - size
        for position in range(cursor, end + 1):
            ranks[order[position]] = average
        cursor = end + 1
    rank_sum = sum(ranks[:n_left])
    u_stat = rank_sum - n_left * (n_left + 1) / 2.0
    total = n_left + n_right
    correction = tie_cube / (total * (total - 1))
    variance = n_left * n_right / 12.0 * ((total + 1) - correction)
    if variance <= 0:
        return None
    z_stat = (u_stat - (n_left * n_right / 2.0)) / math.sqrt(variance)
    return min(1.0, 2.0 * normal_sf(abs(z_stat)))
