"""Wilson interval against the algebra and, when installed, SciPy."""

from __future__ import annotations

import math

import pytest

from pravah.ingest.constants import Z_95
from pravah.ingest.proportion import required_n_for_half_width, wilson_interval


def _algebra(successes: int, total: int, z: float = Z_95) -> tuple[float, float, float]:
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


@pytest.mark.parametrize(
    ("successes", "total"),
    [(5, 5), (4, 5), (8, 9), (11, 12), (81, 100), (0, 10)],
)
def test_wilson_matches_hand_algebra(successes: int, total: int) -> None:
    point, lower, upper = wilson_interval(successes, total)
    expected_point, expected_lower, expected_upper = _algebra(successes, total)
    assert point == pytest.approx(expected_point)
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_five_of_five_clears_a_coin_flip_and_four_of_five_does_not() -> None:
    _point, lower, _upper = wilson_interval(5, 5)
    assert lower > 0.5
    _point, shaky, _upper = wilson_interval(4, 5)
    assert shaky < 0.5


def test_published_81_of_100_bounds() -> None:
    # Computed from the Wilson formula with z = 1.96 and checked by the algebra test.
    _point, lower, upper = wilson_interval(81, 100)
    assert lower == pytest.approx(0.7222097592919214)
    assert upper == pytest.approx(0.8748534600729574)


def test_planning_n_at_the_precedence_boundary_is_73() -> None:
    assert required_n_for_half_width(0.75, 0.10) == 73


def test_planning_n_does_not_collapse_when_the_rate_is_perfect() -> None:
    assert required_n_for_half_width(1.0, 0.10) > 1


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(6, 5)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_scipy_norm_quantile_rounds_to_the_constant_we_use() -> None:
    scipy = pytest.importorskip("scipy")
    quantile = float(scipy.stats.norm.ppf(0.975))
    assert round(quantile, 2) == Z_95
