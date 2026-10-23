"""Wilson interval for a binomial proportion, from scratch.

No SciPy in the core. Tests may compare this with SciPy.
"""

from __future__ import annotations

import math

from pravah.ingest.constants import MIN_ROWS_FOR_INFERENCE, Z_95


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = Z_95,
) -> tuple[float, float, float]:
    """95% Wilson score interval for a proportion.

    WHAT: For ``successes`` out of ``total``, with normal quantile ``z``,
    let ``p = successes / total`` and ``z2 = z^2``. The interval is centred at

        (p + z2 / (2n)) / (1 + z2 / n)

    with half-width

        z * sqrt(p(1-p)/n + z2 / (4n^2)) / (1 + z2 / n).

    The result is clipped to [0, 1] because a proportion cannot leave that
    range; the raw algebra can overshoot by a fraction of a unit in the last
    place when ``p`` is 0 or 1.

    WHY: Ingestion reports rates from small sheets (often n under 30). The
    Wilson interval has better coverage there than the Wald interval, and it
    does not collapse to a fake point when ``p`` is 0 or 1.

    WHY NOT: The Wald interval ``p ± z sqrt(p(1-p)/n)`` is the usual first
    textbook choice and is rejected because its width is 0 when every row
    agrees, which is exactly when a 5-row sheet is most misleading. The
    Clopper-Pearson interval is exact but needs a beta quantile, which would
    pull SciPy into the core. Bootstrap intervals belong to the statistics
    phase and are random; a proportion has a closed form, so resampling would
    add noise without adding information.

    ASSUMPTIONS: Each row is an independent Bernoulli trial with the same
    success probability. ``z`` is the two-sided normal quantile (1.96 ≈ 95%).

    COMPLEXITY: O(1) time and space.

    FAILS WHEN: The rows are dependent (several events from the same case, a
    duplicated file). The interval is then too narrow. It also fails as a
    description of a rate that is not a proportion, such as the weighted
    health score.
    """
    if total <= 0:
        raise ValueError("total must be positive to estimate a proportion")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie between 0 and total inclusive")
    if z <= 0:
        raise ValueError("z must be positive")

    n = float(total)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return p, lower, upper


def required_n_for_half_width(
    proportion: float,
    half_width: float = 0.10,
    *,
    z: float = Z_95,
) -> int:
    """Wald planning sample size, with ``proportion`` pulled away from 0 and 1.

    WHAT: ``n = ceil(z^2 * p * (1-p) / d^2)``, where ``d`` is the half-width.
    ``p`` is clipped to [0.05, 0.95] so a perfect sample does not plan ``n = 0``.

    WHY: A founder asks "how many rows would make this rate precise?" The
    Wald formula answers that in closed form. Clipping ``p`` keeps the answer
    honest when the upload happens to be perfectly clean.

    WHY NOT: Planning with the Wilson width inverted is messier and, at the
    sample sizes we recommend, agrees with Wald within a few rows. Using the
    observed ``p`` unclipped is rejected because ``p = 0`` yields ``n = 0``.

    ASSUMPTIONS: Future rows are independent Bernoulli trials. The clip is a
    planning prior, not a claim about the data.

    COMPLEXITY: O(1) time and space.

    FAILS WHEN: The rate is not a proportion, or rows are clustered inside
    cases. Clustered data needs more rows than this formula says.
    """
    if half_width <= 0 or half_width >= 1:
        raise ValueError("half_width must lie strictly between 0 and 1")
    p = min(0.95, max(0.05, proportion))
    n = (z * z * p * (1.0 - p)) / (half_width * half_width)
    return int(math.ceil(n))


def proportion_is_sufficient(total: int, required: int = MIN_ROWS_FOR_INFERENCE) -> bool:
    """True when ``total`` meets the worst-case ±10 point planning size."""
    return total >= required
