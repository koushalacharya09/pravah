"""Exact Shapley values against the training-mean background."""

from __future__ import annotations

import itertools

import numpy as np


def exact_shapley(predict, row: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Share of one prediction among its features, averaged over every order.

    WHAT: The value of a coalition is the model at the row's values inside the coalition and the background values outside it. A feature's Shapley value is its average marginal contribution over the subsets of the other features. The values plus the background prediction equal the row's prediction.
    WHY: Permutation importance says which column moves the score on average. A founder looking at one open case needs the share of that case's score. Averaging over orders stops a correlated column from taking the whole credit just because it was entered first.
    WHY NOT: A single TreeSHAP implementation tied to one library. The feature count here is the small set allowed by the events-per-variable rule, so the 2^p sum is the definition rather than an approximation. A product of coefficients would only be right for a linear model.
    ASSUMPTIONS: ``predict`` accepts a 2-d array and returns one number per row. The background is one vector, the training mean. Features are treated as the players. This is interventional Shapley at that single background, not an average over every training row.
    COMPLEXITY: O(2^p) model calls. Space O(p).
    FAILS WHEN: More than 12 features. The caller must not ask, because 2^12 calls per case is the limit this page will spend. A model that is not a function of its inputs makes the values fail to add up.
    """
    features = len(row)
    if features > 12:
        raise ValueError("Exact Shapley is limited to 12 features.")
    baseline = float(np.asarray(predict(background.reshape(1, -1))).reshape(-1)[0])
    values = np.zeros(features)
    for index in range(features):
        others = [position for position in range(features) if position != index]
        total = 0.0
        weight = 0.0
        for size in range(len(others) + 1):
            for coalition in itertools.combinations(others, size):
                without = background.copy()
                with_feature = background.copy()
                for member in coalition:
                    without[member] = row[member]
                    with_feature[member] = row[member]
                with_feature[index] = row[index]
                before = float(np.asarray(predict(without.reshape(1, -1))).reshape(-1)[0])
                after = float(np.asarray(predict(with_feature.reshape(1, -1))).reshape(-1)[0])
                ways = math_comb(len(others), size)
                share = 1.0 / (features * ways)
                total += share * (after - before)
                weight += share
        values[index] = total
    _ = baseline, weight
    return values


def math_comb(n_items: int, chosen: int) -> int:
    """Binomial coefficient, used as the Shapley subset weight."""
    if chosen < 0 or chosen > n_items:
        return 0
    result = 1
    for step in range(chosen):
        result = result * (n_items - step) // (step + 1)
    return result
