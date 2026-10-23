"""0/1 knapsack by dynamic programming, with the greedy contrast."""

from __future__ import annotations


def knapsack(values: list[float], costs: list[int], budget: int) -> dict[str, object]:
    """Choose a subset with integer costs that maximises value inside a budget.

    WHAT: ``best[c]`` is the highest value achievable with budget ``c``. Each item is considered once, from the top of the budget downward, so it cannot be picked twice. The greedy order is value divided by cost, walking until the next item does not fit. When the two subsets differ, the result names both totals.
    WHY: A ratio sort is the heuristic people do by hand. The dynamic program is the one that is allowed to skip a shiny cheap action for two actions that fit better together.
    WHY NOT: A greedy sort presented as optimal. The classic counter-example, values 60, 100, 120 and costs 10, 20, 30 inside budget 50, is the reason: greedy keeps 160 and the program keeps 220.
    ASSUMPTIONS: Costs are positive integers. Values may be fractional. Budget is a non-negative integer. Item identity is its position.
    COMPLEXITY: O(items * budget) time and space.
    FAILS WHEN: A cost is zero or negative. That item is skipped, because a free action is not a knapsack decision and would loop the update.
    """
    limit = max(0, int(budget))
    best = [0.0] * (limit + 1)
    choice: list[list[int]] = [[] for _ in range(limit + 1)]
    for index, (value, cost) in enumerate(zip(values, costs)):
        if cost <= 0 or cost > limit or value <= 0:
            continue
        for money in range(limit, cost - 1, -1):
            candidate = best[money - cost] + float(value)
            if candidate > best[money] + 1e-12:
                best[money] = candidate
                choice[money] = choice[money - cost] + [index]
    picked = choice[limit]
    greedy = _greedy(values, costs, limit)
    greedy_value = sum(float(values[index]) for index in greedy)
    return {
        "chosen": picked,
        "value": round(float(best[limit]), 4),
        "greedy": greedy,
        "greedy_value": round(greedy_value, 4),
        "differs": picked != greedy,
    }


def _greedy(values: list[float], costs: list[int], budget: int) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (values[index] / costs[index]) if costs[index] > 0 else -1, reverse=True)
    left = budget
    picked = []
    for index in order:
        if costs[index] <= 0 or values[index] <= 0:
            continue
        if costs[index] <= left:
            picked.append(index)
            left -= costs[index]
    return picked
