"""Two-group log-rank tests, then Benjamini–Hochberg across the set."""

from __future__ import annotations

import numpy as np

from pravah.stats.inference import benjamini_hochberg, chi2_sf


def logrank(durations: np.ndarray, observed: np.ndarray, groups: np.ndarray) -> tuple[float, float] | None:
    """Mantel–Haenszel log-rank statistic and a chi-square p-value.

    WHAT: At each time where either group has an event, the first group's expected deaths are ``n1 * deaths / n``. The statistic is ``(observed - expected)^2 / variance``, with variance ``n1 * n2 * deaths * (n - deaths) / (n * n * (n - 1))``. The p-value is the chi-square upper tail on 1 degree of freedom.
    WHY: Two Kaplan–Meier curves can look apart because of one early case. The log-rank adds the differences at every event time, weighted by how many cases were still at risk.
    WHY NOT: A Wilcoxon test that weights early times more heavily. The question here is the whole curve, not only the first week. A plain t-test on the durations treats censored times as finished times.
    ASSUMPTIONS: The two labels are the only groups. Censoring is independent of the group given what is in the file. The chi-square tail uses the Wilson–Hilferty approximation already used for Benford.
    COMPLEXITY: O(n log n) time, O(n) space.
    FAILS WHEN: One group never has anyone at risk at an event time, so the variance is zero and no test is returned. A crossing of the curves can leave a small statistic even though the curves differ.
    """
    times = np.asarray(durations, dtype=float)
    flags = np.asarray(observed, dtype=int)
    labels = np.asarray(groups)
    levels = [level for level in dict.fromkeys(labels.tolist())]
    if len(levels) != 2:
        return None
    first = levels[0]
    order = np.argsort(times, kind="mergesort")
    times, flags, labels = times[order], flags[order], labels[order]
    observed_diff = 0.0
    variance = 0.0
    cursor = 0
    total = len(times)
    while cursor < total:
        moment = times[cursor]
        deaths = 0
        deaths_first = 0
        while cursor < total and times[cursor] == moment:
            if labels[cursor] == first:
                deaths_first += int(flags[cursor] == 1)
            deaths += int(flags[cursor] == 1)
            cursor += 1
        n1 = int(np.sum((times >= moment) & (labels == first)))
        n2 = int(np.sum((times >= moment) & (labels != first)))
        n = n1 + n2
        if deaths == 0 or n <= 1 or n1 == 0 or n2 == 0:
            continue
        expected = n1 * deaths / n
        observed_diff += deaths_first - expected
        variance += n1 * n2 * deaths * (n - deaths) / (n * n * (n - 1))
    if variance <= 0:
        return None
    statistic = observed_diff**2 / variance
    return statistic, chi2_sf(statistic, 1)


def compare_segments(
    durations: np.ndarray,
    observed: np.ndarray,
    columns: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Pairwise log-rank tests against the largest level, corrected together.

    WHAT: Each segment column is split into its levels. Every smaller level is compared with the largest level. When more than one comparison is made, Benjamini–Hochberg at 0.05 decides which differences are kept.
    WHY: Area, resource, and channel are the same kind of question asked several times. Reporting every raw 0.05 test would crown a noisy level.
    WHY NOT: One multi-group log-rank. A single p-value would not say which level differs, and the page has to name the level. Bonferroni was rejected in the missingness work for the same reason it is rejected here.
    ASSUMPTIONS: The largest level is the reference. Levels with fewer than 3 cases are skipped. At most the first 4 columns passed in are tested.
    COMPLEXITY: O(c n log n) for ``c`` comparisons.
    FAILS WHEN: Every level is smaller than 3. The column is then absent. A column that is a case id in disguise has too many levels and must be filtered before this function.
    """
    pending: list[dict[str, object]] = []
    curves_by_column: dict[str, list[str]] = {}
    for name, values in list(columns.items())[:4]:
        text = np.asarray(values).astype(str)
        levels, counts = np.unique(text, return_counts=True)
        if len(levels) < 2:
            continue
        order = np.argsort(-counts)
        reference = str(levels[order[0]])
        curves_by_column[name] = [str(level) for level in levels[order]]
        for level, count in zip(levels[order][1:], counts[order][1:]):
            if int(count) < 3 or int(counts[order[0]]) < 3:
                continue
            mask = np.isin(text, [reference, str(level)])
            result = logrank(durations[mask], observed[mask], text[mask])
            if result is None:
                continue
            statistic, p_value = result
            pending.append(
                {
                    "column": name,
                    "level": str(level),
                    "reference": reference,
                    "statistic": round(float(statistic), 4),
                    "p": float(p_value),
                }
            )
    rejected = benjamini_hochberg([float(row["p"]) for row in pending]) if len(pending) > 1 else [float(row["p"]) < 0.05 for row in pending]
    grouped: dict[str, dict[str, object]] = {}
    for row, keep in zip(pending, rejected):
        bucket = grouped.setdefault(
            str(row["column"]),
            {"column": row["column"], "reference": row["reference"], "comparisons": []},
        )
        bucket["comparisons"].append(
            {
                "level": row["level"],
                "reference": row["reference"],
                "statistic": row["statistic"],
                "p": round(float(row["p"]), 4),
                "rejected": bool(keep),
            }
        )
    return list(grouped.values())
