"""Ask whether missing cells line up with other columns."""

from __future__ import annotations

import pandas as pd

from pravah.stats.inference import benjamini_hochberg, chi_square_independence, mann_whitney_p


def _categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series.nunique(dropna=True) <= 8
    return series.nunique(dropna=True) <= 20


def missingness_report(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Label each partly-missing column as likely MCAR or not MCAR.

    WHAT: Build a missingness flag. Against each other categorical column, run a chi-square test when every expected count is at least 5. Against each other numeric column, run a Mann–Whitney test when both sides have at least 8 values. Benjamini–Hochberg at 0.05 decides which tests count. A column is not MCAR when any test survives. It is likely MCAR when every valid test fails to survive.
    WHY: "12% missing" does not say whether the average of the filled rows is biased. If demo dates are missing mainly for one centre, the filled average describes that centre's complement.
    WHY NOT: Little's MCAR test needs a multivariate normal assumption and a covariance matrix. These sheets are small, mixed type, and not normal. A logistic regression of the flag is a reasonable alternative and is rejected because it needs a fitted model and a penalty choice before the simple tests have been reported.
    ASSUMPTIONS: Each row is one independent record. A column is tested only when its missing rate is between 5% and 80%. "Not MCAR" means the flag is related to an observed column. It does not prove the missingness is MAR rather than MNAR.
    COMPLEXITY: O(m^2 n) time for ``m`` columns and ``n`` rows, dominated by the rank tests. Space O(n).
    FAILS WHEN: The rows are events of the same case, so one student's missingness is counted many times. The test then overstates the evidence. It also stays silent when every other column fails the expected-count or sample-size rule.
    """
    reports: list[dict[str, object]] = []
    if frame.empty:
        return reports
    candidates = []
    for name in frame.columns:
        missing_rate = float(frame[name].isna().mean())
        if 0.05 <= missing_rate <= 0.80:
            candidates.append(str(name))
    if not candidates:
        return [
            {
                "column": None,
                "label": "not tested",
                "sentence": "No column is missing often enough to test whether the gaps are random.",
            }
        ]
    tests: list[dict[str, object]] = []
    for name in candidates:
        flag = frame[name].isna()
        for other in frame.columns:
            if str(other) == name:
                continue
            other_series = frame[other]
            if _categorical(other_series):
                levels = [level for level in other_series.dropna().unique().tolist()][:12]
                table = []
                for level in levels:
                    in_level = other_series.eq(level) if not pd.api.types.is_datetime64_any_dtype(other_series) else other_series.eq(level)
                    present = int((~flag & in_level).sum())
                    missing = int((flag & in_level).sum())
                    table.append([present, missing])
                tested = chi_square_independence(table)
                if tested is None:
                    continue
                _stat, p_value, _df = tested
                worst_level = None
                worst_rate = -1.0
                for level in levels:
                    in_level = other_series.eq(level)
                    support = int(in_level.sum())
                    if support < 5:
                        continue
                    level_rate = int((flag & in_level).sum()) / support
                    if level_rate > worst_rate:
                        worst_rate = level_rate
                        worst_level = level
                tests.append(
                    {
                        "column": name,
                        "other": str(other),
                        "p_value": p_value,
                        "kind": "category",
                        "level": None if worst_level is None else str(worst_level),
                        "level_rate": worst_rate,
                    }
                )
            elif pd.api.types.is_numeric_dtype(other_series):
                present_values = other_series[~flag].dropna().tolist()
                missing_values = other_series[flag].dropna().tolist()
                p_value = mann_whitney_p(present_values, missing_values)
                if p_value is None:
                    continue
                tests.append({"column": name, "other": str(other), "p_value": p_value, "kind": "number"})
    rejections = benjamini_hochberg([float(test["p_value"]) for test in tests]) if tests else []
    linked: dict[str, list[dict[str, object]]] = {name: [] for name in candidates}
    for test, rejected in zip(tests, rejections):
        if rejected:
            linked[str(test["column"])].append(test)
    for name in candidates:
        rate = float(frame[name].isna().mean())
        hits = linked[name]
        if hits:
            lead = hits[0]
            if lead.get("level"):
                sentence = (
                    f"{name} is missing on {rate:.0%} of rows, and {float(lead['level_rate']):.0%} of "
                    f"{lead['level']} rows are missing it. The gaps line up with {lead['other']}, "
                    "so the data is not missing at random and an average of the filled rows will be biased."
                )
            else:
                joined = ", ".join(str(hit["other"]) for hit in hits[:3])
                sentence = (
                    f"{name} is missing on {rate:.0%} of rows, mostly in step with {joined}. "
                    "The gaps are not random, so an average of the filled rows will be biased."
                )
            label = "not MCAR"
        elif not tests:
            sentence = (
                f"{name} is missing on {rate:.0%} of rows. The other columns are too small "
                "or too uneven to test whether those gaps are random."
            )
            label = "not tested"
        else:
            sentence = (
                f"{name} is missing on {rate:.0%} of rows. Those gaps do not line up with the "
                "other columns that could be tested, so the filled rows are closer to a random subset."
            )
            label = "likely MCAR"
        reports.append({"column": name, "label": label, "missing_rate": round(rate, 3), "sentence": sentence})
    return reports
