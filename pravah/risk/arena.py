"""Time-based model arena, calibration, and split conformal intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score

from pravah.ingest.constants import RANDOM_SEED

NUMERIC = ["time_since_start", "events_so_far", "time_since_last", "rework_count", "resource_load", "day_of_week"]


def matrix(frame: pd.DataFrame, activities: list[str]) -> np.ndarray:
    """Numeric prefix features plus one column per activity for the latest step."""
    columns = [frame[name].to_numpy(dtype=float) for name in NUMERIC]
    last = frame["last_activity"].astype(str)
    for activity in activities:
        columns.append((last == activity).to_numpy(dtype=float))
    return np.column_stack(columns)


def feature_names(activities: list[str]) -> list[str]:
    """Column names in the same order as ``matrix``."""
    return NUMERIC + [f"last:{activity}" for activity in activities]


def _split_cases(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = frame.groupby("case", sort=False)["start"].min().sort_values().index.tolist()
    n_cases = len(order)
    train_cut = max(1, int(n_cases * 0.6))
    cal_cut = max(train_cut + 1, int(n_cases * 0.8))
    train_ids = set(order[:train_cut])
    cal_ids = set(order[train_cut:cal_cut])
    test_ids = set(order[cal_cut:])
    return frame[frame["case"].isin(train_ids)], frame[frame["case"].isin(cal_ids)], frame[frame["case"].isin(test_ids)]


def _labels(frame: pd.DataFrame, sla: float) -> np.ndarray:
    breach = frame["duration"].to_numpy(dtype=float) > sla
    dropped = frame["dropped"].to_numpy(dtype=int) == 1
    return (breach | dropped).astype(int)


def reliability(y_true: np.ndarray, scores: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    """Bin predicted probabilities and the rate that actually happened."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & (scores <= right if right == 1 else scores < right)
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "low": round(float(left), 2),
                "high": round(float(right), 2),
                "predicted": round(float(scores[mask].mean()), 4),
                "observed": round(float(y_true[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return rows


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _safe_pr(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def run_arena(frame: pd.DataFrame, activities: list[str]) -> dict[str, object]:
    """Compare a naive baseline with logistic regression, a forest, and boosting.

    WHAT: Cases are ordered by their start time. The earliest 60% train, the next 20% calibrate, and the latest 20% test. Classification scores drop or an SLA breach. The SLA is the 75th percentile of durations in the training cases only. Regression scores remaining time. The winner is the model with the higher PR-AUC when the rarer class is under 40% of the test rows, and the higher ROC-AUC otherwise. A reliability gap above 0.10 is corrected with isotonic regression fit on the calibration cases. Remaining-time intervals use split conformal residuals at 90% coverage.
    WHY: A random split lets a later event teach the model about an earlier case of the same week. Time order is the order the file will arrive in production. PR-AUC is the right ranking when dropouts are rare, because ROC-AUC can look strong while every dropout is missed.
    WHY NOT: Accuracy. A model that calls every case safe scores high when dropouts are rare and is useless. A single random forest with no baseline would hide whether the extra machinery beat the training rate.
    ASSUMPTIONS: ``RANDOM_SEED`` is 42. Forests and boosting use 40 trees and depth 3. The conformal quantile is ``ceil((n+1) * 0.9) / n`` of the absolute calibration residuals. Isotonic is skipped when the calibration slice has one class.
    COMPLEXITY: Tree fitting dominates, roughly O(trees * n log n * p). Space O(n p).
    FAILS WHEN: The test slice has one class, so AUC is null and that model is not chosen. Fewer than three cases cannot fill three time slices; the caller refuses before this function.
    """
    train, calibrate, test = _split_cases(frame[frame["open"] == 0])
    if train.empty or test.empty:
        return {"trained": False, "sentence": "The time split has no finished cases in both the past and the future."}
    sla = float(np.quantile(train.groupby("case")["duration"].first(), 0.75))
    y_train = _labels(train, sla)
    y_test = _labels(test, sla)
    x_train = matrix(train, activities)
    x_test = matrix(test, activities)
    rate = float(y_train.mean())
    imbalanced = min(rate, 1.0 - rate) < 0.4
    classifiers = {
        "naive baseline": _prior(rate),
        "logistic regression": LogisticRegression(max_iter=400, random_state=RANDOM_SEED),
        "random forest": RandomForestClassifier(n_estimators=40, max_depth=3, random_state=RANDOM_SEED),
        "gradient boosting": GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=RANDOM_SEED),
    }
    rows = []
    fitted: dict[str, object] = {}
    raw_test: dict[str, np.ndarray] = {}
    for name, model in classifiers.items():
        if name == "naive baseline":
            scores = np.full(len(test), rate)
        else:
            if len(set(y_train.tolist())) < 2:
                continue
            model.fit(x_train, y_train)
            scores = model.predict_proba(x_test)[:, 1]
            fitted[name] = model
        raw_test[name] = scores
        auc = _safe_auc(y_test, scores)
        pr = _safe_pr(y_test, scores)
        rows.append(
            {
                "model": name,
                "roc_auc": None if auc is None else round(auc, 4),
                "pr_auc": None if pr is None else round(pr, 4),
                "brier": round(float(brier_score_loss(y_test, np.clip(scores, 0, 1))), 4),
            }
        )
    key = "pr_auc" if imbalanced else "roc_auc"
    ranked = [row for row in rows if row[key] is not None]
    winner = max(ranked, key=lambda row: float(row[key]))["model"] if ranked else None
    reason = (
        "Classes are imbalanced, so the winner is the higher PR-AUC. ROC-AUC can look healthy while the rare dropout is missed."
        if imbalanced
        else "The classes are not rare, so the winner is the higher ROC-AUC."
    )
    curve = reliability(y_test, raw_test[winner]) if winner and winner in raw_test else []
    gap = max((abs(point["predicted"] - point["observed"]) for point in curve), default=0.0)
    calibrator = None
    if winner and winner in fitted and gap > 0.10 and not calibrate.empty and len(set(_labels(calibrate, sla).tolist())) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        raw_cal = fitted[winner].predict_proba(matrix(calibrate, activities))[:, 1]
        calibrator.fit(raw_cal, _labels(calibrate, sla))
        adjusted = calibrator.predict(raw_test[winner])
        curve = reliability(y_test, adjusted)
        for row in rows:
            if row["model"] == winner:
                row["brier"] = round(float(brier_score_loss(y_test, np.clip(adjusted, 0, 1))), 4)
    regressors = _regression(train, test, activities)
    conformal = _conformal(train, calibrate, test, activities)
    importance = _importance(fitted.get(winner), x_test, y_test, activities) if winner else []
    return {
        "trained": True,
        "imbalanced": imbalanced,
        "selection": reason,
        "winner": winner,
        "sla_days": round(sla, 3),
        "models": rows,
        "calibration": curve,
        "calibrated": calibrator is not None,
        "regression": regressors,
        "conformal": conformal,
        "importance": importance,
        "activities": activities,
        "_fitted": fitted,
        "_calibrator": calibrator,
        "_rate": rate,
        "_sla": sla,
    }


def _prior(rate: float):
    return None


def _importance(model, x_test: np.ndarray, y_test: np.ndarray, activities: list[str]) -> list[dict[str, object]]:
    """How much the Brier score worsens when one column is shuffled.

    WHAT: Fit is left untouched. Each column of the test matrix is permuted in turn and the rise in Brier score is recorded.
    WHY: A global ranking answers which fact the model uses. The per-case Shapley values answer a different question and can disagree on one student.
    WHY NOT: The absolute size of a coefficient. A one-hot activity column and a day count are not on the same scale.
    ASSUMPTIONS: The seed is the project seed. The naive baseline has no columns to shuffle, so the list is empty.
    COMPLEXITY: O(p) extra predictions. Space O(n p).
    FAILS WHEN: The winner was not fitted. The list is empty.
    """
    if model is None or len(y_test) < 5:
        return []
    names = feature_names(activities)
    base = float(brier_score_loss(y_test, np.clip(model.predict_proba(x_test)[:, 1], 0, 1)))
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for index, name in enumerate(names):
        shuffled = x_test.copy()
        rng.shuffle(shuffled[:, index])
        score = float(brier_score_loss(y_test, np.clip(model.predict_proba(shuffled)[:, 1], 0, 1)))
        rows.append({"name": name, "brier_increase": round(score - base, 4)})
    rows.sort(key=lambda row: -float(row["brier_increase"]))
    return rows[:8]


def _regression(train: pd.DataFrame, test: pd.DataFrame, activities: list[str]) -> list[dict[str, object]]:
    y_train = train["remaining"].to_numpy(dtype=float)
    y_test = test["remaining"].to_numpy(dtype=float)
    x_train = matrix(train, activities)
    x_test = matrix(test, activities)
    baseline = float(y_train.mean())
    models = {
        "naive baseline": None,
        "linear regression": LinearRegression(),
        "random forest": RandomForestRegressor(n_estimators=40, max_depth=3, random_state=RANDOM_SEED),
        "gradient boosting": GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=RANDOM_SEED),
    }
    rows = []
    base_mae = float(mean_absolute_error(y_test, np.full(len(test), baseline)))
    for name, model in models.items():
        if model is None:
            pred = np.full(len(test), baseline)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        mae = float(mean_absolute_error(y_test, pred))
        rows.append({"model": name, "mae": round(mae, 4), "baseline_mae": round(base_mae, 4), "beats_baseline": mae < base_mae - 1e-9})
    return rows


def _conformal(train: pd.DataFrame, calibrate: pd.DataFrame, test: pd.DataFrame, activities: list[str]) -> dict[str, object]:
    if calibrate.empty or test.empty:
        return {"coverage": None, "quantile": None, "sentence": "The calibration slice is empty, so no conformal interval is claimed."}
    model = GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=RANDOM_SEED)
    model.fit(matrix(train, activities), train["remaining"].to_numpy(dtype=float))
    residuals = np.abs(calibrate["remaining"].to_numpy(dtype=float) - model.predict(matrix(calibrate, activities)))
    n_cal = len(residuals)
    level = math.ceil((n_cal + 1) * 0.9) / n_cal
    quantile = float(np.quantile(residuals, min(1.0, level)))
    error = np.abs(test["remaining"].to_numpy(dtype=float) - model.predict(matrix(test, activities)))
    coverage = float(np.mean(error <= quantile))
    return {
        "coverage": round(coverage, 4),
        "target": 0.9,
        "quantile": round(quantile, 4),
        "sentence": f"The 90% conformal interval covered {coverage:.0%} of the held-out remaining times.",
        "_model": model,
    }
