"""Weekly arrivals, a seasonal decomposition, and a seasonal-naive baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import SchemaResult


def weekly_counts(frame: pd.DataFrame, schema: SchemaResult) -> pd.Series:
    """One count per week, using the first timestamp of each case."""
    if not (schema.case_id and schema.timestamp):
        return pd.Series(dtype=float)
    if schema.case_id not in frame.columns or schema.timestamp not in frame.columns:
        return pd.Series(dtype=float)
    working = frame.dropna(subset=[schema.case_id, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    first = working.groupby(schema.case_id)[schema.timestamp].min()
    if first.empty:
        return pd.Series(dtype=float)
    weeks = first.dt.to_period("W-SUN").dt.start_time
    counts = weeks.value_counts().sort_index()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="W-MON")
    return counts.reindex(full, fill_value=0).astype(float)


def seasonal_forecast(counts: pd.Series, period: int = 4) -> dict[str, object]:
    """Forecast next weeks and say whether decomposition beat the seasonal naive.

    WHAT: The seasonal naive repeats the value from ``period`` weeks earlier. The decomposition subtracts a  ``period`` rolling mean and averages the leftover by position in the cycle, then adds that seasonal piece back to the last trend. Both are scored by mean absolute error on the last ``period`` weeks. The forecast that is shipped is the one with the lower error.
    WHY: A seasonal model that loses to "same as last month" has not earned the extra sentence. The comparison is the claim.
    WHY NOT: A fitted SARIMA. Four to twelve weeks cannot identify that many coefficients, and the naive baseline would be missing.
    ASSUMPTIONS: The period is 4 weeks. Fewer than 2 periods means neither model is scored and the forecast is the mean.
    COMPLEXITY: O(n) time. Space O(n).
    FAILS WHEN: There are no weekly counts. The forecast is empty and the simulation keeps the observed arrival volume.
    """
    values = counts.to_numpy(dtype=float)
    if values.size == 0:
        return {"ready": False, "sentence": "No weekly arrivals to forecast.", "next": []}
    if values.size < period * 2:
        mean = float(values.mean())
        return {
            "ready": True,
            "beats_naive": False,
            "sentence": f"Only {values.size} weeks. Seasonal decomposition needs {period * 2}, so the forecast is the mean of {mean:.1f} arrivals.",
            "next": [round(mean, 3)] * period,
            "decomposition_mae": None,
            "naive_mae": None,
        }
    naive = values[period:]
    naive_error = np.abs(naive - values[:-period])
    trend = pd.Series(values).rolling(period, min_periods=1).mean().to_numpy()
    detrended = values - trend
    seasonal = np.zeros(period)
    for position in range(period):
        seasonal[position] = float(detrended[position::period].mean())
    fitted = trend + np.array([seasonal[index % period] for index in range(len(values))])
    hold = slice(-period, None)
    dec_error = np.abs(values[hold] - fitted[hold])
    dec_mae = float(dec_error.mean())
    naive_mae = float(naive_error[-period:].mean())
    beats = dec_mae < naive_mae - 1e-9
    last_trend = float(trend[-1])
    if beats:
        nxt = [max(0.0, last_trend + float(seasonal[(len(values) + step) % period])) for step in range(period)]
        sentence = f"Seasonal decomposition MAE {dec_mae:.2f} beat the seasonal naive MAE {naive_mae:.2f}."
    else:
        nxt = [max(0.0, float(values[-(period - step)])) for step in range(period)]
        sentence = f"Seasonal decomposition MAE {dec_mae:.2f} did not beat the seasonal naive MAE {naive_mae:.2f}, so the naive forecast is used."
    return {
        "ready": True,
        "beats_naive": beats,
        "sentence": sentence,
        "next": [round(value, 3) for value in nxt],
        "decomposition_mae": round(dec_mae, 4),
        "naive_mae": round(naive_mae, 4),
    }
