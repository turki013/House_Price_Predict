"""Forecast evaluation helpers: baselines, scale-free metrics, rolling-origin splits."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def seasonal_naive(y_train, horizon: int, season: int = 1) -> np.ndarray:
    """Forecast = the last observed season repeated. season=1 gives the plain last-value (naive) forecast."""
    y = np.asarray(y_train, dtype=float)
    if y.size < season:
        raise ValueError("need at least one full season of history")
    last = y[-season:]
    return np.tile(last, int(np.ceil(horizon / season)))[:horizon]


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (0 when both are zero). Prefer WAPE/MASE when values can be near zero."""
    a, f = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(a) + np.abs(f)) / 2
    return float(100 * np.mean(np.where(denom == 0, 0.0, np.abs(a - f) / np.where(denom == 0, 1, denom))))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error in percent: sum|error| / sum|actual|."""
    a, f = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.abs(a - f).sum() / np.abs(a).sum())


def mase(y_true, y_pred, y_train, season: int = 1) -> float:
    """Mean absolute scaled error: MAE divided by the in-sample MAE of the (seasonal) naive forecast.
    Below 1 means better than the naive baseline on the training scale."""
    y = np.asarray(y_train, float)
    scale = np.mean(np.abs(y[season:] - y[:-season]))
    if scale == 0:
        raise ValueError("naive in-sample error is zero; MASE undefined")
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))) / scale)


def rolling_origin_splits(n: int, initial: int, horizon: int, step: int | None = None
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window backtest: train = [0, origin), test = [origin, origin + horizon)."""
    step = step or horizon
    out, origin = [], initial
    while origin + horizon <= n:
        out.append((np.arange(0, origin), np.arange(origin, origin + horizon)))
        origin += step
    if not out:
        raise ValueError("series too short for the requested initial window and horizon")
    return out
