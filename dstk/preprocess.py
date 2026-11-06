"""Pipeline-safe outlier handling: statistics are learned on TRAIN only."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [lower, upper] quantiles learned in fit(). Put it inside the Pipeline
    so cross-validation refits the bounds per fold. Never drops rows (row counts stay equal)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = np.asarray(df.columns, dtype=object)
        self.lo_ = df.quantile(self.lower).to_numpy(dtype=float)
        self.hi_ = df.quantile(self.upper).to_numpy(dtype=float)
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        arr = df.to_numpy(dtype=float)
        return np.clip(arr, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def fit_iqr_bounds(train: pd.DataFrame, cols: Sequence[str], k: float = 1.5) -> Dict[str, Tuple[float, float]]:
    """Tukey fences learned on train: (Q1 - k*IQR, Q3 + k*IQR) per column."""
    out = {}
    for c in cols:
        q1, q3 = train[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        out[c] = (float(q1 - k * iqr), float(q3 + k * iqr))
    return out


def robust_zscore(x: pd.Series, center: float | None = None, mad: float | None = None) -> pd.Series:
    """0.6745 * (x - median) / MAD. Pass center/mad learned on train when scoring other data."""
    center = float(x.median()) if center is None else center
    mad = float((x - x.median()).abs().median()) if mad is None else mad
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - center) / mad


def outlier_flags(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Boolean frame: True where a value lies outside its train-learned fences."""
    return pd.DataFrame({c: (df[c] < lo) | (df[c] > hi) for c, (lo, hi) in bounds.items()}, index=df.index)


def outlier_report(train: pd.DataFrame, other: pd.DataFrame, cols: Sequence[str], k: float = 1.5) -> pd.DataFrame:
    """Counts and shares of values outside train-learned Tukey fences, for train and `other`."""
    b = fit_iqr_bounds(train, cols, k)
    ft, fo = outlier_flags(train, b), outlier_flags(other, b)
    return pd.DataFrame({"lo": {c: b[c][0] for c in cols}, "hi": {c: b[c][1] for c in cols},
                         "train_n": ft.sum(), "train_share": ft.mean(),
                         "other_n": fo.sum(), "other_share": fo.mean()}).loc[list(cols)]
