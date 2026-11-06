"""Model pipelines, candidate ladder and metrics.

Every learned transform (target encoding, imputation, scaling) lives inside the
sklearn ``Pipeline`` and is fit on training rows only. The target is
``log(price)``; predictions are converted back with ``exp`` (an estimate of the
conditional median asking price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBRegressor

from data import RANDOM_STATE
from features import CATEGORICAL_COL, FEATURE_COLS

# ----------------------------------------------------------------------------- feature sets
# "pr1": the 31 features of PR #1 (street width imputed with the constant 15 m, as before).
# "v2":  drops the constant is_villa (100% of rows) and two regex flags that match
#        under 1% of rows; street width imputed with the train median + missing indicator.
_RARE_OR_CONSTANT = ["is_villa", "has_double_height", "elevator_installed"]
FEATURE_SETS = {
    "pr1": {"cols": FEATURE_COLS, "street_width": "constant15"},
    "v2": {"cols": [c for c in FEATURE_COLS if c not in _RARE_OR_CONSTANT], "street_width": "median+indicator"},
}
MISSING_CAPABLE = ["bedrooms", "bathrooms"]  # NaN only at serving time if a caller omits them


def model_input_cols(feature_set: str) -> list:
    return [CATEGORICAL_COL] + FEATURE_SETS[feature_set]["cols"]


def make_preprocessor(feature_set: str, scale: bool = False) -> ColumnTransformer:
    cols = FEATURE_SETS[feature_set]["cols"]
    encoder = TargetEncoder(smooth="auto", cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
    if FEATURE_SETS[feature_set]["street_width"] == "constant15":
        sw = SimpleImputer(strategy="constant", fill_value=15.0)
    else:
        sw = SimpleImputer(strategy="median", add_indicator=True)
    rest = [c for c in cols if c != "street_width"]
    transformers = [
        ("nh", encoder, [CATEGORICAL_COL]),
        ("sw", sw, ["street_width"]),
        ("num", SimpleImputer(strategy="median"), rest),
    ]
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    if scale:
        return Pipeline([("ct", ct), ("scale", StandardScaler())])
    return ct


def make_pipeline(regressor, feature_set: str = "v2", scale: bool = False) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(feature_set, scale)), ("model", regressor)])


# ----------------------------------------------------------------------------- relative model
class AnchoredRegressor(BaseEstimator, RegressorMixin):
    """Fits ``inner`` on ``log(price) - log(anchor)`` where the anchor is the
    external Aqar index value for (district, size bucket) at the 2015-16 level.
    ``predict`` returns ``log(price)`` so it is a drop-in candidate."""

    def __init__(self, inner=None, mapper=None):
        self.inner = inner
        self.mapper = mapper

    def _log_anchor(self, X: pd.DataFrame) -> np.ndarray:
        bm = self.mapper.benchmarks_frame(X[CATEGORICAL_COL], X["size"])
        return np.log(bm["hist_price_base"].to_numpy(float))

    def fit(self, X, y):
        self.inner_ = clone(self.inner).fit(X, np.asarray(y, float) - self._log_anchor(X))
        return self

    def predict(self, X):
        return self.inner_.predict(X) + self._log_anchor(X)


# ----------------------------------------------------------------------------- candidates
GBR_PR1_PARAMS = dict(n_estimators=160, max_depth=4, min_samples_leaf=12, min_samples_split=24,
                      subsample=0.85, max_features=0.85, learning_rate=0.05, random_state=RANDOM_STATE)
XGB_PR1_PARAMS = dict(n_estimators=140, max_depth=4, min_child_weight=8, subsample=0.85,
                      colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=2.0, learning_rate=0.05,
                      random_state=RANDOM_STATE, n_jobs=1)
HGB_DEFAULT_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                          l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)


def candidates(feature_set: str = "v2", mapper=None) -> dict:
    """Baseline ladder in order of complexity (simplest first)."""
    c = {
        "dummy_median": make_pipeline(DummyRegressor(strategy="median"), feature_set),
        "ridge": make_pipeline(Ridge(alpha=10.0), feature_set, scale=True),
        "gbr_pr1_params": make_pipeline(GradientBoostingRegressor(**GBR_PR1_PARAMS), feature_set),
        "hgb": make_pipeline(HistGradientBoostingRegressor(**HGB_DEFAULT_PARAMS), feature_set),
        "xgb_pr1_params": make_pipeline(XGBRegressor(**XGB_PR1_PARAMS), feature_set),
    }
    if mapper is not None:
        c["relative_gbr"] = AnchoredRegressor(
            make_pipeline(GradientBoostingRegressor(**GBR_PR1_PARAMS), feature_set), mapper)
    return c


COMPLEXITY_ORDER = ["dummy_median", "ridge", "gbr_pr1_params", "hgb", "xgb_pr1_params",
                    "hgb_tuned", "xgb_tuned", "relative_gbr"]

HGB_SEARCH_SPACE = {
    "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "model__max_iter": [200, 300, 500, 800],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 40, 80],
    "model__l2_regularization": [0.0, 0.5, 1.0, 3.0, 10.0],
    "model__max_features": [0.5, 0.7, 1.0],
}
XGB_SEARCH_SPACE = {
    "model__n_estimators": [200, 400, 700],
    "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_child_weight": [1, 4, 8, 16],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.6, 0.85, 1.0],
    "model__reg_lambda": [0.5, 2.0, 5.0, 10.0],
}


# ----------------------------------------------------------------------------- metrics
def abs_log_error(price_true, price_pred) -> np.ndarray:
    """Per-item loss used for every paired comparison (symmetric for over/under)."""
    return np.abs(np.log(np.asarray(price_pred, float) / np.asarray(price_true, float)))


def price_metrics(price_true, price_pred) -> dict:
    t, p = np.asarray(price_true, float), np.asarray(price_pred, float)
    ape = np.abs(p - t) / t
    le = np.log(p / t)
    lt, lp = np.log(t), np.log(p)
    r2 = 1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)
    return {
        "n": int(t.size),
        "mean_abs_log_error": float(np.mean(np.abs(le))),
        "MAPE_pct": float(ape.mean() * 100),
        "MdAPE_pct": float(np.median(ape) * 100),
        "within_10pct_share": float((ape <= 0.10).mean()),
        "within_20pct_share": float((ape <= 0.20).mean()),
        "median_bias_pct": float((np.exp(np.median(le)) - 1) * 100),
        "R2_log_price": float(r2),
        "MAE_SAR": float(np.mean(np.abs(p - t))),
    }
