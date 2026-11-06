"""Project hooks integrating House_Price_Predict with dstk test suite.

Allows test_pipeline_parity, test_data_leakage, and test_model_regression
to test the actual project implementation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from data import cv_splits, split_frame
from dstk.leakage import frame_fingerprint
from features import ALL_INPUT_COLS, RAW_INPUT_COLS, build_feature_frame, extract_features
from modeling import abs_log_error, model_input_cols

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _split_cache():
    return split_frame()


def get_split() -> dict:
    s = _split_cache()
    cols = model_input_cols("v2")
    return {
        "train": s.train,
        "test": s.test,
        "feature_cols": cols,
        "target": "price",
        "task": "regression",
        "group_col": "group_id",
        "temporal_split": False,
        "max_overlap_share": 0.0,
    }


def get_cv() -> dict:
    tr = get_split()["train"]
    splits = cv_splits(tr)
    return {"splits": splits, "groups": tr["group_id"].to_numpy(), "times": None}


def get_raw_sample() -> pd.DataFrame:
    s = _split_cache()
    cols = model_input_cols("v2")
    return s.test[cols].head(100).reset_index(drop=True)


def bulk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized / batch feature creation from raw input columns."""
    if set(RAW_INPUT_COLS).issubset(df.columns):
        return build_feature_frame(df)[model_input_cols("v2")]
    return df[model_input_cols("v2")].copy()


def single_features(rec: dict) -> dict:
    """Serving-time feature extraction for one raw record."""
    if "listTitle" in rec and "location" in rec:
        f = extract_features(
            details=rec.get("details"),
            title=rec.get("listTitle"),
            size=rec["size"],
            bedrooms=rec.get("bedrooms"),
            bathrooms=rec.get("bathrooms"),
            district=rec["location"],
        )
        return {k: f[k] for k in model_input_cols("v2")}
    return {k: rec[k] for k in model_input_cols("v2")}


@lru_cache(maxsize=1)
def load_pipeline():
    artifact = joblib.load(ROOT / "house_price_model.joblib")
    return artifact["pipeline"]


@lru_cache(maxsize=1)
def get_frozen_test() -> tuple:
    s = _split_cache()
    cols = model_input_cols("v2")
    X = s.test[cols].reset_index(drop=True)
    y = np.log(s.test["price"].to_numpy(float))
    return X, y


@lru_cache(maxsize=1)
def expected_frozen_fingerprint() -> str:
    X, y = get_frozen_test()
    return frame_fingerprint(X.assign(__y__=y))


@lru_cache(maxsize=1)
def _baseline_model():
    s = _split_cache()
    cols = model_input_cols("v2")
    dummy = DummyRegressor(strategy="median")
    dummy.fit(s.train[cols], np.log(s.train["price"].to_numpy(float)))
    return dummy


def baseline_predict(X: pd.DataFrame) -> np.ndarray:
    return _baseline_model().predict(X)


def per_item_loss(y_true, y_pred) -> np.ndarray:
    # abs log error on log-scale targets
    return np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))


def regression_config() -> dict:
    return {"max_mean_loss": 0.20, "min_median_gain": 0.05}
