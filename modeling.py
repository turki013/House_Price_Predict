"""Shared modelling helpers: leak-free pipeline factory, fixed split, metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from features import CATEGORICAL_COL

RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_pipeline(regressor, scale: bool = False) -> Pipeline:
    """Neighbourhood target-encoding happens INSIDE the pipeline (out-of-fold on fit)."""
    encoder = TargetEncoder(
        smooth="auto", cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    prep = ColumnTransformer([("nh", encoder, [CATEGORICAL_COL])], remainder="passthrough")
    steps = [("prep", prep)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", regressor))
    return Pipeline(steps)


def split_index(frame: pd.DataFrame):
    """Identical train/test rows for every script that uses the same frame."""
    return train_test_split(frame.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def price_metrics(price_true, price_pred, villa_mask=None) -> dict:
    price_true, price_pred = np.asarray(price_true, float), np.asarray(price_pred, float)
    ape = np.abs(price_pred - price_true) / price_true
    out = {
        "MAE_SAR": round(float(np.mean(np.abs(price_pred - price_true))), 0),
        "MAPE_pct": round(float(ape.mean() * 100), 2),
        "MdAPE_pct": round(float(np.median(ape) * 100), 2),
        "R2_logprice": round(float(r2_score(np.log1p(price_true), np.log1p(price_pred))), 4),
    }
    if villa_mask is not None:
        m = np.asarray(villa_mask, bool)
        out["MAPE_villa_only_pct"] = round(float(ape[m].mean() * 100), 2)
        out["n_villa"] = int(m.sum())
    return out
