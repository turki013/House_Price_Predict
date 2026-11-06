"""Train/serve parity checks: the same raw input must yield the same features and predictions
whether it goes through the bulk training path, a single-record serving path, a saved-and-reloaded
artifact, or arrives with columns in a different order."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd


def assert_feature_parity(bulk: pd.DataFrame, single: pd.DataFrame, rtol: float = 1e-9,
                          atol: float = 1e-9) -> None:
    """Compare features built by the vectorized training path (`bulk`) with those built row by row by
    the serving path (`single`). Same columns, same values (numeric within tolerance)."""
    bulk, single = bulk.reset_index(drop=True), single.reset_index(drop=True)
    missing, extra = set(bulk.columns) - set(single.columns), set(single.columns) - set(bulk.columns)
    if missing or extra:
        raise AssertionError(f"feature columns differ: only in training path {sorted(missing)}, "
                             f"only in serving path {sorted(extra)}")
    if len(bulk) != len(single):
        raise AssertionError(f"row count differs: {len(bulk)} vs {len(single)}")
    problems = []
    for c in bulk.columns:
        a, b = bulk[c], single[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            ok = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
            if not ok.all():
                i = int(np.flatnonzero(~ok)[0])
                problems.append(f"'{c}': {int((~ok).sum())} rows differ, first at row {i}: "
                                f"training={av[i]!r} serving={bv[i]!r}")
        else:
            ok = (a.astype(object).where(a.notna(), None) == b.astype(object).where(b.notna(), None))
            if not ok.all():
                i = int(np.flatnonzero(~ok.to_numpy())[0])
                problems.append(f"'{c}': {int((~ok).sum())} rows differ, first at row {i}: "
                                f"training={a.iloc[i]!r} serving={b.iloc[i]!r}")
    if problems:
        raise AssertionError("training/serving feature mismatch:\n- " + "\n- ".join(problems))


def assert_batch_equals_rowwise(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                                n: int = 50, rtol: float = 1e-7, atol: float = 1e-9) -> None:
    sample = df.head(n)
    batch = np.asarray(predict_fn(sample), dtype=float).ravel()
    rows = np.array([np.asarray(predict_fn(sample.iloc[[i]]), dtype=float).ravel()[0]
                     for i in range(len(sample))])
    if not np.allclose(batch, rows, rtol=rtol, atol=atol, equal_nan=True):
        i = int(np.flatnonzero(~np.isclose(batch, rows, rtol=rtol, atol=atol, equal_nan=True))[0])
        raise AssertionError(f"batch and single-row predictions differ (row {i}: {batch[i]!r} vs {rows[i]!r})")


def assert_roundtrip_equal(estimator, df: pd.DataFrame, path: str | Path, rtol: float = 1e-9) -> None:
    """Predictions must be identical after saving and reloading the artifact."""
    before = np.asarray(estimator.predict(df), dtype=float)
    joblib.dump(estimator, path)
    after = np.asarray(joblib.load(path).predict(df), dtype=float)
    if not np.allclose(before, after, rtol=rtol, equal_nan=True):
        raise AssertionError("predictions changed after joblib save/load")


def assert_column_order_invariant(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                                  seed: int = 0) -> None:
    """Callers may send columns in any order; predictions must not depend on it."""
    cols = list(df.columns)
    shuffled = list(np.random.default_rng(seed).permutation(cols))
    if shuffled == cols:
        shuffled = cols[::-1]
    a = np.asarray(predict_fn(df), dtype=float)
    b = np.asarray(predict_fn(df[shuffled]), dtype=float)
    if not np.allclose(a, b, equal_nan=True):
        raise AssertionError("predictions depend on input column order")


def assert_deterministic(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame) -> None:
    a, b = np.asarray(predict_fn(df), dtype=float), np.asarray(predict_fn(df), dtype=float)
    if not np.array_equal(a, b, equal_nan=True):
        raise AssertionError("two identical predict calls returned different results")
