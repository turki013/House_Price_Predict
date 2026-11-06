"""Train/serve parity tests: bulk vs single-record features, batch vs row-wise predictions,
save/load round trip, column-order invariance, determinism."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from dstk import parity as P


# ----------------------------------------------------------------------------------- hooks
def test_bulk_and_single_feature_paths_match(hooks, require):
    require("get_raw_sample", "bulk_features", "single_features")
    raw = hooks.get_raw_sample()
    bulk = hooks.bulk_features(raw)
    single = pd.DataFrame([hooks.single_features(r) for r in raw.to_dict("records")])
    P.assert_feature_parity(bulk, single)


def test_batch_predictions_equal_single_row_predictions(hooks, require):
    require("get_raw_sample", "load_pipeline")
    P.assert_batch_equals_rowwise(hooks.load_pipeline().predict, hooks.get_raw_sample(), n=40)


def test_predictions_survive_save_and_load(hooks, require, tmp_path):
    require("get_raw_sample", "load_pipeline")
    P.assert_roundtrip_equal(hooks.load_pipeline(), hooks.get_raw_sample(), tmp_path / "pipe.joblib")


def test_predictions_do_not_depend_on_column_order(hooks, require):
    require("get_raw_sample", "load_pipeline")
    P.assert_column_order_invariant(hooks.load_pipeline().predict, hooks.get_raw_sample())


def test_predictions_are_deterministic(hooks, require):
    require("get_raw_sample", "load_pipeline")
    P.assert_deterministic(hooks.load_pipeline().predict, hooks.get_raw_sample())


def test_predictions_are_finite(hooks, require):
    require("get_raw_sample", "load_pipeline")
    preds = np.asarray(hooks.load_pipeline().predict(hooks.get_raw_sample()), dtype=float)
    assert np.isfinite(preds).all()


# ----------------------------------------------------------------------------------- self-tests
def test_feature_parity_detects_integer_division_bug():
    raw = pd.DataFrame({"size": [100.0, 250.0, 333.0], "rooms": [3, 4, 5]})
    bulk = pd.DataFrame({"ratio": raw["size"] / raw["rooms"]})
    buggy = pd.DataFrame({"ratio": [float(int(s) // int(r)) for s, r in zip(raw["size"], raw["rooms"])]})
    with pytest.raises(AssertionError, match="'ratio'"):
        P.assert_feature_parity(bulk, buggy)
    P.assert_feature_parity(bulk, bulk.copy())


def test_feature_parity_detects_column_mismatch_and_categorical_difference():
    a = pd.DataFrame({"x": [1, 2], "c": ["a", "b"]})
    with pytest.raises(AssertionError, match="only in training path"):
        P.assert_feature_parity(a, a[["x"]])
    b = a.copy()
    b.loc[1, "c"] = "B"
    with pytest.raises(AssertionError, match="'c'"):
        P.assert_feature_parity(a, b)


class _PositionalModel(BaseEstimator, RegressorMixin):
    """Broken on purpose: reads the first column by position instead of by name."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X.iloc[:, 0], dtype=float) * 2.0


def test_column_order_check_catches_positional_models():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [9.0, 8.0, 7.0]})
    with pytest.raises(AssertionError, match="column order"):
        P.assert_column_order_invariant(_PositionalModel().predict, df)


class _Flaky(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.random.default_rng().normal(size=len(X))


def test_determinism_check_catches_random_predictions():
    with pytest.raises(AssertionError, match="different results"):
        P.assert_deterministic(_Flaky().predict, pd.DataFrame({"a": [1, 2, 3]}))
