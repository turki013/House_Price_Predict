"""Leakage and split-integrity tests.

Part 1 runs against `hooks` (bundled example + your project hooks when present).
Part 2 are self-tests proving each detector fires on planted leakage and stays quiet on clean data.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline

from dstk import leakage as L


# ----------------------------------------------------------------------------------- part 1: hooks
def test_no_exact_row_overlap_between_train_and_test(hooks, require):
    require("get_split")
    s = hooks.get_split()
    cols = list(s["feature_cols"]) + [s["target"]]
    L.assert_no_row_overlap(s["train"], s["test"], cols, max_share=s.get("max_overlap_share", 0.0))


def test_no_group_overlap_between_train_and_test(hooks, require):
    require("get_split")
    s = hooks.get_split()
    if not s.get("group_col"):
        pytest.skip("no group_col declared")
    L.assert_no_group_overlap(s["train"][s["group_col"]], s["test"][s["group_col"]])


def test_temporal_split_is_ordered(hooks, require):
    require("get_split")
    s = hooks.get_split()
    if not (s.get("temporal_split") and s.get("time_col")):
        pytest.skip("split is not declared temporal")
    L.assert_temporal_order(s["train"][s["time_col"]], s["test"][s["time_col"]])


def test_cv_folds_have_no_leakage(hooks, require):
    require("get_cv")
    cv = hooks.get_cv()
    checked = False
    if cv.get("groups") is not None:
        L.assert_folds_respect_groups(cv["splits"], cv["groups"])
        checked = True
    if cv.get("times") is not None:
        L.assert_folds_temporal(cv["splits"], cv["times"])
        checked = True
    if not checked:
        pytest.skip("get_cv declares neither groups nor times")


def test_label_shuffle_is_near_chance(hooks, require):
    require("get_split", "load_pipeline")
    s = hooks.get_split()
    X, y = s["train"][s["feature_cols"]], s["train"][s["target"]]
    scoring, chance = ("roc_auc", 0.5) if s["task"] == "classification" else ("r2", 0.0)
    groups = s["train"][s["group_col"]] if s.get("group_col") else None
    cv = GroupKFold(5) if groups is not None else 5
    res = L.label_shuffle_check(hooks.load_pipeline(), X, y, scoring, cv=cv, groups=groups, n_repeats=2)
    L.assert_label_shuffle_near_chance(res, chance, tolerance=0.15)


def test_train_test_distribution_is_similar(hooks, require):
    require("get_split")
    s = hooks.get_split()
    res = L.adversarial_validation(s["train"][s["feature_cols"]], s["test"][s["feature_cols"]])
    assert res["auc"] < 0.75, f"train and test are easy to tell apart (AUC {res['auc']:.2f}):\n{res['top_shift_features']}"


# ----------------------------------------------------------------------------------- part 2: self-tests
def _entities(n_entities=60, rows_each=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_entities), rows_each)
    base = rng.normal(size=n_entities)[ids]
    return pd.DataFrame({"entity": ids, "x": base + rng.normal(scale=0.1, size=ids.size),
                         "y": base + rng.normal(scale=0.1, size=ids.size)})


def test_detects_planted_duplicates_across_split():
    df = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) % 7, "y": np.arange(100) % 2})
    train, test = df.iloc[:80], pd.concat([df.iloc[80:], df.iloc[:10]])
    with pytest.raises(AssertionError, match="also appear in train"):
        L.assert_no_row_overlap(train, test, ["a", "b", "y"])
    L.assert_no_row_overlap(df.iloc[:80], df.iloc[80:], ["a", "b", "y"])


def test_random_split_leaks_entities_but_group_split_does_not():
    df = _entities()
    tr, te = train_test_split(df, test_size=0.3, random_state=0)
    with pytest.raises(AssertionError, match="groups appear in both"):
        L.assert_no_group_overlap(tr["entity"], te["entity"])
    gi, gj = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(df, groups=df["entity"]))
    L.assert_no_group_overlap(df.iloc[gi]["entity"], df.iloc[gj]["entity"])


def test_cv_group_and_temporal_checks_catch_wrong_splitters():
    df = _entities()
    with pytest.raises(AssertionError):
        L.assert_folds_respect_groups(KFold(5, shuffle=True, random_state=0).split(df), df["entity"])
    L.assert_folds_respect_groups(GroupKFold(5).split(df, groups=df["entity"]), df["entity"])
    t = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D"))
    L.assert_folds_temporal(TimeSeriesSplit(5).split(t), t)
    with pytest.raises(AssertionError, match="not before"):
        L.assert_folds_temporal(KFold(5, shuffle=True, random_state=0).split(t), t)


def test_temporal_order_detects_random_split_of_time_series():
    t = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    L.assert_temporal_order(t.iloc[:80], t.iloc[80:])
    tr, te = train_test_split(t, test_size=0.2, random_state=0)
    with pytest.raises(AssertionError):
        L.assert_temporal_order(tr, te)


def _noise(n=120, p=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.integers(0, 2, size=n)


def test_label_shuffle_flags_selection_done_outside_cv():
    X, y = _noise()

    def leaky(yy):  # selects features with ALL labels, then cross-validates
        Xs = SelectKBest(f_classif, k=20).fit_transform(X, yy)
        return cross_val_score(LogisticRegression(max_iter=500), Xs, yy, scoring="roc_auc", cv=5).mean()

    res = L.label_shuffle_procedure(leaky, y, n_repeats=3)
    assert res["shuffled_mean"] > 0.65, res  # the leak manufactures signal out of pure noise
    with pytest.raises(AssertionError, match="leaks label information"):
        L.assert_label_shuffle_near_chance(res, chance=0.5, tolerance=0.1)


def test_label_shuffle_passes_when_selection_is_inside_the_pipeline():
    X, y = _noise()
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=500))
    res = L.label_shuffle_check(pipe, X, y, "roc_auc", cv=5, n_repeats=3)
    L.assert_label_shuffle_near_chance(res, chance=0.5, tolerance=0.12)


def test_single_feature_auc_scan_flags_target_proxy():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    X = pd.DataFrame({"noise": rng.normal(size=500), "proxy": y + rng.normal(scale=0.01, size=500)})
    out = L.single_feature_auc_scan(X, y)
    assert out.iloc[0]["feature"] == "proxy" and bool(out.iloc[0]["flag"])
    assert not bool(out[out["feature"] == "noise"].iloc[0]["flag"])


def test_adversarial_validation_detects_shift_and_accepts_same_distribution():
    rng = np.random.default_rng(2)
    a = pd.DataFrame({"f": rng.normal(0, 1, 600), "g": rng.normal(0, 1, 600)})
    same = pd.DataFrame({"f": rng.normal(0, 1, 600), "g": rng.normal(0, 1, 600)})
    shifted = pd.DataFrame({"f": rng.normal(2, 1, 600), "g": rng.normal(0, 1, 600)})
    assert L.adversarial_validation(a, same)["auc"] < 0.6
    res = L.adversarial_validation(a, shifted)
    assert res["auc"] > 0.8 and res["top_shift_features"].iloc[0]["feature"] == "f"


def test_text_group_ids_group_near_duplicates_and_keep_empty_rows_separate():
    texts = pd.Series(["Villa  with POOL, 3 rooms!", "villa with pool 3 rooms", "فيلا ٣ غرف", "فيلا 3 غرف", "", "", None])
    g = L.text_group_ids(texts)
    assert g[0] == g[1] and g[2] == g[3] and g[0] != g[2]
    assert len(set(g[4:])) == 3  # empty or missing texts must not be merged


def test_frame_fingerprint_changes_when_data_changes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert L.frame_fingerprint(df) == L.frame_fingerprint(df.copy())
    edited = df.copy()
    edited.loc[1, "a"] = 99
    assert L.frame_fingerprint(df) != L.frame_fingerprint(edited)
