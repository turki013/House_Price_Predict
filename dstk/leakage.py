"""Leakage and split-integrity checks.

What each check catches (and does not):
* assert_no_row_overlap / duplicate_report   exact duplicates across train and test
* text_group_ids                              near-duplicates (same normalized text) -> use as groups
* assert_no_group_overlap / assert_folds_*    entity, group and time leakage across splits or CV folds
* label_shuffle_procedure / _check            PROCEDURAL leakage (feature selection or tuning done outside
                                              CV, label-dependent preprocessing). The WHOLE procedure must be
                                              re-run on permuted labels. It does NOT catch duplicates or
                                              target-derived features: use the checks above and
                                              single_feature_auc_scan for those.
* adversarial_validation                      train/test distribution shift
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")


# ----------------------------------------------------------------------------- text / duplicates
def normalize_text(s) -> str:
    """Lowercase, Arabic digits -> ASCII, strip tashkeel/tatweel and punctuation, collapse spaces."""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    s = str(s).translate(_AR_DIGITS).lower()
    s = _TASHKEEL.sub("", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_group_ids(texts: pd.Series) -> np.ndarray:
    """Integer group id per row; rows with the same normalized text share an id. Empty texts get
    unique ids (they must not be grouped together). Use with GroupShuffleSplit / GroupKFold."""
    norm = pd.Series(texts).map(normalize_text)
    digest = norm.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest() if t else "")
    codes, _ = pd.factorize(digest)
    codes = codes.astype(np.int64)
    empty = (digest == "").to_numpy()
    if empty.any():
        start = codes.max() + 1 if codes.size else 0
        codes[empty] = np.arange(start, start + empty.sum())
    return codes


def duplicate_report(df: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> Dict[str, float]:
    dup = df.duplicated(subset=list(subset) if subset else None, keep="first")
    n = len(df)
    return {"n_rows": n, "n_duplicates": int(dup.sum()), "share": float(dup.mean()) if n else 0.0}


def row_overlap(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> int:
    """Number of test rows whose values on `cols` also appear in train."""
    h_tr = pd.util.hash_pandas_object(train[list(cols)], index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(test[list(cols)], index=False).to_numpy()
    return int(np.isin(h_te, h_tr).sum())


def assert_no_row_overlap(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str],
                          max_share: float = 0.0) -> None:
    n = row_overlap(train, test, cols)
    share = n / max(len(test), 1)
    if share > max_share:
        raise AssertionError(f"{n} of {len(test)} test rows ({share:.1%}) also appear in train "
                             f"on columns {list(cols)}; allowed share is {max_share:.1%}")


# ----------------------------------------------------------------------------- groups / time
def group_overlap(train_groups: Iterable, test_groups: Iterable) -> set:
    return set(pd.Series(list(train_groups)).dropna()) & set(pd.Series(list(test_groups)).dropna())


def assert_no_group_overlap(train_groups: Iterable, test_groups: Iterable) -> None:
    inter = group_overlap(train_groups, test_groups)
    if inter:
        ex = list(inter)[:5]
        raise AssertionError(f"{len(inter)} groups appear in both train and test, e.g. {ex}")


def assert_temporal_order(train_times: Iterable, test_times: Iterable, allow_equal: bool = False) -> None:
    tr, te = pd.Series(list(train_times)), pd.Series(list(test_times))
    last_train, first_test = tr.max(), te.min()
    bad = last_train > first_test if allow_equal else last_train >= first_test
    if bad:
        raise AssertionError(f"latest train time {last_train} is not before earliest test time {first_test}")


def assert_folds_respect_groups(splits: Iterable[Tuple[np.ndarray, np.ndarray]], groups) -> None:
    g = np.asarray(groups)
    for i, (tr, va) in enumerate(splits):
        inter = set(g[tr]) & set(g[va])
        if inter:
            raise AssertionError(f"fold {i}: {len(inter)} groups in both train and validation, "
                                 f"e.g. {list(inter)[:5]}")


def assert_folds_temporal(splits: Iterable[Tuple[np.ndarray, np.ndarray]], times,
                          allow_equal: bool = False) -> None:
    t = pd.Series(np.asarray(times))
    for i, (tr, va) in enumerate(splits):
        assert_temporal_order(t.iloc[tr], t.iloc[va], allow_equal=allow_equal)


# ----------------------------------------------------------------------------- label shuffle
def label_shuffle_procedure(score_fn, y, n_repeats: int = 3, seed: int = 0) -> dict:
    """Run a COMPLETE modeling procedure on real and on permuted labels.

    `score_fn(y_vector) -> float` must contain every step that touches labels (feature selection,
    encoding, tuning, CV). If any such step runs outside the cross-validation loop, the permuted-label
    score will sit above chance, which exposes the leak."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    real = float(score_fn(y))
    shuffled = [float(score_fn(rng.permutation(y))) for _ in range(n_repeats)]
    return {"real": real, "shuffled_mean": float(np.mean(shuffled)),
            "shuffled_std": float(np.std(shuffled)), "shuffled": shuffled}


def label_shuffle_check(estimator, X, y, scoring: str, cv=5, groups=None, n_repeats: int = 3,
                        seed: int = 0) -> dict:
    """Cross-validated score with real labels vs permuted labels for a pipeline-contained estimator.
    With a leak-free procedure the permuted score sits near chance (AUC 0.5, R2 about 0)."""
    def score(yy):
        return cross_val_score(clone(estimator), X, yy, scoring=scoring, cv=cv, groups=groups).mean()
    return label_shuffle_procedure(score, y, n_repeats=n_repeats, seed=seed)


def assert_label_shuffle_near_chance(result: dict, chance: float, tolerance: float = 0.1) -> None:
    """One-sided: leakage makes the permuted-label score ABOVE chance. A score below chance (typical
    for R2 of an over-flexible model on pure noise) is not a leak and passes."""
    excess = result["shuffled_mean"] - chance
    if excess > tolerance:
        raise AssertionError(f"score with shuffled labels is {result['shuffled_mean']:.3f}, which is "
                             f"{excess:.3f} above chance {chance:.3f} (tolerance {tolerance}); "
                             f"the procedure leaks label information")


def single_feature_auc_scan(X: pd.DataFrame, y, top: int = 10, flag_above: float = 0.95) -> pd.DataFrame:
    """Binary targets only: per-feature AUC (direction-free). Very high values suggest a
    target-derived or post-outcome column; investigate before modeling."""
    y = np.asarray(y)
    if len(np.unique(y)) != 2:
        raise ValueError("single_feature_auc_scan supports binary targets only")
    rows: List[Tuple[str, float]] = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.nunique(dropna=True) < 2:
            continue
        s = s.fillna(s.median())
        auc = roc_auc_score(y, s)
        rows.append((c, float(max(auc, 1 - auc))))
    out = pd.DataFrame(rows, columns=["feature", "auc"]).sort_values("auc", ascending=False)
    out = out.head(top).reset_index(drop=True)
    out["flag"] = out["auc"] >= flag_above
    return out


# ----------------------------------------------------------------------------- adversarial validation
def _encode_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype("int64")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c], _ = pd.factorize(out[c])
    return out


def adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame, n_splits: int = 5,
                           seed: int = 0, max_rows: int = 20_000) -> dict:
    """Classifier that tries to tell train rows from test rows. AUC near 0.5: similar
    distributions. AUC above about 0.7: material shift; inspect `top_shift_features`."""
    xa = pd.concat([X_train, X_test], ignore_index=True)
    ya = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    if len(xa) > max_rows:
        keep = np.random.default_rng(seed).choice(len(xa), max_rows, replace=False)
        xa, ya = xa.iloc[keep].reset_index(drop=True), ya[keep]
    xe = _encode_numeric(xa)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=seed),
                          xe, ya, cv=cv, method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(ya, p)),
            "top_shift_features": single_feature_auc_scan(xe, ya, top=5, flag_above=0.7)}


# ----------------------------------------------------------------------------- frozen sets
def frame_fingerprint(df: pd.DataFrame) -> str:
    """Stable content hash of a DataFrame (column names + row values), for frozen test sets."""
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    head = "|".join(map(str, df.columns)).encode("utf-8")
    return hashlib.sha256(head + h.tobytes()).hexdigest()
