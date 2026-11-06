"""Data loading, cleaning rules, near-duplicate groups and the ONE shared split.

Order of operations (each step's row count is recorded in ``AuditLog``):

1. raw load
2. exact duplicates on ``DEDUP_KEY`` dropped (same listing scraped twice)
3. fixed domain bounds on price and size (stated rules, not learned from data)
4. near-duplicate groups (``group_id``): rows that share
   (district, size, bedrooms, bathrooms) OR the same normalised description
   text are put in one group. Re-listings with a changed price and identical
   units of one project end up in the same group, so they can never sit on both
   sides of a split.
5. ``split_frame``: GroupShuffleSplit (20% test) on ``group_id``
6. price-per-m2 validity rule (fixed bounds) applied to train and test alike;
   the excluded test rows are kept aside for a sensitivity check.

Only descriptive checks and fixed rules run before the split. Nothing here
learns from prices.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from dstk.contracts import validate_frame
from dstk.leakage import normalize_text
from features import CATEGORICAL_COL, build_feature_frame, clean_district

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Aqar_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

DEDUP_KEY = ["location", "size", "price", "bedrooms", "bathrooms"]
NEAR_DUP_KEY = ["district", "size", "bedrooms", "bathrooms"]
MIN_TEXT_LEN_FOR_GROUPING = 30

# Fixed domain rules (documented in reports/data_audit.md):
PRICE_BOUNDS = (150_000, 7_000_000)     # SAR, asking price of a whole villa
SIZE_BOUNDS = (100, 1_200)              # m2, plot size
PPM_BOUNDS = (1_000, 15_000)            # SAR per m2; outside = data-entry error
                                        # (e.g. 10,000,000 for 400 m2 in Tuwaiq,
                                        #  85,000 for 200 m2 in Al-Rimal)

RAW_SCHEMA = {
    "columns": {
        "listTitle": {"dtype": "str", "nullable": False},
        "location": {"dtype": "str", "nullable": False},
        "size": {"dtype": "number", "nullable": False, "min": 1},
        "price": {"dtype": "number", "nullable": False, "min": 1},
        "bedrooms": {"dtype": "number", "nullable": False, "min": 1, "max": 7},
        "bathrooms": {"dtype": "number", "nullable": False, "min": 1, "max": 5},
        "details": {"dtype": "str", "nullable": True, "max_null_share": 0.05},
        "time": {"dtype": "str", "nullable": False, "required": False},
    },
    "min_rows": 5_000,
    "strict_columns": True,
}


@dataclass
class AuditLog:
    steps: list = field(default_factory=list)

    def add(self, step: str, before: int, after: int, rule: str) -> None:
        self.steps.append({"step": step, "rows_before": int(before), "rows_after": int(after),
                           "removed": int(before - after), "rule": rule})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _union_find_groups(*keys: np.ndarray) -> np.ndarray:
    """Connected components over rows that share ANY of the given key arrays.
    A key value of -1 means "no key" (the row is not linked through it)."""
    n = len(keys[0])
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        first: dict = {}
        for i, k in enumerate(key):
            if k < 0:
                continue
            j = first.setdefault(k, i)
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    return pd.factorize(roots)[0].astype(np.int64)


def near_duplicate_groups(df: pd.DataFrame) -> np.ndarray:
    """Group id per row (see module docstring, step 4)."""
    k1 = pd.factorize(pd.MultiIndex.from_frame(df[NEAR_DUP_KEY]))[0]
    text = df["details"].map(normalize_text)
    k2 = np.where(text.str.len() > MIN_TEXT_LEN_FOR_GROUPING, pd.factorize(text)[0], -1)
    return _union_find_groups(np.asarray(k1), np.asarray(k2))


def load_clean(path: Path = DATA_PATH, audit: AuditLog | None = None) -> pd.DataFrame:
    """Raw CSV -> frame with raw columns, ``district``, ``group_id``, ``ppm_valid``
    and all model features. Steps 1-4 plus the ppm flag (not yet applied)."""
    audit = audit if audit is not None else AuditLog()
    raw = pd.read_csv(path)
    validate_frame(raw, RAW_SCHEMA).raise_if_failed()
    audit.add("raw load", len(raw), len(raw), f"file sha256 {file_sha256(path)[:12]}")

    n = len(raw)
    raw = raw.drop_duplicates(subset=DEDUP_KEY).copy()
    audit.add("exact duplicates", n, len(raw), "key = " + ", ".join(DEDUP_KEY))

    n = len(raw)
    keep = raw["price"].between(*PRICE_BOUNDS) & raw["size"].between(*SIZE_BOUNDS)
    raw = raw[keep].copy()
    audit.add("price and size bounds", n, len(raw),
              f"price {PRICE_BOUNDS[0]:,}-{PRICE_BOUNDS[1]:,} SAR, size {SIZE_BOUNDS[0]}-{SIZE_BOUNDS[1]} m2")

    raw["district"] = raw["location"].map(clean_district)
    raw["group_id"] = near_duplicate_groups(raw)
    raw["ppm"] = raw["price"] / raw["size"]
    raw["ppm_valid"] = raw["ppm"].between(*PPM_BOUNDS)
    raw["text_len"] = raw["details"].fillna("").str.len()

    feats = build_feature_frame(raw)
    out = pd.concat([raw[["price", "district", "group_id", "ppm", "ppm_valid", "text_len", "details",
                          "listTitle", "location"]], feats], axis=1)
    assert (out["district"] == out[CATEGORICAL_COL]).all()
    return out


@dataclass
class Split:
    train: pd.DataFrame          # ppm-valid training rows
    test: pd.DataFrame           # ppm-valid test rows (the held-out set)
    test_ppm_invalid: pd.DataFrame  # test-group rows removed by the ppm rule (sensitivity only)
    audit: AuditLog


def split_frame(path: Path = DATA_PATH) -> Split:
    """The single split used by every script, test and notebook."""
    audit = AuditLog()
    df = load_clean(path, audit)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_i, te_i = next(gss.split(df, groups=df["group_id"]))
    train, test = df.iloc[tr_i], df.iloc[te_i]

    n_tr, n_te = len(train), len(test)
    train_v, test_v = train[train["ppm_valid"]].copy(), test[test["ppm_valid"]].copy()
    audit.add("price per m2 rule (train)", n_tr, len(train_v),
              f"{PPM_BOUNDS[0]:,}-{PPM_BOUNDS[1]:,} SAR/m2")
    audit.add("price per m2 rule (test)", n_te, len(test_v),
              f"{PPM_BOUNDS[0]:,}-{PPM_BOUNDS[1]:,} SAR/m2; removed rows kept for a sensitivity check")
    return Split(train=train_v, test=test_v, test_ppm_invalid=test[~test["ppm_valid"]].copy(), audit=audit)


def cv_splits(train: pd.DataFrame, n_folds: int = N_FOLDS):
    """Grouped CV folds on the training set (positional indices)."""
    return list(GroupKFold(n_splits=n_folds).split(train, groups=train["group_id"]))
