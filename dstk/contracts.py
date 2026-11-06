"""Data contracts: declare what valid data looks like, then fail loudly when it is not.

Schema format (plain dict, easy to keep in a .py or .json file)::

    SCHEMA = {
        "columns": {
            "price":    {"dtype": "number", "nullable": False, "min": 150_000, "max": 7_000_000},
            "size":     {"dtype": "number", "nullable": False, "min": 100, "max": 1_200},
            "district": {"dtype": "str", "nullable": False, "allowed": ["A", "B"]},
            "listing_id": {"dtype": "str", "regex": r"^L\\d+$"},
        },
        "unique_key": ["listing_id"],        # optional
        "min_rows": 1_000,                   # optional
        "max_duplicate_share": 0.0,          # optional, exact duplicates
        "duplicate_subset": None,            # optional: columns used for the duplicate check
        "strict_columns": False,             # True: unexpected columns are errors
    }

dtype names: int, float, number, bool, str, datetime, category.
Per-column keys: dtype, required (default True), nullable (default True), max_null_share,
min, max, allowed, regex, unique.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when data violates its contract. The message lists every violation."""


@dataclass
class ContractReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "ContractReport":
        if self.errors:
            raise DataContractError("data contract violated:\n- " + "\n- ".join(self.errors))
        return self


def _is_str(s: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        return bool(s.dropna().map(lambda v: isinstance(v, str)).all())
    return False


_DTYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "number": pd.api.types.is_numeric_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "str": _is_str,
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
}


def _examples(index: pd.Index, mask: pd.Series, n: int = 5) -> str:
    return str(list(index[mask.to_numpy()][:n]))


def validate_frame(df: pd.DataFrame, schema: Dict[str, Any], serving: bool = False) -> ContractReport:
    """Validate a DataFrame against a schema. serving=True skips dataset-level rules
    (min_rows, unique_key, duplicates) so single requests can be validated at inference time."""
    rep = ContractReport()
    cols_spec: Dict[str, Dict[str, Any]] = schema.get("columns", {})

    for name, spec in cols_spec.items():
        if name not in df.columns:
            if spec.get("required", True):
                rep.errors.append(f"missing required column '{name}'")
            continue
        s = df[name]
        dtype = spec.get("dtype")
        if dtype:
            check = _DTYPE_CHECKS.get(dtype)
            if check is None:
                rep.errors.append(f"column '{name}': unknown dtype '{dtype}' in schema")
            elif not check(s):
                rep.errors.append(f"column '{name}': expected {dtype}, found {s.dtype}")
                continue
        n_null = int(s.isna().sum())
        share = n_null / max(len(s), 1)
        if spec.get("nullable", True) is False and n_null:
            rep.errors.append(f"column '{name}': {n_null} null values but nullable=False "
                              f"(rows {_examples(df.index, s.isna())})")
        if "max_null_share" in spec and share > spec["max_null_share"]:
            rep.errors.append(f"column '{name}': null share {share:.2%} exceeds {spec['max_null_share']:.2%}")
        nn = s.dropna()
        if "min" in spec and len(nn):
            bad = s < spec["min"]
            if bad.any():
                rep.errors.append(f"column '{name}': {int(bad.sum())} values below min {spec['min']} "
                                  f"(rows {_examples(df.index, bad)})")
        if "max" in spec and len(nn):
            bad = s > spec["max"]
            if bad.any():
                rep.errors.append(f"column '{name}': {int(bad.sum())} values above max {spec['max']} "
                                  f"(rows {_examples(df.index, bad)})")
        if "allowed" in spec:
            bad = ~s.isin(spec["allowed"]) & s.notna()
            if bad.any():
                vals = list(pd.unique(s[bad]))[:5]
                rep.errors.append(f"column '{name}': {int(bad.sum())} values outside the allowed set, e.g. {vals}")
        if "regex" in spec:
            bad = s.notna() & ~s.astype("string").str.match(spec["regex"]).fillna(False).astype(bool)
            if bad.any():
                rep.errors.append(f"column '{name}': {int(bad.sum())} values do not match /{spec['regex']}/ "
                                  f"(rows {_examples(df.index, bad)})")
        if spec.get("unique") and not serving and s.duplicated().any():
            rep.errors.append(f"column '{name}': values must be unique ({int(s.duplicated().sum())} repeats)")

    extra = [c for c in df.columns if c not in cols_spec]
    if extra:
        msg = f"unexpected columns not in schema: {extra}"
        (rep.errors if schema.get("strict_columns") else rep.warnings).append(msg)

    if not serving:
        if schema.get("min_rows") and len(df) < schema["min_rows"]:
            rep.errors.append(f"only {len(df)} rows, schema requires at least {schema['min_rows']}")
        key = schema.get("unique_key")
        if key and set(key) <= set(df.columns):
            d = df.duplicated(subset=list(key), keep=False)
            if d.any():
                rep.errors.append(f"unique_key {list(key)} violated by {int(d.sum())} rows")
        if "max_duplicate_share" in schema:
            subset = schema.get("duplicate_subset") or None
            share = float(df.duplicated(subset=subset, keep="first").mean()) if len(df) else 0.0
            if share > schema["max_duplicate_share"]:
                rep.errors.append(f"duplicate share {share:.2%} exceeds {schema['max_duplicate_share']:.2%} "
                                  f"(subset={subset or 'all columns'})")
    return rep


def validate_records(records: Sequence[Dict[str, Any]], schema: Dict[str, Any]) -> ContractReport:
    """Validate serving-time records (list of dicts) against the same schema, without dataset-level rules."""
    return validate_frame(pd.DataFrame(list(records)), schema, serving=True)


def infer_schema(df: pd.DataFrame, max_categories: int = 30, margin: float = 0.0) -> Dict[str, Any]:
    """Draft a schema from a reference frame. REVIEW IT: observed ranges are a starting point, not truth."""
    cols: Dict[str, Dict[str, Any]] = {}
    for c in df.columns:
        s = df[c]
        spec: Dict[str, Any] = {"nullable": bool(s.isna().any())}
        if pd.api.types.is_bool_dtype(s):
            spec["dtype"] = "bool"
        elif pd.api.types.is_numeric_dtype(s):
            spec["dtype"] = "number"
            if s.notna().any():
                lo, hi = float(s.min()), float(s.max())
                pad = margin * (hi - lo)
                spec["min"], spec["max"] = lo - pad, hi + pad
        elif pd.api.types.is_datetime64_any_dtype(s):
            spec["dtype"] = "datetime"
        else:
            spec["dtype"] = "str"
            if s.nunique(dropna=True) <= max_categories:
                spec["allowed"] = sorted(map(str, s.dropna().unique()))
        cols[c] = spec
    return {"columns": cols, "min_rows": 1, "strict_columns": False}
