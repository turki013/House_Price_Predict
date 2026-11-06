"""Distribution drift: PSI (numeric + categorical) and KS reports."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _st

PSI_STABLE, PSI_MODERATE = 0.10, 0.25


def psi_status(value: float) -> str:
    return "stable" if value < PSI_STABLE else "moderate shift" if value < PSI_MODERATE else "major shift"


def psi_categorical(expected: Iterable, actual: Iterable, eps: float = 1e-6) -> float:
    e = pd.Series(list(expected)).value_counts(normalize=True)
    a = pd.Series(list(actual)).value_counts(normalize=True)
    cats = e.index.union(a.index)
    pe = np.clip(e.reindex(cats, fill_value=0.0).to_numpy(), eps, None)
    pa = np.clip(a.reindex(cats, fill_value=0.0).to_numpy(), eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi(expected: Iterable, actual: Iterable, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index of `actual` against the `expected` (reference) sample.
    Bin edges are reference quantiles. Rule of thumb: <0.10 stable, 0.10-0.25 moderate, >0.25 major.
    PSI is a screening signal, not a significance test."""
    e = pd.to_numeric(pd.Series(list(expected)), errors="coerce").dropna().to_numpy(float)
    a = pd.to_numeric(pd.Series(list(actual)), errors="coerce").dropna().to_numpy(float)
    if e.size == 0 or a.size == 0:
        raise ValueError("psi: empty sample")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # near-constant reference: fall back to category comparison
        return psi_categorical(e, a, eps)
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / e.size
    pa = np.histogram(a, edges)[0] / a.size
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, numeric_cols: Sequence[str],
               categorical_cols: Sequence[str] = (), bins: int = 10) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        v = psi(reference[c], current[c], bins=bins)
        rows.append((c, "numeric", v, psi_status(v)))
    for c in categorical_cols:
        v = psi_categorical(reference[c], current[c])
        rows.append((c, "categorical", v, psi_status(v)))
    return pd.DataFrame(rows, columns=["feature", "kind", "psi", "status"]).sort_values(
        "psi", ascending=False).reset_index(drop=True)


def ks_report(reference: pd.DataFrame, current: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Two-sample KS per column. With large samples p-values are tiny for trivial shifts; read the
    statistic (max CDF gap) alongside the p-value."""
    rows = []
    for c in cols:
        r = pd.to_numeric(reference[c], errors="coerce").dropna()
        n = pd.to_numeric(current[c], errors="coerce").dropna()
        res = _st.ks_2samp(r, n)
        rows.append((c, float(res.statistic), float(res.pvalue)))
    return pd.DataFrame(rows, columns=["feature", "ks_stat", "p_value"]).sort_values(
        "ks_stat", ascending=False).reset_index(drop=True)
