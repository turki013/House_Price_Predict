"""A/B-test helpers: two-proportion test, sample-ratio-mismatch check, CUPED, multiple-testing correction."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats as _st


def two_proportion_test(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> Dict[str, float]:
    """Two-sided z-test (pooled) for p2 - p1 plus an unpooled Wald CI of the difference."""
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 else 0.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = _st.norm.ppf(1 - alpha / 2)
    d = p2 - p1
    return {"p1": p1, "p2": p2, "diff": d, "ci_lo": d - zc * se, "ci_hi": d + zc * se,
            "z": z, "p_value": float(2 * (1 - _st.norm.cdf(abs(z))))}


def srm_check(observed: Sequence[int], expected_ratios: Sequence[float], alpha: float = 0.001) -> Dict[str, float]:
    """Sample-ratio-mismatch: chi-square of observed group sizes vs the designed split. If it fires,
    the assignment or logging is broken and results must not be interpreted."""
    obs = np.asarray(observed, float)
    exp = np.asarray(expected_ratios, float) / np.sum(expected_ratios) * obs.sum()
    chi2, p = _st.chisquare(obs, exp)
    return {"chi2": float(chi2), "p_value": float(p), "srm_detected": bool(p < alpha)}


def cuped_adjust(y, x_pre) -> Tuple[np.ndarray, float, float]:
    """CUPED: y_adj = y - theta * (x_pre - mean(x_pre)), theta = cov(y, x_pre) / var(x_pre).
    x_pre must be measured BEFORE assignment. Returns (y_adj, theta, variance_reduction_share)."""
    y, x = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta), float(1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1))


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm step-down: returns (adjusted p-values, reject flags). Controls family-wise error."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """BH step-up: controls the false discovery rate. Returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for i, idx in enumerate(order):
        rank = m - i
        running = min(running, p[idx] * m / rank)
        adj[idx] = running
    return adj, adj < alpha
