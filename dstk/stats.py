"""Uncertainty and paired-comparison helpers (numpy + scipy only)."""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy import stats as _st


def _boot_stats(x: np.ndarray, stat: Callable, n_boot: int, rng: np.random.Generator,
                max_cells: int = 5_000_000) -> np.ndarray:
    """Resample x with replacement n_boot times and apply stat(..., axis=1) in memory-safe batches."""
    n = x.size
    batch = max(1, min(n_boot, max_cells // max(n, 1)))
    out, done = [], 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        out.append(np.asarray(stat(x[idx], axis=1)))
        done += b
    return np.concatenate(out)


def bootstrap_ci(values: Iterable[float], stat: Callable = np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for stat(values). `stat` must accept an `axis` argument
    (np.mean, np.median, np.std, ...). Returns (estimate, lo, hi). NaNs are dropped."""
    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("bootstrap_ci: no finite values")
    rng = np.random.default_rng(seed)
    boots = _boot_stats(x, stat, n_boot, rng)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap_ci(loss_a: Iterable[float], loss_b: Iterable[float], stat: Callable = np.median,
                        n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
                        ) -> Tuple[float, float, float]:
    """CI for stat(loss_a - loss_b) over the SAME items. Lower loss is better, so a CI entirely
    below 0 means A is better, entirely above 0 means B is better."""
    a = np.asarray(list(loss_a), dtype=float)
    b = np.asarray(list(loss_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap_ci: loss_a and loss_b must be item-aligned")
    return bootstrap_ci(a - b, stat=stat, n_boot=n_boot, alpha=alpha, seed=seed)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0 or not (0 <= k <= n):
        raise ValueError("wilson_interval: need n > 0 and 0 <= k <= n")
    z = _st.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def within_threshold_report(abs_errors: Iterable[float], thresholds: Sequence[float] = (0.15, 0.25),
                            alpha: float = 0.05) -> Dict[float, dict]:
    """For each threshold: count, n, share and Wilson CI of items with |error| <= threshold.
    Always quote the counts (k of n) next to the percentage."""
    e = np.abs(np.asarray(list(abs_errors), dtype=float))
    e = e[~np.isnan(e)]
    out = {}
    for t in thresholds:
        k = int((e <= t).sum())
        lo, hi = wilson_interval(k, e.size, alpha)
        out[t] = {"k": k, "n": int(e.size), "share": k / e.size, "ci_lo": lo, "ci_hi": hi}
    return out


def error_summary(errors: Iterable[float], n_boot: int = 5_000, seed: int = 42) -> dict:
    """Mean and median absolute error with bootstrap CIs (report both: heavy tails distort means)."""
    e = np.abs(np.asarray(list(errors), dtype=float))
    m, mlo, mhi = bootstrap_ci(e, np.mean, n_boot=n_boot, seed=seed)
    d, dlo, dhi = bootstrap_ci(e, np.median, n_boot=n_boot, seed=seed)
    return {"n": int(np.isfinite(e).sum()), "mean": m, "mean_ci": (mlo, mhi),
            "median": d, "median_ci": (dlo, dhi)}


def paired_comparison(loss_a: Iterable[float], loss_b: Iterable[float], alpha: float = 0.05,
                      n_boot: int = 10_000, seed: int = 42, margin: float = 0.0) -> dict:
    """Compare two methods on the same items (lower loss is better).

    Returns win counts, sign-test p, Wilcoxon p, bootstrap CI of the median difference and a
    plain-language verdict. The verdict is 'no clear difference' unless the CI excludes
    `margin` around zero AND the Wilcoxon p-value is below alpha."""
    a = np.asarray(list(loss_a), dtype=float)
    b = np.asarray(list(loss_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_comparison: arrays must be item-aligned")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    d = a - b
    wins_a, wins_b = int((d < 0).sum()), int((d > 0).sum())
    ties = int((d == 0).sum())
    sign_p = float(_st.binomtest(wins_a, wins_a + wins_b, 0.5).pvalue) if wins_a + wins_b else 1.0
    if np.allclose(d, 0):
        wil_p = 1.0
    else:
        wil_p = float(_st.wilcoxon(d, zero_method="wilcox").pvalue)
    est, lo, hi = bootstrap_ci(d, np.median, n_boot=n_boot, alpha=alpha, seed=seed)
    if hi < -margin and wil_p < alpha:
        verdict = "A better"
    elif lo > margin and wil_p < alpha:
        verdict = "B better"
    else:
        verdict = "no clear difference"
    return {"n": int(d.size), "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "sign_p": sign_p, "wilcoxon_p": wil_p, "median_diff": est, "ci": (lo, hi),
            "verdict": verdict}


def ab_sample_size_two_proportions(p_base: float, p_variant: float, alpha: float = 0.05,
                                   power: float = 0.8) -> int:
    """Per-group sample size for a two-sided two-proportion z-test (normal approximation)."""
    if p_base == p_variant:
        raise ValueError("effect size is zero")
    za, zb = _st.norm.ppf(1 - alpha / 2), _st.norm.ppf(power)
    pbar = (p_base + p_variant) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) +
           zb * math.sqrt(p_base * (1 - p_base) + p_variant * (1 - p_variant))) ** 2
    return int(math.ceil(num / (p_base - p_variant) ** 2))
