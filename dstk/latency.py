"""Latency measurement for inference callables."""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np


def measure_latency(fn: Callable[[], object], n: int = 100, warmup: int = 5) -> Dict[str, float]:
    """Call fn() warmup + n times; return p50/p95/p99/mean/max in milliseconds (warm-up excluded)."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000.0)
    a = np.asarray(t)
    return {"n": float(n), "p50_ms": float(np.quantile(a, 0.5)), "p95_ms": float(np.quantile(a, 0.95)),
            "p99_ms": float(np.quantile(a, 0.99)), "mean_ms": float(a.mean()), "max_ms": float(a.max())}
