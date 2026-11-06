"""Map (district, plot size) to the external Aqar residential price index.

``aqar_benchmarks_2026.json`` holds, per (district, size bucket), the index
value averaged over the half-years 1_2015..2_2016 (``hist_price_2015_16``) and
the value for 1_2026 (``avg_price_2026``). The 2015-16 average is the level
that best matched the training listings in PR #1's comparison.

Districts that are not in the file receive the median of the listed districts
for the same size bucket and are flagged ``is_imputed=True``; callers must
surface that flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import clean_district

DEFAULT_PATH = Path(__file__).resolve().parent / "aqar_benchmarks_2026.json"
BUCKETS = ["150-275 m²", "275-400 m²", "400-600 m²", "600-900 m²"]


def get_size_bucket(size_m2) -> str:
    """< 275 | 275-400 (inclusive) | (400, 600] | > 600, matching the index bins."""
    s = float(size_m2)
    if s < 275:
        return BUCKETS[0]
    if s <= 400:
        return BUCKETS[1]
    if s <= 600:
        return BUCKETS[2]
    return BUCKETS[3]


class AqarBenchmarkMapper:
    def __init__(self, json_path: str | Path | None = None):
        self.json_path = Path(json_path) if json_path else DEFAULT_PATH
        if not self.json_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.json_path}")
        items = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.lookup: dict = {}
        for it in items:
            hist = float(it["hist_price_2015_16"])
            now = float(it["avg_price_2026"])
            if hist <= 0 or now <= 0:
                raise ValueError(f"non-positive benchmark value in {it}")
            key = (clean_district(it["district"]), it["size_bucket"].strip())
            self.lookup[key] = {"avg_price_2026": now, "hist_price_base": hist, "growth_ratio": now / hist}
        self.bucket_medians: dict = {}
        for b in BUCKETS:
            vals = [v for (d, bb), v in self.lookup.items() if bb == b]
            if not vals:
                raise ValueError(f"No benchmark data for size bucket {b}")
            self.bucket_medians[b] = {k: float(np.median([v[k] for v in vals]))
                                      for k in ("avg_price_2026", "hist_price_base", "growth_ratio")}
        self.districts = sorted({d for d, _ in self.lookup})

    get_size_bucket = staticmethod(get_size_bucket)

    def get_benchmark(self, district, size_m2) -> dict:
        bucket = get_size_bucket(size_m2)
        d = clean_district(district)
        hit = self.lookup.get((d, bucket))
        res = dict(hit if hit else self.bucket_medians[bucket])
        res.update({"district": d, "size_bucket": bucket, "is_imputed": hit is None})
        return res

    def benchmarks_frame(self, districts, sizes) -> pd.DataFrame:
        """Vectorised ``get_benchmark`` for aligned sequences."""
        rows = [self.get_benchmark(d, s) for d, s in zip(districts, sizes)]
        return pd.DataFrame(rows)

    def assign_benchmarks(self, df: pd.DataFrame, district_col="neighborhood", size_col="size") -> pd.DataFrame:
        bm = self.benchmarks_frame(df[district_col], df[size_col])
        out = df.copy()
        out["benchmark_2026"] = bm["avg_price_2026"].to_numpy()
        out["benchmark_hist"] = bm["hist_price_base"].to_numpy()
        out["growth_ratio"] = bm["growth_ratio"].to_numpy()
        out["benchmark_imputed"] = bm["is_imputed"].to_numpy()
        return out
