import os
import json
import warnings
import numpy as np
import pandas as pd

class AqarBenchmarkMapper:
    """
    Empirical benchmark mapper linking verified Riyadh residential indicators
    from Aqar to property listings by district and unified size bucket.
    Baseline is smoothed across contiguous 2015-2016 half-years (1_2015 to 2_2016)
    to eliminate single-period survey noise when scaling to 2026.
    """
    def __init__(self, json_path=None):
        if json_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, "aqar_benchmarks_2026.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(os.getcwd(), "aqar_benchmarks_2026.json")
                
        self.json_path = os.path.abspath(json_path)
        self.lookup = {}
        self.bucket_medians = {}
        self._load_benchmarks()

    def _load_benchmarks(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Benchmark file not found: {self.json_path}")
            
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        bucket_ratios = {}
        bucket_prices_2026 = {}
        bucket_prices_hist = {}

        # If loading aqar_index_full.json, compute multi-period mean
        if isinstance(raw_data, list) and len(raw_data) > 0 and "period_range" in raw_data[0]:
            df_idx = pd.DataFrame(raw_data)
            base = (df_idx[df_idx.period_range.isin(["1_2015", "2_2015", "1_2016", "2_2016"])]
                      .groupby(["district", "size_bucket"])["value"].mean()
                      .rename("hist_price_2015_16").reset_index())
            now = df_idx[df_idx.period_range == "1_2026"][["district", "size_bucket", "value"]].rename(columns={"value": "avg_price_2026"})
            g = base.merge(now, on=["district", "size_bucket"])
            g["growth_ratio"] = g["avg_price_2026"] / g["hist_price_2015_16"]
            items = g.to_dict(orient="records")
        else:
            items = raw_data

        for item in items:
            dist = self.clean_district(item["district"])
            bucket = item["size_bucket"].strip()
            avg_2026 = float(item["avg_price_2026"])
            hist_base = float(item.get("hist_price_2015_16", item.get("hist_price_base", item.get("hist_price_2020", 0.0))))
            growth = float(item.get("growth_ratio", avg_2026 / hist_base if hist_base > 0 else 1.0))

            self.lookup[(dist, bucket)] = {
                "district": dist,
                "size_bucket": bucket,
                "avg_price_2026": avg_2026,
                "hist_price_base": hist_base,
                "growth_ratio": growth
            }

            bucket_ratios.setdefault(bucket, []).append(growth)
            bucket_prices_2026.setdefault(bucket, []).append(avg_2026)
            bucket_prices_hist.setdefault(bucket, []).append(hist_base)

        # Compute data-derived medians for unseen districts
        for bucket in ["150-275 m²", "275-400 m²", "400-600 m²", "600-900 m²"]:
            if bucket in bucket_ratios and len(bucket_ratios[bucket]) > 0:
                self.bucket_medians[bucket] = {
                    "growth_ratio": float(np.median(bucket_ratios[bucket])),
                    "avg_price_2026": float(np.median(bucket_prices_2026[bucket])),
                    "hist_price_base": float(np.median(bucket_prices_hist[bucket]))
                }
            else:
                raise ValueError(f"No benchmark data available to compute median for size bucket: {bucket}")

    @staticmethod
    def get_size_bucket(size_m2):
        """
        Unified size bucketing matching Aqar residential indicator bins:
        - size < 275 -> '150-275 m²'
        - 275 <= size <= 400 -> '275-400 m²'
        - 400 < size <= 600 -> '400-600 m²'
        - size > 600 -> '600-900 m²'
        """
        s = float(size_m2)
        if s < 275:
            return "150-275 m²"
        elif s <= 400:
            return "275-400 m²"
        elif s <= 600:
            return "400-600 m²"
        else:
            return "600-900 m²"

    @staticmethod
    def clean_district(district_str):
        if not district_str:
            return ""
        return str(district_str).replace("حي", "").replace("- الرياض", "").replace("الرياض", "").strip()

    def get_benchmark(self, district, size_m2):
        bucket = self.get_size_bucket(size_m2)
        clean_dist = self.clean_district(district)
        key = (clean_dist, bucket)

        if key in self.lookup:
            res = self.lookup[key].copy()
            res["is_imputed"] = False
            return res
        else:
            med = self.bucket_medians[bucket]
            return {
                "district": clean_dist,
                "size_bucket": bucket,
                "avg_price_2026": med["avg_price_2026"],
                "hist_price_base": med["hist_price_base"],
                "growth_ratio": med["growth_ratio"],
                "is_imputed": True
            }

    def calibrate(self, hist_price_sar, district, size_m2):
        bm = self.get_benchmark(district, size_m2)
        pred_2026 = float(hist_price_sar) * bm["growth_ratio"]
        return pred_2026, bm

    def assign_benchmarks(self, df, district_col="neighborhood", size_col="size"):
        benchmarks_2026 = []
        growth_ratios = []
        hist_bases = []
        is_imputeds = []

        for _, row in df.iterrows():
            bm = self.get_benchmark(row[district_col], row[size_col])
            benchmarks_2026.append(bm["avg_price_2026"])
            hist_bases.append(bm["hist_price_base"])
            growth_ratios.append(bm["growth_ratio"])
            is_imputeds.append(bm["is_imputed"])

        df_out = df.copy()
        df_out["benchmark_2026"] = benchmarks_2026
        df_out["benchmark_hist"] = hist_bases
        df_out["growth_ratio"] = growth_ratios
        df_out["benchmark_imputed"] = is_imputeds
        return df_out
