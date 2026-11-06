"""Paired comparison of pricing methods on a frozen sample of live listings.

  1. Aqar 2026 district/size average alone
  2. Champion model (log price) x growth ratio
  3. Relative model (log price / anchor) x Aqar 2026 anchor

The listings file (live_eval_dataset.json) is raw scraped data and is NOT
committed; build your own with collect_live_listings.py. Only aggregate metrics
are written to live_eval_summary.json (committed); per-listing rows go to
local_outputs/ (gitignored).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from aqar_benchmark_mapper import AqarBenchmarkMapper
from data import split_frame
from features import extract_features

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
LISTINGS = ROOT / "live_eval_dataset.json"
OUT_LOCAL = ROOT / "local_outputs"
METHODS = {
    "M1_aqar_avg": "1. Aqar average alone",
    "M2_model_x_growth": "2. Champion model x growth",
}
N_BOOT = 10_000

if not LISTINGS.exists():
    print(f"Note: {LISTINGS.name} not present (raw scraped data is gitignored). Skipping live eval run.")
    sys.exit(0)

artifact = joblib.load(ROOT / "house_price_model.joblib")
std = artifact["pipeline"]
feature_cols = artifact["feature_cols"]
mapper = AqarBenchmarkMapper()

split = split_frame()
BED_MED = float(split.train["bedrooms"].median())
BATH_MED = float(split.train["bathrooms"].median())
listings = json.loads(LISTINGS.read_text(encoding="utf-8"))


def build_inputs(impute_beds_baths: bool) -> pd.DataFrame:
    rows = []
    for it in listings:
        title = it["title"]
        details = it["full_text"].replace(title, "", 1)
        beds = BED_MED if impute_beds_baths else it.get("bedrooms")
        baths = BATH_MED if impute_beds_baths else it.get("bathrooms")
        rows.append(extract_features(details=details, title=title, size=it["size"],
                                     bedrooms=beds, bathrooms=baths, district=it["district"]))
    return pd.DataFrame(rows)[feature_cols]


def predictions(X: pd.DataFrame) -> pd.DataFrame:
    bms = [mapper.get_benchmark(it["district"], it["size"]) for it in listings]
    anchor_2026 = np.array([b["avg_price_2026"] for b in bms])
    growth = np.array([b["growth_ratio"] for b in bms])
    return pd.DataFrame({
        "M1_aqar_avg": anchor_2026,
        "M2_model_x_growth": np.exp(std.predict(X)) * growth,
    })


def summarize(pred: pd.Series, actual: np.ndarray) -> dict:
    log_err = np.log(pred.values / actual)
    ape = np.abs(pred.values - actual) / actual
    return {
        "bias_median_pct": round(float((np.exp(np.median(log_err)) - 1) * 100), 2),
        "MAPE_pct": round(float(ape.mean() * 100), 2),
        "MdAPE_pct": round(float(np.median(ape) * 100), 2),
        "within_15pct": round(float((ape <= 0.15).mean() * 100), 1),
        "within_25pct": round(float((ape <= 0.25).mean() * 100), 1),
    }


def paired(pred: pd.DataFrame, actual: np.ndarray, a: str, b: str) -> dict:
    d = np.abs(np.log(pred[a].values / actual)) - np.abs(np.log(pred[b].values / actual))
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    a_wins, b_wins = int((d < 0).sum()), int((d > 0).sum())
    return {
        "a": a, "b": b, "a_closer": a_wins, "b_closer": b_wins,
        "sign_test_p": round(float(stats.binomtest(a_wins, a_wins + b_wins, 0.5).pvalue), 3),
        "wilcoxon_p": round(float(stats.wilcoxon(d).pvalue), 3),
        "mean_diff_abs_log_err": round(float(d.mean()), 4),
        "mean_diff_95ci": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
    }


actual = np.array([float(it["price"]) for it in listings])
summary = {"n_listings": len(listings), "variants": {}}
records = None
for variant, impute in (("beds_baths_imputed_with_training_median", True), ("beds_baths_as_collected", False)):
    X = build_inputs(impute)
    if X[["bedrooms", "bathrooms"]].isna().any().any():
        X = X.fillna({"bedrooms": BED_MED, "bathrooms": BATH_MED})
    pred = predictions(X)
    block = {"metrics": {METHODS[m]: summarize(pred[m], actual) for m in METHODS},
             "paired": [paired(pred, actual, a, b) for a, b in combinations(METHODS, 2)]}
    summary["variants"][variant] = block
    if impute:
        records = pd.concat([pd.DataFrame({"district": [i["district"] for i in listings],
                                           "size": [i["size"] for i in listings], "actual": actual}), pred], axis=1)
    print(f"\n=== {variant} (n={len(listings)}) ===")
    print(pd.DataFrame(block["metrics"]).T.to_string())
    for p in block["paired"]:
        print(f"{p['a']} vs {p['b']}: closer {p['a_closer']}/{p['b_closer']}  sign p={p['sign_test_p']}  "
              f"wilcoxon p={p['wilcoxon_p']}  mean d={p['mean_diff_abs_log_err']} CI={p['mean_diff_95ci']}")

summary["notes"] = ["Live-listing features come from the same features.extract_features used in training.",
                    "Champion model prediction scaled by Aqar benchmark growth ratio."]
(ROOT / "live_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
OUT_LOCAL.mkdir(exist_ok=True)
if records is not None:
    records.to_json(OUT_LOCAL / "live_eval_records.json", orient="records", force_ascii=False, indent=2)
print("\nwrote live_eval_summary.json")
