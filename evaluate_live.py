"""Paired comparison of three pricing methods on a frozen sample of live listings.

  1. Aqar 2026 district/size average alone
  2. Standard model (log price) x growth ratio
  3. Relative model (log price / anchor) x Aqar 2026 anchor

The listings file (live_eval_dataset.json) is raw scraped data and is NOT
committed; build your own with collect_live_listings.py. Only aggregate metrics
are written to live_eval_summary.json (committed); per-listing rows go to
local_outputs/ (gitignored).
"""
import json
import sys
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from aqar_benchmark_mapper import AqarBenchmarkMapper
from features import MODEL_INPUT_COLS, ROOT, extract_features, load_training_frame

sys.stdout.reconfigure(encoding="utf-8")

LISTINGS = ROOT / "live_eval_dataset.json"
OUT_LOCAL = ROOT / "local_outputs"
METHODS = {"M1_aqar_avg": "1. Aqar average alone",
           "M2_model_x_growth": "2. Standard model x growth",
           "M3_relative_model": "3. Relative model x 2026 anchor"}
N_BOOT = 10_000

if not LISTINGS.exists():
    sys.exit(f"{LISTINGS.name} not found. Run collect_live_listings.py first (data is not in the repo).")

std = joblib.load(ROOT / "house_price_pipeline.joblib")["pipe"]
rel = joblib.load(ROOT / "house_price_relative_pipeline.joblib")["pipe"]
mapper = AqarBenchmarkMapper()
train_frame = load_training_frame(verbose=False)
BED_MED, BATH_MED = float(train_frame["bedrooms"].median()), float(train_frame["bathrooms"].median())
listings = json.loads(LISTINGS.read_text(encoding="utf-8"))


def build_inputs(impute_beds_baths: bool) -> pd.DataFrame:
    rows = []
    for it in listings:
        title = it["title"]
        details = it["full_text"].replace(title, "", 1)
        beds, baths = (BED_MED, BATH_MED) if impute_beds_baths else (it.get("bedrooms"), it.get("bathrooms"))
        rows.append(extract_features(details=details, title=title, size=it["size"],
                                     bedrooms=beds, bathrooms=baths, district=it["district"]))
    return pd.DataFrame(rows)[MODEL_INPUT_COLS]


def predictions(X: pd.DataFrame) -> pd.DataFrame:
    bms = [mapper.get_benchmark(it["district"], it["size"]) for it in listings]
    anchor_2026 = np.array([b["avg_price_2026"] for b in bms])
    growth = np.array([b["growth_ratio"] for b in bms])
    return pd.DataFrame({
        "M1_aqar_avg": anchor_2026,
        "M2_model_x_growth": np.expm1(std.predict(X)) * growth,
        "M3_relative_model": anchor_2026 * np.exp(rel.predict(X)),
    })


def summarize(pred: pd.Series, actual: np.ndarray) -> dict:
    log_err = np.log(pred.values / actual)
    ape = np.abs(pred.values - actual) / actual
    return {
        "bias_median_pct": round(float((np.exp(np.median(log_err)) - 1) * 100), 2),  # from median log(pred/actual)
        "MAPE_pct": round(float(ape.mean() * 100), 2),
        "MdAPE_pct": round(float(np.median(ape) * 100), 2),
        "within_15pct": round(float((ape <= 0.15).mean() * 100), 1),
        "within_25pct": round(float((ape <= 0.25).mean() * 100), 1),
    }


def paired(pred: pd.DataFrame, actual: np.ndarray, a: str, b: str) -> dict:
    """d = |log err_a| - |log err_b|; negative d means method a is closer."""
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
    if X[["bedrooms", "bathrooms"]].isna().any().any():  # collected values may be missing -> impute explicitly
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

summary["notes"] = ["Primary = first variant (bedrooms/bathrooms were not reliably collected).",
                    "Live-listing features come from the same features.extract_features used in training.",
                    "Three pairwise tests are reported without multiplicity correction."]
(ROOT / "live_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
OUT_LOCAL.mkdir(exist_ok=True)
records.to_json(OUT_LOCAL / "live_eval_records.json", orient="records", force_ascii=False, indent=2)
print("\nwrote live_eval_summary.json (aggregate) and local_outputs/live_eval_records.json (local only)")
