"""Detailed error analysis and worst-performing slice reporting.

Evaluates predictions across key sub-populations:
- By district (sample size >= 20)
- By property size bucket
- By price tier
- By missing indicator status
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import split_frame
from dstk.stats import bootstrap_ci
from modeling import abs_log_error, model_input_cols, price_metrics

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def evaluate_slices():
    split = split_frame()
    test_df = split.test
    artifact = joblib.load(ROOT / "house_price_model.joblib")
    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    X_test = test_df[feature_cols]
    price_true = test_df["price"].to_numpy(float)
    price_pred = np.exp(pipeline.predict(X_test))

    test_df = test_df.copy()
    test_df["pred_price"] = price_pred
    test_df["ape"] = np.abs(price_pred - price_true) / price_true
    test_df["abs_log_err"] = abs_log_error(price_true, price_pred)

    slices_report = []

    # 1. Size Buckets
    size_bins = [0, 275, 400, 600, 2000]
    size_labels = ["<275 m²", "275-400 m²", "400-600 m²", ">600 m²"]
    test_df["size_bucket"] = pd.cut(test_df["size"], bins=size_bins, labels=size_labels)

    for b, grp in test_df.groupby("size_bucket", observed=False):
        if len(grp) == 0:
            continue
        m = price_metrics(grp["price"], grp["pred_price"])
        slices_report.append({
            "slice_category": "size_bucket",
            "slice_name": str(b),
            "n": len(grp),
            "MAPE_pct": m["MAPE_pct"],
            "MdAPE_pct": m["MdAPE_pct"],
            "median_bias_pct": m["median_bias_pct"],
        })

    # 2. Price Tiers
    price_bins = [0, 1_000_000, 2_000_000, 3_500_000, 10_000_000]
    price_labels = ["<1.0M SAR", "1.0M-2.0M SAR", "2.0M-3.5M SAR", ">3.5M SAR"]
    test_df["price_tier"] = pd.cut(test_df["price"], bins=price_bins, labels=price_labels)

    for pt, grp in test_df.groupby("price_tier", observed=False):
        if len(grp) == 0:
            continue
        m = price_metrics(grp["price"], grp["pred_price"])
        slices_report.append({
            "slice_category": "price_tier",
            "slice_name": str(pt),
            "n": len(grp),
            "MAPE_pct": m["MAPE_pct"],
            "MdAPE_pct": m["MdAPE_pct"],
            "median_bias_pct": m["median_bias_pct"],
        })

    # 3. Top Districts (n >= 20)
    for dist, grp in test_df.groupby("district", observed=False):
        if len(grp) >= 20:
            m = price_metrics(grp["price"], grp["pred_price"])
            slices_report.append({
                "slice_category": "district",
                "slice_name": dist,
                "n": len(grp),
                "MAPE_pct": m["MAPE_pct"],
                "MdAPE_pct": m["MdAPE_pct"],
                "median_bias_pct": m["median_bias_pct"],
            })

    # 4. Sensitivity on PPM-invalid test set
    if len(split.test_ppm_invalid) > 0:
        inv_df = split.test_ppm_invalid.copy()
        inv_pred = np.exp(pipeline.predict(inv_df[feature_cols]))
        m_inv = price_metrics(inv_df["price"], inv_pred)
        slices_report.append({
            "slice_category": "sensitivity_ppm_invalid",
            "slice_name": "excluded_ppm_test_rows",
            "n": len(inv_df),
            "MAPE_pct": m_inv["MAPE_pct"],
            "MdAPE_pct": m_inv["MdAPE_pct"],
            "median_bias_pct": m_inv["median_bias_pct"],
        })

    df_slices = pd.DataFrame(slices_report)
    print("\n=== Worst Performing Slices (Sorted by MAPE) ===")
    worst = df_slices.sort_values("MAPE_pct", ascending=False)
    print(worst.to_string(index=False))

    # Top 10 Worst Individual Predictions for Error Analysis
    worst_20 = test_df.sort_values("abs_log_err", ascending=False).head(20)[
        ["district", "size", "bedrooms", "bathrooms", "price", "pred_price", "ape", "listTitle"]
    ]
    worst_20.to_json(REPORTS_DIR / "worst_predictions.json", orient="records", indent=2, force_ascii=False)
    df_slices.to_json(REPORTS_DIR / "slices_report.json", orient="records", indent=2, force_ascii=False)
    print(f"\nSaved slice report to {REPORTS_DIR / 'slices_report.json'}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate_slices()
