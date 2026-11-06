"""Experiment runner and model training script.

Executes the model ladder on grouped cross-validation, runs hyperparameter tuning,
evaluates on the held-out test set, and saves the champion pipeline and metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.base import clone
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from aqar_benchmark_mapper import AqarBenchmarkMapper
from data import cv_splits, split_frame
from dstk.explog import log_experiment
from dstk.stats import bootstrap_ci, paired_comparison
from modeling import (
    COMPLEXITY_ORDER,
    GBR_PR1_PARAMS,
    HGB_SEARCH_SPACE,
    RANDOM_STATE,
    XGB_SEARCH_SPACE,
    abs_log_error,
    candidates,
    make_pipeline,
    model_input_cols,
    price_metrics,
)

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
EXPLOG_PATH = str(REPORTS_DIR / "experiments.jsonl")
RESULTS_PATH = REPORTS_DIR / "results.json"


def cv_evaluate(model, X: pd.DataFrame, y: np.ndarray, cv, prices: np.ndarray) -> dict:
    oof_preds = np.zeros(len(y), dtype=float)
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        m = clone(model)
        m.fit(X_tr, y_tr)
        pred_val = m.predict(X_val)
        oof_preds[val_idx] = pred_val
        val_loss = abs_log_error(prices[val_idx], np.exp(pred_val)).mean()
        fold_scores.append(val_loss)

    oof_price_pred = np.exp(oof_preds)
    metrics = price_metrics(prices, oof_price_pred)
    metrics["cv_mean_loss"] = float(np.mean(fold_scores))
    metrics["cv_std_loss"] = float(np.std(fold_scores))
    return metrics, oof_price_pred


def run_experiments():
    print("=" * 60)
    print("1. Loading Data & Constructing Strict Group Split")
    print("=" * 60)
    split = split_frame()
    train_df, test_df = split.train, split.test
    mapper = AqarBenchmarkMapper()

    # Save audit log
    audit_df = split.audit.to_frame()
    audit_df.to_markdown(REPORTS_DIR / "data_audit.md", index=False)
    print(audit_df.to_string(index=False))

    cv = cv_splits(train_df)
    cols_v2 = model_input_cols("v2")
    X_train_v2 = train_df[cols_v2]
    y_train = np.log(train_df["price"].to_numpy(float))
    price_train = train_df["price"].to_numpy(float)

    cols_pr1 = model_input_cols("pr1")
    X_train_pr1 = train_df[cols_pr1]

    X_test_v2 = test_df[cols_v2]
    y_test = np.log(test_df["price"].to_numpy(float))
    price_test = test_df["price"].to_numpy(float)

    print("\n" + "=" * 60)
    print("2. Cross-Validation Model Ladder (on Train only)")
    print("=" * 60)

    # Clear previous experiment log
    if Path(EXPLOG_PATH).exists():
        Path(EXPLOG_PATH).unlink()

    cand_v2 = candidates("v2", mapper=mapper)
    cv_results = {}
    oof_predictions = {}

    for name, model in cand_v2.items():
        print(f"Evaluating {name} on 5-fold grouped CV...")
        m, oof = cv_evaluate(model, X_train_v2, y_train, cv, price_train)
        cv_results[name] = m
        oof_predictions[name] = oof
        print(f"  -> MAPE: {m['MAPE_pct']:.2f}% | MdAPE: {m['MdAPE_pct']:.2f}% | Mean Abs Log Error: {m['mean_abs_log_error']:.4f}")

        log_experiment(
            EXPLOG_PATH,
            id=f"CV_{name}",
            hypothesis=f"Baseline/Candidate ladder evaluation for {name}",
            change=f"Model architecture: {name}",
            rows=f"train CV n={len(train_df)} across 5 grouped folds",
            metric="MAPE_pct",
            value=m["MAPE_pct"],
            ci_lo=m["MAPE_pct"] - m["cv_std_loss"] * 100,
            ci_hi=m["MAPE_pct"] + m["cv_std_loss"] * 100,
            baseline_value=cv_results["dummy_median"]["MAPE_pct"],
            paired_verdict="A better" if m["MAPE_pct"] < cv_results["dummy_median"]["MAPE_pct"] else "no clear difference",
            decision="candidate",
        )

    print("\n" + "=" * 60)
    print("3. Feature Ablation: PR1 vs V2 Cleaned Features (using GBR)")
    print("=" * 60)
    gbr_pr1 = candidates("pr1")["gbr_pr1_params"]
    m_pr1, oof_pr1 = cv_evaluate(gbr_pr1, X_train_pr1, y_train, cv, price_train)
    m_v2 = cv_results["gbr_pr1_params"]
    print(f"GBR (PR1 features, 31 cols): MAPE={m_pr1['MAPE_pct']:.2f}%, MdAPE={m_pr1['MdAPE_pct']:.2f}%")
    print(f"GBR (V2 features, pruned):    MAPE={m_v2['MAPE_pct']:.2f}%, MdAPE={m_v2['MdAPE_pct']:.2f}%")

    loss_pr1 = abs_log_error(price_train, oof_pr1)
    loss_v2 = abs_log_error(price_train, oof_predictions["gbr_pr1_params"])
    comp = paired_comparison(loss_v2, loss_pr1)
    print(f"Paired comparison V2 vs PR1: verdict={comp['verdict']}, median_diff={comp['median_diff']:.5f}, p={comp['wilcoxon_p']:.4f}")

    print("\n" + "=" * 60)
    print("4. Hyperparameter Tuning (HistGradientBoosting on Train CV)")
    print("=" * 60)
    base_hgb = candidates("v2")["hgb"]
    search = RandomizedSearchCV(
        base_hgb,
        param_distributions=HGB_SEARCH_SPACE,
        n_iter=25,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train_v2, y_train)
    best_hgb = search.best_estimator_
    print("Best HGB Params:", search.best_params_)
    m_tuned, oof_tuned = cv_evaluate(best_hgb, X_train_v2, y_train, cv, price_train)
    cv_results["hgb_tuned"] = m_tuned
    oof_predictions["hgb_tuned"] = oof_tuned
    print(f"Tuned HGB on CV: MAPE={m_tuned['MAPE_pct']:.2f}%, MdAPE={m_tuned['MdAPE_pct']:.2f}%")

    # Select champion based on CV mean absolute log error
    champion_name = "hgb_tuned" if m_tuned["mean_abs_log_error"] <= cv_results["gbr_pr1_params"]["mean_abs_log_error"] else "gbr_pr1_params"
    champion_model = best_hgb if champion_name == "hgb_tuned" else cand_v2["gbr_pr1_params"]
    print(f"\nSelected Champion for Final Test: {champion_name}")

    print("\n" + "=" * 60)
    print("5. Held-out Final Test Evaluation (Touch Once)")
    print("=" * 60)
    # Fit champion and baselines on full training set, predict on test
    test_evaluations = {}

    models_to_test = {
        "dummy_median": cand_v2["dummy_median"],
        "ridge": cand_v2["ridge"],
        "gbr_pr1_params": cand_v2["gbr_pr1_params"],
        "relative_gbr": cand_v2["relative_gbr"],
        "hgb_tuned": best_hgb,
    }

    test_preds = {}
    for name, model in models_to_test.items():
        m_fit = clone(model)
        m_fit.fit(X_train_v2, y_train)
        pred_log = m_fit.predict(X_test_v2)
        pred_price = np.exp(pred_log)
        test_preds[name] = pred_price
        metrics = price_metrics(price_test, pred_price)

        # Bootstrap CI for MAPE and MdAPE
        apes = np.abs(pred_price - price_test) / price_test * 100
        _, mape_lo, mape_hi = bootstrap_ci(apes, np.mean, n_boot=2000)
        _, mdape_lo, mdape_hi = bootstrap_ci(apes, np.median, n_boot=2000)
        metrics["MAPE_ci"] = [round(mape_lo, 2), round(mape_hi, 2)]
        metrics["MdAPE_ci"] = [round(mdape_lo, 2), round(mdape_hi, 2)]
        test_evaluations[name] = metrics

    test_summary_df = pd.DataFrame(test_evaluations).T
    print(test_summary_df[["MAPE_pct", "MdAPE_pct", "within_10pct_share", "within_20pct_share", "R2_log_price", "MAE_SAR"]].to_string())

    # Paired comparisons against champion on test set
    champ_pred = test_preds[champion_name]
    champ_loss = abs_log_error(price_test, champ_pred)
    pairwise_tests = {}
    for name, pred in test_preds.items():
        if name == champion_name:
            continue
        base_loss = abs_log_error(price_test, pred)
        pw = paired_comparison(champ_loss, base_loss)
        pairwise_tests[f"{champion_name}_vs_{name}"] = pw
        print(f"Pairwise {champion_name} vs {name}: {pw['verdict']} (p_wilcox={pw['wilcoxon_p']:.4f}, median_diff={pw['median_diff']:.4f})")

    print("\n" + "=" * 60)
    print("6. Serialization & Packaging")
    print("=" * 60)
    # Fit final champion on all training data
    champion_pipeline = clone(champion_model)
    champion_pipeline.fit(X_train_v2, y_train)

    # Save frozen test set for regression testing
    test_export = test_df[cols_v2].copy()
    test_export.to_parquet(ROOT / "frozen_test.parquet", index=False)
    np.save(ROOT / "frozen_test_prices.npy", price_test)

    # Save model artifact
    joblib.dump(
        {
            "pipeline": champion_pipeline,
            "feature_set": "v2",
            "feature_cols": cols_v2,
            "target": "log(price)",
            "champion_name": champion_name,
            "n_train": len(train_df),
            "test_metrics": test_evaluations[champion_name],
        },
        ROOT / "house_price_model.joblib",
    )

    # Save results.json
    results_payload = {
        "cv_results": cv_results,
        "test_results": test_evaluations,
        "pairwise_test_vs_champion": pairwise_tests,
        "champion_name": champion_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_set": "v2",
        "feature_count": len(cols_v2),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved champion pipeline to {ROOT / 'house_price_model.joblib'}")

    return results_payload


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_experiments()
