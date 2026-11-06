"""Train the relative-ratio model: target = log(price / Aqar_hist_anchor).

The anchor comes from the external Aqar index (not from this dataset's prices),
so it does not leak the target. Same frame, same split, same encoder setup as
train_house_price.py.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from aqar_benchmark_mapper import AqarBenchmarkMapper
from features import MODEL_INPUT_COLS, ROOT, load_training_frame
from modeling import RANDOM_STATE, make_pipeline, price_metrics, split_index

sys.stdout.reconfigure(encoding="utf-8")

frame = load_training_frame()
mapper = AqarBenchmarkMapper()
anchored = mapper.assign_benchmarks(frame, district_col="neighborhood", size_col="size")
train_idx, test_idx = split_index(frame)

X = anchored[MODEL_INPUT_COLS]
y_rel = np.log(anchored["price"] / anchored["benchmark_hist"])
X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
price_tr, price_te = anchored["price"].loc[train_idx], anchored["price"].loc[test_idx]
base_tr, base_te = anchored["benchmark_hist"].loc[train_idx], anchored["benchmark_hist"].loc[test_idx]
villa_te = X_te["is_villa"].eq(1)
direct_te = ~anchored["benchmark_imputed"].loc[test_idx]

pipe = make_pipeline(GradientBoostingRegressor(
    n_estimators=160, max_depth=4, min_samples_leaf=12, min_samples_split=24,
    subsample=0.85, max_features=0.85, learning_rate=0.05, random_state=RANDOM_STATE))
pipe.fit(X_tr, y_rel.loc[train_idx])

pred_tr = base_tr * np.exp(pipe.predict(X_tr))
pred_te = base_te * np.exp(pipe.predict(X_te))
m_tr, m_te = price_metrics(price_tr, pred_tr), price_metrics(price_te, pred_te, villa_te)
anchor_only = price_metrics(price_te, base_te, villa_te)
direct = price_metrics(price_te[direct_te], pred_te[direct_te])

print(f"train={len(X_tr)} test={len(X_te)}  (anchor from a listed district: {int(direct_te.sum())} test rows)")
print(f"relative model  train R2(log price)={m_tr['R2_logprice']:.4f}  test R2(log price)={m_te['R2_logprice']:.4f}")
print(f"relative model  test: MAPE={m_te['MAPE_pct']}%  MdAPE={m_te['MdAPE_pct']}%  "
      f"villa-only MAPE={m_te['MAPE_villa_only_pct']}%  MAE={m_te['MAE_SAR']:,.0f}")
print(f"anchor alone    test: MAPE={anchor_only['MAPE_pct']}%  MdAPE={anchor_only['MdAPE_pct']}%")
print(f"relative model, rows with direct district anchor only: MAPE={direct['MAPE_pct']}%")

joblib.dump({"pipe": pipe, "feature_cols": MODEL_INPUT_COLS, "target": "log(price/benchmark_hist)"},
            ROOT / "house_price_relative_pipeline.joblib")
(ROOT / "relative_model_metrics.json").write_text(json.dumps({
    "n_train": len(X_tr), "n_test": len(X_te), "train": m_tr, "test": m_te,
    "anchor_alone_test": anchor_only, "direct_anchor_rows_test": direct,
}, indent=2), encoding="utf-8")
print("saved house_price_relative_pipeline.joblib (local only, gitignored)")
