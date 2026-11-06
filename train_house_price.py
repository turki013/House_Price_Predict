"""Train the standard (absolute price) model + baselines on log1p(price).

Everything shares features.py (feature extraction, dedup, bounds) and
modeling.py (leak-free pipeline, fixed split). The test split is evaluated once
here; note it was also consulted when hyper-parameters were chosen in earlier
rounds, so it is not a pristine holdout.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from features import FEATURE_COLS, MODEL_INPUT_COLS, ROOT, load_training_frame
from modeling import RANDOM_STATE, make_pipeline, price_metrics, split_index

sys.stdout.reconfigure(encoding="utf-8")

frame = load_training_frame()
train_idx, test_idx = split_index(frame)
X, y = frame[MODEL_INPUT_COLS], np.log1p(frame["price"])
X_tr, X_te, y_tr = X.loc[train_idx], X.loc[test_idx], y.loc[train_idx]
price_tr, price_te = frame["price"].loc[train_idx], frame["price"].loc[test_idx]
villa_te = X_te["is_villa"].eq(1)
print(f"train={len(X_tr)} test={len(X_te)} | share of rows with is_villa=1: {X['is_villa'].mean():.1%} (villa-only filtering is a no-op on this dataset)")

GBR_PARAMS = dict(n_estimators=160, max_depth=4, min_samples_leaf=12, min_samples_split=24,
                  subsample=0.85, max_features=0.85, learning_rate=0.05, random_state=RANDOM_STATE)

models = {
    "Dummy (mean)": (make_pipeline(DummyRegressor(strategy="mean")), X_tr, y_tr),
    "Ridge (alpha=10)": (make_pipeline(Ridge(alpha=10.0), scale=True), X_tr, y_tr),
    "GBR depth=4": (make_pipeline(GradientBoostingRegressor(**GBR_PARAMS)), X_tr, y_tr),
    "XGBoost": (make_pipeline(XGBRegressor(
        n_estimators=140, max_depth=4, min_child_weight=8, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=2.0, learning_rate=0.05, random_state=RANDOM_STATE)), X_tr, y_tr),
}

rows = []
for name, (pipe, Xf, yf) in models.items():
    pipe.fit(Xf, yf)
    tr_pred = np.expm1(pipe.predict(X_tr))
    te_pred = np.expm1(pipe.predict(X_te))
    m_tr = price_metrics(price_tr, tr_pred)
    m_te = price_metrics(price_te, te_pred, villa_te)
    rows.append({"Model": name, "Train_R2_log": m_tr["R2_logprice"], "Test_R2_log": m_te["R2_logprice"],
                 "Gap_R2": round(m_tr["R2_logprice"] - m_te["R2_logprice"], 4), **{
                     k: m_te[k] for k in ("MAE_SAR", "MAPE_pct", "MdAPE_pct", "MAPE_villa_only_pct")}})
results = pd.DataFrame(rows)
print(results.to_string(index=False))

champion = models["GBR depth=4"][0]
names = [n.split("__", 1)[1] for n in champion[:-1].get_feature_names_out()]
importances = (pd.DataFrame({"feature": names, "importance": champion[-1].feature_importances_})
               .sort_values("importance", ascending=False))
print(importances.head(10).to_string(index=False))

joblib.dump({"pipe": champion, "feature_cols": MODEL_INPUT_COLS, "target": "log1p(price)"},
            ROOT / "house_price_pipeline.joblib")

(ROOT / "model_metadata.json").write_text(json.dumps({
    "target": "log1p(price)", "rows_after_dedup_and_bounds": len(frame),
    "n_train": len(X_tr), "n_test": len(X_te), "features": FEATURE_COLS,
    "metrics": rows, "top_features": importances.head(10).to_dict("records"),
    "environment": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
}, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved house_price_pipeline.joblib (local only, gitignored) and model_metadata.json")

