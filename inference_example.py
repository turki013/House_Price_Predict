"""Price one listing with the three methods. Requires the local .joblib bundles
(run train_house_price.py and train_relative_model.py first)."""
import sys

import joblib
import numpy as np
import pandas as pd

from aqar_benchmark_mapper import AqarBenchmarkMapper
from features import MODEL_INPUT_COLS, ROOT, extract_features, load_training_frame

sys.stdout.reconfigure(encoding="utf-8")

std = joblib.load(ROOT / "house_price_pipeline.joblib")["pipe"]
rel = joblib.load(ROOT / "house_price_relative_pipeline.joblib")["pipe"]
mapper = AqarBenchmarkMapper()
_train = load_training_frame(verbose=False)
_BED, _BATH = float(_train["bedrooms"].median()), float(_train["bathrooms"].median())


def price_listing(*, details, title, size, district, bedrooms=None, bathrooms=None) -> dict:
    """Missing bedrooms/bathrooms are imputed with the training median and reported."""
    imputed = bedrooms is None or bathrooms is None
    row = extract_features(details=details, title=title, size=size, district=district,
                           bedrooms=_BED if bedrooms is None else bedrooms,
                           bathrooms=_BATH if bathrooms is None else bathrooms)
    X = pd.DataFrame([row])[MODEL_INPUT_COLS]
    bm = mapper.get_benchmark(district, size)
    return {
        "aqar_2026_average": bm["avg_price_2026"],
        "model_x_growth": float(np.expm1(std.predict(X))[0]) * bm["growth_ratio"],
        "relative_model_x_anchor": bm["avg_price_2026"] * float(np.exp(rel.predict(X))[0]),
        "size_bucket": bm["size_bucket"], "anchor_is_city_median": bm["is_imputed"],
        "bedrooms_bathrooms_imputed": imputed,
    }


if __name__ == "__main__":
    example = price_listing(
        details="فيلا بتصميم عصري 4 غرف ماستر مجلس وصالة غرفة خادمة غرفة سائق مصعد راكب دبل هايت تكييف مخفي واجهة شمالية",
        title="فيلا للبيع في حي النرجس", size=200, district="النرجس")
    for k, v in example.items():
        print(f"{k:28s} {v:,.0f}" if isinstance(v, float) else f"{k:28s} {v}")
