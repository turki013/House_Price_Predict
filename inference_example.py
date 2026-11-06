"""Example script demonstrating local inference on a single Riyadh property.

Usage:
    python inference_example.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from aqar_benchmark_mapper import AqarBenchmarkMapper
from features import extract_features

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent


def predict_riyadh_property(
    size: float,
    district: str,
    bedrooms: float = None,
    bathrooms: float = None,
    title: str = "",
    details: str = "",
) -> dict:
    model_path = ROOT / "house_price_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Run train.py first.")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    mapper = AqarBenchmarkMapper()

    # Extract features using single source of truth
    row = extract_features(
        details=details,
        title=title,
        size=size,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        district=district,
    )

    X = pd.DataFrame([row])[feature_cols]

    # Predict baseline log(price)
    pred_log = pipeline.predict(X)[0]
    hist_price_sar = float(np.exp(pred_log))

    # Benchmark calibration
    bm = mapper.get_benchmark(district, size)
    price_2026_sar = hist_price_sar * bm["growth_ratio"]

    return {
        "district": bm["district"],
        "size_m2": size,
        "size_bucket": bm["size_bucket"],
        "estimated_price_2016_sar": hist_price_sar,
        "aqar_growth_ratio": bm["growth_ratio"],
        "estimated_price_2026_sar": price_2026_sar,
        "aqar_benchmark_2026_sar": bm["avg_price_2026"],
        "is_district_median_imputed": bm["is_imputed"],
    }


if __name__ == "__main__":
    sample = {
        "district": "حي النرجس - الرياض",
        "size": 320,
        "bedrooms": 5,
        "bathrooms": 5,
        "title": "فيلا مودرن فاخرة للبيع في النرجس",
        "details": "فيلا مودرن درج صالة مصعد راكب مسبح تكييف مخفي غرفة خادمة غرفة سائق شارع 20 شمالي تشطيب راقي",
    }

    res = predict_riyadh_property(**sample)
    print("=" * 60)
    print("RIYADH PROPERTY PRICE ESTIMATE")
    print("=" * 60)
    for k, v in res.items():
        if isinstance(v, float) and "price" in k:
            print(f"{k:32s}: {v:,.0f} SAR")
        elif isinstance(v, float):
            print(f"{k:32s}: {v:.4f}")
        else:
            print(f"{k:32s}: {v}")
    print("=" * 60)
