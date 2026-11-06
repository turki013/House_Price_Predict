"""FastAPI Serving Application for Riyadh House Price Prediction.

Provides endpoints for:
- Health & metadata inspection
- Price prediction with calibrated benchmarks (both historical and 2026 calibrated levels)
- Input contract validation
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from aqar_benchmark_mapper import AqarBenchmarkMapper
from features import ALL_INPUT_COLS, extract_features

ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="Riyadh Villa Price Estimator API",
    description="Estimates Riyadh villa asking prices based on machine learning and Aqar benchmark calibration.",
    version="2.0.0",
)

# Global artifacts loaded at startup
_MODEL_ARTIFACT = None
_MAPPER = None


def get_model():
    global _MODEL_ARTIFACT
    if _MODEL_ARTIFACT is None:
        model_path = ROOT / "house_price_model.joblib"
        if not model_path.exists():
            raise RuntimeError("Model artifact not found. Please run train.py first.")
        _MODEL_ARTIFACT = joblib.load(model_path)
    return _MODEL_ARTIFACT


def get_mapper():
    global _MAPPER
    if _MAPPER is None:
        _MAPPER = AqarBenchmarkMapper()
    return _MAPPER


class PropertyInput(BaseModel):
    size: float = Field(..., ge=100.0, le=1200.0, description="Plot size in square meters (100 to 1200 m²)")
    district: str = Field(..., min_length=2, description="Riyadh district name, e.g., 'النرجس' or 'حي الياسمين - الرياض'")
    bedrooms: Optional[float] = Field(None, ge=1.0, le=20.0, description="Number of bedrooms (optional, imputed if missing)")
    bathrooms: Optional[float] = Field(None, ge=1.0, le=15.0, description="Number of bathrooms (optional, imputed if missing)")
    details: Optional[str] = Field("", description="Full text description of the property")
    title: Optional[str] = Field("", description="Listing title")


class PricePredictionResponse(BaseModel):
    predicted_price_2016_sar: float = Field(..., description="Estimated price at 2015-2016 baseline price level (SAR)")
    predicted_price_2026_sar: float = Field(..., description="Estimated price scaled to 2026 Aqar market level (SAR)")
    aqar_benchmark_2026_sar: float = Field(..., description="Aqar benchmark median for the district and size bucket (SAR)")
    growth_ratio_applied: float = Field(..., description="Scaling factor from 2015-16 to 2026")
    size_bucket: str = Field(..., description="Unified size bracket")
    is_district_benchmark_imputed: bool = Field(..., description="True if district was not in Aqar index and city median was used")
    features_used: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    properties: List[PropertyInput] = Field(..., min_length=1, max_length=100)


@app.get("/health")
def health_check():
    artifact = get_model()
    return {
        "status": "healthy",
        "model_version": "v2.0.0",
        "champion_architecture": artifact.get("champion_name"),
        "training_samples": artifact.get("n_train"),
    }


@app.post("/predict", response_model=PricePredictionResponse)
def predict_single_price(prop: PropertyInput):
    artifact = get_model()
    mapper = get_mapper()

    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    # Extract single features
    row_dict = extract_features(
        details=prop.details or "",
        title=prop.title or "",
        size=prop.size,
        bedrooms=prop.bedrooms,
        bathrooms=prop.bathrooms,
        district=prop.district,
    )

    df_in = pd.DataFrame([row_dict])[feature_cols]

    # Predict log(price)
    pred_log = pipeline.predict(df_in)[0]
    pred_hist_price = float(np.exp(pred_log))

    # Benchmark calibration to 2026
    bm = mapper.get_benchmark(prop.district, prop.size)
    growth_ratio = bm["growth_ratio"]
    pred_2026_price = pred_hist_price * growth_ratio

    return PricePredictionResponse(
        predicted_price_2016_sar=round(pred_hist_price, -3),
        predicted_price_2026_sar=round(pred_2026_price, -3),
        aqar_benchmark_2026_sar=round(bm["avg_price_2026"], -3),
        growth_ratio_applied=round(growth_ratio, 4),
        size_bucket=bm["size_bucket"],
        is_district_benchmark_imputed=bm["is_imputed"],
        features_used={k: row_dict[k] for k in feature_cols if k != "neighborhood"},
    )


@app.post("/predict_batch", response_model=List[PricePredictionResponse])
def predict_batch_prices(req: BatchPredictionRequest):
    return [predict_single_price(p) for p in req.properties]
