"""Test suite for the FastAPI Serving API."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app, get_model, get_mapper

client = TestClient(app)


def test_api_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "v2.0.0"
    assert "champion_architecture" in data


def test_api_single_prediction_valid():
    payload = {
        "size": 350.0,
        "district": "حي النرجس - الرياض",
        "bedrooms": 5,
        "bathrooms": 5,
        "title": "فيلا مودرن للبيع في النرجس",
        "details": "فيلا مودرن درج صالة مصعد مسبح غرفة خادمة غرفة سائق شارع 20 شمالي",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_price_2026_sar"] > 1_000_000
    assert data["predicted_price_2016_sar"] > 500_000
    assert data["size_bucket"] == "275-400 m²"
    assert data["growth_ratio_applied"] > 1.0
    assert not data["is_district_benchmark_imputed"]


def test_api_rejection_on_invalid_bounds():
    # Size below 100 m²
    res = client.post("/predict", json={"size": 50, "district": "النرجس"})
    assert res.status_code == 422

    # Size above 1200 m²
    res = client.post("/predict", json={"size": 1500, "district": "النرجس"})
    assert res.status_code == 422


def test_api_batch_prediction():
    req = {
        "properties": [
            {"size": 250, "district": "الياسمين"},
            {"size": 450, "district": "العارض"},
        ]
    }
    res = client.post("/predict_batch", json=req)
    assert res.status_code == 200
    assert len(res.json()) == 2
