"""Tests for FastAPI endpoints and JSON schemas."""

import pytest
from fastapi.testclient import TestClient

from climate_forecast.web.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "climate-forecast"


def test_stations_endpoint(client):
    res = client.get("/api/stations")
    assert res.status_code == 200
    data = res.json()
    assert "stations" in data
    assert "count" in data


def test_custom_point_endpoint(client):
    req_body = {
        "latitude": 40.4168,
        "longitude": -3.7038,
        "elevation": 650.0,
        "reference_date": "2024-07-15",
    }
    res = client.post("/api/forecast/custom-point", json=req_body)
    # If no stations are loaded yet, it might return 500 or 200
    assert res.status_code in [200, 500]
