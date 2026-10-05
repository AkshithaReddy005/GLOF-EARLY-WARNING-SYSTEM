"""
test_flask_routes.py
====================
Integration tests for Flask web application routes.
Tests route availability and response codes without requiring
a trained model — uses mocked inference results.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


MOCK_RESULT = {
    "class": 0,
    "message": 0,
    "probability": 0.23,
    "risk_percentage": "23.0%",
    "risk_label": "✅ Normal — Low Risk",
    "inference_ms": 12.5,
    "engine": "Keras .h5",
    "lake_id": "GL_TEST",
    "latitude": 30.128,
    "longitude": 93.898,
}


@pytest.fixture
def client():
    """Create a Flask test client with mocked model loading."""
    with patch.dict(os.environ, {"ENABLE_MQTT": "0"}):
        with patch("main.input_model", return_value=MOCK_RESULT):
            from main import app
            app.config["TESTING"] = True
            with app.test_client() as c:
                yield c


class TestPageRoutes:
    """Test that all HTML page routes return 200."""

    def test_index_get(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_about(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_services(self, client):
        resp = client.get("/services")
        assert resp.status_code == 200

    def test_contact(self, client):
        resp = client.get("/contact")
        assert resp.status_code == 200

    def test_sign(self, client):
        resp = client.get("/sign")
        assert resp.status_code == 200

    def test_batch_get(self, client):
        resp = client.get("/predict/batch")
        assert resp.status_code == 200


class TestHealthEndpoint:
    """Test the /health monitoring probe."""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"


class TestAPIPredict:
    """Test the /api/predict REST endpoint."""

    def test_predict_missing_input(self, client):
        resp = client.post("/api/predict",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "error" in data

    def test_predict_with_input(self, client):
        resp = client.post("/api/predict",
                           data=json.dumps({"input_string": "1,2,3,4,5"}),
                           content_type="application/json")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "class" in data
        assert "probability" in data


class TestAPIHistory:
    """Test the /api/history endpoint."""

    def test_history_returns_list(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "predictions" in data
        assert "count" in data


class TestAPIStats:
    """Test the /api/stats endpoint."""

    def test_stats_returns_dict(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "total_predictions" in data
