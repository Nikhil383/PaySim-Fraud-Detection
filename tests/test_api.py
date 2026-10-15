"""Tests for FastAPI prediction service."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from fraud_capstone.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health returns HTTP 200 with service and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_artifact_present" in data


def test_score_endpoint_missing_artifacts(client):
    """Verify that /score raises 503 or error when model artifacts are absent."""
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    # Patch get_predictor to simulate missing files
    with patch("fraud_capstone.api.get_predictor") as mock_predictor:
        mock_predictor.side_effect = FileNotFoundError("Model artifacts not found.")
        response = client.post("/score", json=payload)
        assert response.status_code == 503
        assert "Model unavailable" in response.json()["detail"]


def test_score_endpoint_successful_prediction(client):
    """Verify that /score returns valid probabilities and predictions when model is active."""
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    mock_pred_instance = MagicMock()
    mock_pred_instance.predict_single.return_value = {
        "fraud_probability": 0.895,
        "is_fraud_prediction": 1,
        "decision_threshold": 0.5,
    }

    with patch("fraud_capstone.api.get_predictor", return_value=mock_pred_instance):
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["fraud_probability"] == 0.895
        assert data["is_fraud_prediction"] == 1
        assert data["decision_threshold"] == 0.5
        assert data["status"] == "success"


def test_score_endpoint_invalid_payload(client):
    """Verify validation error when payload has missing or negative amount."""
    invalid_payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": -50.0,  # Invalid negative amount
        "nameOrig": "C123",
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 50.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/score", json=invalid_payload)
    assert response.status_code == 422
