"""FastAPI request/response schema contract.

Runs with whatever weights are available - a real checkpoint if one exists, an
untrained model otherwise - because this asserts the CONTRACT, not the accuracy.
That is what lets it run in CI with no data and no trained model.
"""

import importlib.util
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_ROOT, "scripts", "serving", "app.py")

EXPECTED_SAMPLES = 6000
EXPECTED_FS = 2000


def load_app_module():
    spec = importlib.util.spec_from_file_location("serving_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client():
    module = load_app_module()
    with TestClient(module.app) as test_client:
        yield test_client


def make_payload(n_samples=EXPECTED_SAMPLES, fs=EXPECTED_FS):
    rng = np.random.default_rng(0)
    time = np.arange(n_samples) / float(EXPECTED_FS)
    ecg = (np.sin(2 * np.pi * 1.2 * time) + 0.1 * rng.normal(size=n_samples)).tolist()
    pcg = (0.5 * np.sin(2 * np.pi * 40 * time) + 0.05 * rng.normal(size=n_samples)).tolist()
    return {"ecg": ecg, "pcg": pcg, "fs": fs}


def test_health_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    for field in ["status", "model_loaded", "supports_explain", "model_version", "checkpoint", "device", "preprocessing"]:
        assert field in body, f"/health is missing '{field}'"
    assert isinstance(body["model_loaded"], bool)
    assert body["status"] in ("ok", "degraded")


def test_predict_contract(client):
    response = client.post("/predict", json=make_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    for field in ["label", "label_index", "probability_abnormal", "confidence", "model_version"]:
        assert field in body, f"/predict is missing '{field}'"

    assert body["label"] in ("normal", "abnormal")
    assert body["label_index"] in (0, 1)
    assert 0.0 <= body["probability_abnormal"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    # The two must agree, or the reported confidence is for the other class.
    expected_index = 1 if body["probability_abnormal"] >= 0.5 else 0
    assert body["label_index"] == expected_index
    assert body["confidence"] >= 0.5 - 1e-9


def test_predict_rejects_wrong_length(client):
    response = client.post("/predict", json=make_payload(n_samples=1000))
    assert response.status_code == 422
    assert "6000" in response.text


def test_predict_rejects_wrong_sampling_rate(client):
    response = client.post("/predict", json=make_payload(fs=500))
    assert response.status_code == 422


def test_predict_rejects_non_finite_input(client):
    """A literal NaN cannot travel over JSON, so the reachable route to a
    non-finite value is a float64 that overflows when cast to float32. 1e39 is
    finite in the payload and becomes +inf inside the service."""
    payload = make_payload()
    payload["ecg"][17] = 1e39
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "NaN" in response.text or "Inf" in response.text


def test_predict_rejects_missing_field(client):
    payload = make_payload()
    del payload["pcg"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_explain_returns_a_png(client):
    response = client.post("/explain", json=make_payload())
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n", "response is not a PNG"
    assert len(response.content) > 1000


def test_preprocessing_constants_match_params():
    """The service duplicates the scalogram constants. If params.yaml moves and
    this does not, inference silently sees a different distribution than training."""
    module = load_app_module()
    status = module.check_preprocessing_matches_params()
    assert "MISMATCH" not in status, status
