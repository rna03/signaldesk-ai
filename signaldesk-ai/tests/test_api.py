"""API contract tests; no Hugging Face downloads or saved models needed."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signaldesk.api.main import app
from signaldesk.api.services import SignalDeskService, get_service


SAMPLE = "My internet keeps disconnecting every few minutes."
ANALYSIS = {
    "domain": "telecom",
    "domain_confidence": 0.72,
    "semantic_cluster": 6,
    "cluster_similarity": 0.81,
    "cluster_descriptive_terms": ["internet", "modem"],
}
DEMO = {
    "temporal_mode": "synthetic_demo",
    "is_real_time": False,
    "alerts": [{
        "timestamp": "2026-01-02T22:00:00+00:00",
        "cluster_id": 5,
        "current_count": 16,
        "historical_mean": 0.667,
        "historical_std": 0.943,
        "anomaly_score": 15.333,
        "increase_ratio": 24.0,
        "descriptive_terms": ["jacket", "return"],
    }],
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_does_not_load_service(client):
    app.dependency_overrides[get_service] = lambda: pytest.fail("service should not load")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_and_docs(client):
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    assert response.json()["temporal_mode"] == "synthetic_demo"
    assert response.json()["semantic_cluster_count"] == 16
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_text_is_rejected(client, text):
    response = client.post("/api/v1/analyze", json={"customer_text": text})
    assert response.status_code == 422


def test_analyze_uses_injected_service(client):
    class FakeService:
        def analyze(self, text):
            assert text == SAMPLE
            return ANALYSIS

    app.dependency_overrides[get_service] = lambda: FakeService()
    response = client.post("/api/v1/analyze", json={"customer_text": SAMPLE})
    assert response.status_code == 200
    assert response.json() == ANALYSIS


def test_missing_artifact_is_controlled_503(client, tmp_path):
    service = SignalDeskService(domain_path=tmp_path / "missing.joblib")
    app.dependency_overrides[get_service] = lambda: service
    response = client.post("/api/v1/analyze", json={"customer_text": SAMPLE})
    assert response.status_code == 503
    assert "Domain classifier artifact is missing" in response.json()["detail"]
    assert str(tmp_path) not in response.text


def test_demo_response_schema(client):
    class FakeService:
        demo_alerts = DEMO

    app.dependency_overrides[get_service] = lambda: FakeService()
    response = client.get("/api/v1/alerts/demo")
    assert response.status_code == 200
    assert response.json()["temporal_mode"] == "synthetic_demo"
    assert response.json()["is_real_time"] is False
    assert set(response.json()["alerts"][0]) == {
        "timestamp", "cluster_id", "current_count", "historical_mean",
        "historical_std", "anomaly_score", "increase_ratio", "descriptive_terms",
    }


def test_unexpected_error_hides_details(client):
    class BrokenService:
        def analyze(self, text):
            raise RuntimeError("private/path/should/not/leak")

    app.dependency_overrides[get_service] = lambda: BrokenService()
    response = client.post("/api/v1/analyze", json={"customer_text": SAMPLE})
    assert response.status_code == 500
    assert "private/path" not in response.text


def test_service_assigns_fitted_cluster_without_refitting(monkeypatch):
    class FakeDomainModel:
        def predict(self, texts):
            return ["telecom"]

        def predict_proba(self, texts):
            return [[0.28, 0.72]]

    class FakeClusterer:
        cluster_centers_ = np.array([[1.0, 0.0]])

        def predict(self, matrix):
            assert matrix.shape == (1, 2)
            return [0]

    service = SignalDeskService()
    service.__dict__.update({
        "domain_model": FakeDomainModel(),
        "clusterer": FakeClusterer(),
        "metadata": {"descriptive_terms": {"0": ["internet", "modem"]}},
        "embedding_model": object(),
    })
    monkeypatch.setattr(
        "signaldesk.api.services.encode_full_conversations",
        lambda model, texts, verbose: np.array([[1.0, 0.0]]),
    )
    result = service.analyze(SAMPLE)
    assert result == {
        "domain": "telecom",
        "domain_confidence": 0.72,
        "semantic_cluster": 0,
        "cluster_similarity": pytest.approx(1.0),
        "cluster_descriptive_terms": ["internet", "modem"],
    }
