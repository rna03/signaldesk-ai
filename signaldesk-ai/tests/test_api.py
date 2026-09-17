"""API contract tests; no Hugging Face downloads or real models needed."""

import math

from joblib import dump
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from signaldesk.api.main import app
from signaldesk.api.services import ServiceUnavailable, SignalDeskService, get_service


SAMPLE = "My internet keeps disconnecting every few minutes."
ANALYSIS = {
    "domain": "telecom",
    "domain_confidence": 0.72,
    "semantic_cluster": 6,
    "cluster_similarity": 0.81,
    "cluster_descriptive_terms": ["internet", "modem"],
    "analysis_metadata": {
        "domain_model": "TF-IDF + Logistic Regression",
        "domain_input_mode": "customer_text",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "semantic_cluster_count": 16,
    },
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


def test_ready_uses_service_without_embedding_inference(client, monkeypatch):
    service = SignalDeskService()
    service.__dict__.update({
        "domain_artifact": {"input_mode": "customer_text", "pipeline": object()},
        "clusterer": object(),
        "metadata": {"descriptive_terms": {}},
    })
    monkeypatch.setattr(
        "signaldesk.api.services.load_embedding_model",
        lambda: pytest.fail("readiness must not load MiniLM"),
    )
    app.dependency_overrides[get_service] = lambda: service
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_missing_artifact_is_controlled_503(client, tmp_path):
    service = SignalDeskService(domain_path=tmp_path / "missing.joblib")
    app.dependency_overrides[get_service] = lambda: service
    response = client.get("/ready")
    assert response.status_code == 503
    assert "Domain classifier artifact is missing" in response.json()["detail"]
    assert str(tmp_path) not in response.text


def test_ready_rejects_corrupt_clusterer_as_503(client, tmp_path):
    clusterer_path = tmp_path / "clusterer.joblib"
    dump({"not": "a fitted clusterer"}, clusterer_path)
    service = SignalDeskService(clusterer_path=clusterer_path)
    service.__dict__["domain_artifact"] = {"input_mode": "customer_text", "pipeline": object()}
    app.dependency_overrides[get_service] = lambda: service
    response = client.get("/ready")
    assert response.status_code == 503
    assert "clusterer artifact cannot be read" in response.json()["detail"]
    assert str(tmp_path) not in response.text


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
    assert response.json()["analysis_metadata"]["domain_input_mode"] == "customer_text"


def test_missing_artifact_is_controlled_503(client, tmp_path):
    service = SignalDeskService(domain_path=tmp_path / "missing.joblib")
    app.dependency_overrides[get_service] = lambda: service
    response = client.post("/api/v1/analyze", json={"customer_text": SAMPLE})
    assert response.status_code == 503
    assert "Domain classifier artifact is missing" in response.json()["detail"]
    assert str(tmp_path) not in response.text


def test_legacy_domain_artifact_contract_mismatch_is_503(client, tmp_path):
    path = tmp_path / "legacy.joblib"
    dump({"input_mode": "customer_text+agent_text", "pipeline": Pipeline([])}, path)
    service = SignalDeskService(domain_path=path)
    app.dependency_overrides[get_service] = lambda: service
    for endpoint, payload in [
        ("/ready", None),
        ("/api/v1/analyze", {"customer_text": SAMPLE}),
    ]:
        response = client.get(endpoint) if payload is None else client.post(endpoint, json=payload)
        assert response.status_code == 503
        assert "input contract" in response.json()["detail"]
        assert str(tmp_path) not in response.text


def test_batch_two_valid_inputs_preserves_order_and_schema(client):
    banking = {**ANALYSIS, "domain": "banking", "semantic_cluster": 2}

    class FakeService:
        def analyze_many(self, texts):
            assert texts == [SAMPLE, "I need help with a bank transfer."]
            return [ANALYSIS, banking]

    app.dependency_overrides[get_service] = lambda: FakeService()
    response = client.post("/api/v1/analyze/batch", json={"items": [
        {"customer_text": SAMPLE},
        {"customer_text": "I need help with a bank transfer."},
    ]})
    assert response.status_code == 200
    assert response.json() == {"results": [ANALYSIS, banking]}


@pytest.mark.parametrize("items", [
    [],
    [{"customer_text": SAMPLE}] * 51,
    [{"customer_text": SAMPLE}, {"customer_text": " \t "}],
])
def test_invalid_batch_is_rejected_before_analysis(client, items):
    # FastAPI may resolve the lightweight dependency before request validation.
    # The invalid body must still be rejected before the service runs analysis.
    app.dependency_overrides[get_service] = lambda: object()
    response = client.post("/api/v1/analyze/batch", json={"items": items})
    assert response.status_code == 422


@pytest.mark.parametrize("error, status_code", [
    (ServiceUnavailable("Required serving artifact is unavailable."), 503),
    (RuntimeError("private/path/should/not/leak"), 500),
])
def test_batch_failure_has_controlled_response(client, error, status_code):
    class BrokenService:
        def analyze_many(self, texts):
            raise error

    app.dependency_overrides[get_service] = lambda: BrokenService()
    response = client.post("/api/v1/analyze/batch", json={"items": [{"customer_text": SAMPLE}]})
    assert response.status_code == status_code
    assert "private/path" not in response.text


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


def test_corrupt_demo_artifact_is_controlled_503(client, tmp_path):
    demo_path = tmp_path / "early_warning_demo.json"
    demo_path.write_text("{broken json", encoding="utf-8")
    app.dependency_overrides[get_service] = lambda: SignalDeskService(demo_path=demo_path)
    response = client.get("/api/v1/alerts/demo")
    assert response.status_code == 503
    assert "Synthetic demo artifact cannot be read" in response.json()["detail"]
    assert str(tmp_path) not in response.text


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
        n_clusters = 1
        n_features_in_ = 2

        def predict(self, matrix):
            assert matrix.shape == (1, 2)
            return [0]

    service = SignalDeskService()
    service.__dict__.update({
        "domain_artifact": {"input_mode": "customer_text", "pipeline": FakeDomainModel()},
        "clusterer": FakeClusterer(),
        "metadata": {"descriptive_terms": {"0": ["internet", "modem"]}},
        "embedding_model": object(),
    })
    monkeypatch.setattr(
        "signaldesk.api.services.encode_full_conversations",
        lambda model, texts, verbose: np.array([[1.0, 0.0]]),
    )
    result = service.analyze(SAMPLE)
    assert result["domain"] == "telecom"
    assert result["domain_confidence"] == pytest.approx(0.72)
    assert result["semantic_cluster"] == 0
    assert math.isfinite(result["cluster_similarity"])
    assert result["cluster_similarity"] == pytest.approx(1.0)
    assert result["cluster_descriptive_terms"] == ["internet", "modem"]
    assert result["analysis_metadata"]["domain_input_mode"] == "customer_text"
    assert result["analysis_metadata"]["semantic_cluster_count"] == 1


def test_batch_service_encodes_once_and_uses_assigned_centroids(monkeypatch):
    texts = [SAMPLE, "I need help with a bank transfer."]

    class FakeDomainModel:
        def predict(self, values):
            assert values == texts
            return ["telecom", "banking"]

        def predict_proba(self, values):
            assert values == texts
            return [[0.72, 0.28], [0.31, 0.69]]

    class FakeClusterer:
        cluster_centers_ = np.array([[1.0, 0.0], [0.0, 1.0]])
        n_clusters = 2
        n_features_in_ = 2

        def predict(self, matrix):
            assert matrix.shape == (2, 2)
            assert matrix.dtype == np.float32  # Match the fitted K-Means input dtype.
            return [0, 1]

    calls = []

    def fake_encode(model, values, verbose):
        calls.append(list(values))
        assert verbose is False
        return np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)

    service = SignalDeskService()
    service.__dict__.update({
        "domain_artifact": {"input_mode": "customer_text", "pipeline": FakeDomainModel()},
        "clusterer": FakeClusterer(),
        "metadata": {"descriptive_terms": {"0": ["internet"], "1": ["account"]}},
        "embedding_model": object(),
    })
    monkeypatch.setattr("signaldesk.api.services.encode_full_conversations", fake_encode)

    results = service.analyze_many(texts)
    assert calls == [texts]  # The semantic encoder sees one batch, not one call per item.
    assert [result["domain"] for result in results] == ["telecom", "banking"]
    assert [result["semantic_cluster"] for result in results] == [0, 1]
    assert [result["cluster_descriptive_terms"] for result in results] == [["internet"], ["account"]]
    assert all(math.isfinite(result["cluster_similarity"]) for result in results)
    # A wrong centroid would give 0.5/sqrt(1.25), not 1/sqrt(1.25).
    expected_similarity = 1.0 / math.sqrt(1.25)
    assert [result["cluster_similarity"] for result in results] == pytest.approx([
        expected_similarity, expected_similarity,
    ])


def test_nonfinite_embedding_has_safe_500(client, monkeypatch):
    class FakeDomainModel:
        def predict(self, texts):
            return ["telecom"]

        def predict_proba(self, texts):
            return [[0.72, 0.28]]

    class FakeClusterer:
        cluster_centers_ = np.array([[1.0, 0.0]])
        n_clusters = 1
        n_features_in_ = 2

        def predict(self, matrix):
            pytest.fail("nonfinite embedding must be rejected before assignment")

    service = SignalDeskService()
    service.__dict__.update({
        "domain_artifact": {"input_mode": "customer_text", "pipeline": FakeDomainModel()},
        "clusterer": FakeClusterer(),
        "metadata": {"descriptive_terms": {"0": ["internet"]}},
        "embedding_model": object(),
    })
    monkeypatch.setattr(
        "signaldesk.api.services.encode_full_conversations",
        lambda model, texts, verbose: np.array([[float("nan"), 0.0]]),
    )
    with pytest.raises(ValueError, match="invalid embeddings"):
        service.analyze(SAMPLE)

    app.dependency_overrides[get_service] = lambda: service
    response = client.post("/api/v1/analyze", json={"customer_text": SAMPLE})
    assert response.status_code == 500
    assert "nan" not in response.text.lower()
