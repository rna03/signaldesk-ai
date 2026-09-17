"""The dashboard and API share one FastAPI app without network or model downloads."""

from fastapi.testclient import TestClient

from signaldesk.api.main import app
from signaldesk.api.services import get_service


RESULT = {
    "domain": "telecom",
    "domain_confidence": 0.2,
    "semantic_cluster": 6,
    "cluster_similarity": 0.3,
    "cluster_descriptive_terms": ["internet"],
    "analysis_metadata": {
        "domain_model": "TF-IDF + Logistic Regression",
        "domain_input_mode": "customer_text",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "semantic_cluster_count": 16,
    },
}


def test_dashboard_and_static_assets_are_served():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert "SignalDesk AI" in page.text
        assert "Synthetic Temporal Demo" in page.text
        assert "/static/css/styles.css" in page.text
        assert "/static/js/app.js" in page.text

        for path, marker in (
            ("/static/css/styles.css", ".status-grid"),
            ("/static/js/api.js", "analyzeBatch"),
            ("/static/js/app.js", "submitSingle"),
        ):
            response = client.get(path)
            assert response.status_code == 200
            assert marker in response.text


def test_dashboard_mount_does_not_capture_api_routes():
    class FakeService:
        def ready(self):
            return {"status": "ready"}

        def analyze(self, text):
            assert text == "My modem disconnects."
            return RESULT

        def analyze_many(self, texts):
            assert texts == ["My modem disconnects.", "My bank transfer is late."]
            return [RESULT, {**RESULT, "domain": "banking"}]

        demo_alerts = {"temporal_mode": "synthetic_demo", "is_real_time": False, "alerts": []}

    app.dependency_overrides[get_service] = lambda: FakeService()
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
            assert client.get("/api/v1/info").status_code == 200
            assert client.get("/docs").status_code == 200
            assert client.get("/api/v1/alerts/demo").status_code == 200
            single = client.post("/api/v1/analyze", json={"customer_text": "My modem disconnects."})
            assert single.status_code == 200
            assert single.json()["domain"] == "telecom"
            batch = client.post("/api/v1/analyze/batch", json={"items": [
                {"customer_text": "My modem disconnects."},
                {"customer_text": "My bank transfer is late."},
            ]})
            assert batch.status_code == 200
            assert [item["domain"] for item in batch.json()["results"]] == ["telecom", "banking"]
    finally:
        app.dependency_overrides.clear()
