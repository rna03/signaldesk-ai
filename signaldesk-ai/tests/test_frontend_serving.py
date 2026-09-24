"""The dashboard and API share one FastAPI app without network or model downloads."""

from fastapi.testclient import TestClient

from signaldesk.api.main import FRONTEND_DIR, app
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
        assert "charset=utf-8" in page.headers["content-type"]
        assert "SignalDesk AI" in page.text
        assert '<html lang="tr">' in page.text
        assert "Çağrı Merkezi Akıllı Analiz Sistemi" in page.text
        assert "Müşteri görüşmelerini analiz edin, tekrar eden sorunları keşfedin" in page.text
        assert "Analiz etmek istediğiniz müşteri görüşmesini buraya girin..." in page.text
        assert "Görüşmeyi Analiz Et" in page.text
        assert "Erken Uyarılar" in page.text
        assert "Sentetik Zaman Serisi Demosu" in page.text
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


def test_visible_copy_is_turkish_and_api_contract_is_preserved():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (FRONTEND_DIR / "js" / "app.js").read_text(encoding="utf-8")
    api_js = (FRONTEND_DIR / "js" / "api.js").read_text(encoding="utf-8")

    obsolete_visible_copy = (
        "Single Conversation Analysis",
        "Analyze Conversation",
        "Early Warning Demo",
        "Analysis complete.",
        "No result yet",
        "Synthetic demo alerts are unavailable.",
        "Cannot reach the API server.",
    )
    combined = "\n".join((html, app_js, api_js))
    for phrase in obsolete_visible_copy:
        assert phrase not in combined

    for endpoint in (
        "/health",
        "/ready",
        "/api/v1/info",
        "/api/v1/alerts/demo",
        "/api/v1/analyze",
        "/api/v1/analyze/batch",
    ):
        assert endpoint in api_js
    assert "customer_text" in api_js


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
