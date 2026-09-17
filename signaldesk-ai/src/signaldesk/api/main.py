"""Small HTTP routes; reusable analysis stays in services.py."""

import logging

from fastapi import Depends, FastAPI, HTTPException

from signaldesk.api.schemas import AnalyzeRequest, AnalyzeResponse, DemoAlertsResponse, InfoResponse
from signaldesk.api.services import ServiceUnavailable, SignalDeskService, get_service
from signaldesk.clustering.discover_issues_semantic import MODEL_NAME
from signaldesk.monitoring.detect_emerging_issues import SEMANTIC_CLUSTER_COUNT


logger = logging.getLogger(__name__)
app = FastAPI(
    title="SignalDesk AI API",
    description="Customer issue intelligence and early-warning API",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/info", response_model=InfoResponse)
def info() -> dict:
    return {
        "project": "SignalDesk AI",
        "api_version": app.version,
        "domain_classifier": "TF-IDF + Logistic Regression (Phase 3)",
        "semantic_embedding_model": MODEL_NAME,
        "semantic_cluster_count": SEMANTIC_CLUSTER_COUNT,
        "temporal_mode": "synthetic_demo",
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, service: SignalDeskService = Depends(get_service)) -> dict:
    try:
        return service.analyze(request.customer_text)
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed unexpectedly.") from exc


@app.get("/api/v1/alerts/demo", response_model=DemoAlertsResponse)
def alerts_demo(service: SignalDeskService = Depends(get_service)) -> dict:
    try:
        return service.demo_alerts
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Synthetic demo loading failed")
        raise HTTPException(status_code=500, detail="Synthetic demo is unavailable unexpectedly.") from exc
