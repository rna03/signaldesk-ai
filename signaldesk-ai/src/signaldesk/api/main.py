"""Small HTTP routes; reusable analysis stays in services.py."""

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from signaldesk.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    DemoAlertsResponse,
    InfoResponse,
)
from signaldesk.api.services import ServiceUnavailable, SignalDeskService, get_service
from signaldesk.clustering.discover_issues_semantic import MODEL_NAME
from signaldesk.monitoring.detect_emerging_issues import SEMANTIC_CLUSTER_COUNT


logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
app = FastAPI(
    title="SignalDesk AI API",
    description="Customer issue intelligence and early-warning API",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(service: SignalDeskService = Depends(get_service)) -> dict[str, str]:
    try:
        return service.ready()
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=500, detail="Readiness check failed unexpectedly.") from exc


@app.get("/api/v1/info", response_model=InfoResponse)
def info() -> dict:
    return {
        "project": "SignalDesk AI",
        "api_version": app.version,
        "domain_classifier": "TF-IDF + Logistic Regression (customer_text)",
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


@app.post("/api/v1/analyze/batch", response_model=BatchAnalyzeResponse)
def analyze_batch(request: BatchAnalyzeRequest, service: SignalDeskService = Depends(get_service)) -> dict:
    try:
        return {"results": service.analyze_many([item.customer_text for item in request.items])}
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch analysis failed")
        raise HTTPException(status_code=500, detail="Batch analysis failed unexpectedly.") from exc


@app.get("/api/v1/alerts/demo", response_model=DemoAlertsResponse)
def alerts_demo(service: SignalDeskService = Depends(get_service)) -> dict:
    try:
        return service.demo_alerts
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Synthetic demo loading failed")
        raise HTTPException(status_code=500, detail="Synthetic demo is unavailable unexpectedly.") from exc
