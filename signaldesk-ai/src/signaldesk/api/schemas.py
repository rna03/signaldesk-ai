"""HTTP request and response shapes; no model logic lives here."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    customer_text: str

    @field_validator("customer_text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("customer_text must not be empty or whitespace")
        return value  # Keep the original transcript unchanged.


class AnalysisMetadata(BaseModel):
    domain_model: str
    domain_input_mode: str
    embedding_model: str
    semantic_cluster_count: int


class AnalyzeResponse(BaseModel):
    domain: str
    domain_confidence: float
    semantic_cluster: int
    cluster_similarity: float
    cluster_descriptive_terms: list[str]
    analysis_metadata: AnalysisMetadata


class BatchAnalyzeRequest(BaseModel):
    items: list[AnalyzeRequest] = Field(min_length=1, max_length=50)


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]


class InfoResponse(BaseModel):
    project: str
    api_version: str
    domain_classifier: str
    semantic_embedding_model: str
    semantic_cluster_count: int
    temporal_mode: str


class DemoAlert(BaseModel):
    timestamp: datetime
    cluster_id: int
    current_count: int
    historical_mean: float
    historical_std: float
    anomaly_score: float
    increase_ratio: float | None
    descriptive_terms: list[str]


class DemoAlertsResponse(BaseModel):
    temporal_mode: str
    is_real_time: bool
    alerts: list[DemoAlert]
