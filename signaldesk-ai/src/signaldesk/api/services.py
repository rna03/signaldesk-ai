"""Load existing artifacts lazily and run the Phase 3/5/6 analysis."""

import json
from functools import cached_property
from pathlib import Path

from joblib import load

from signaldesk.clustering.discover_issues_semantic import (
    CLUSTERER_PATH,
    METADATA_PATH,
    MODEL_NAME,
    cosine_similarity_value,
    encode_full_conversations,
    load_embedding_model,
)
from signaldesk.ml.predict_domain import ARTIFACT_PATH, predict_text
from signaldesk.monitoring.detect_emerging_issues import DEMO_ARTIFACT_PATH


class ServiceUnavailable(Exception):
    """A required local artifact or pretrained model is unavailable."""


class SignalDeskService:
    def __init__(
        self,
        domain_path: Path = ARTIFACT_PATH,
        clusterer_path: Path = CLUSTERER_PATH,
        metadata_path: Path = METADATA_PATH,
        demo_path: Path = DEMO_ARTIFACT_PATH,
    ):
        self.domain_path = domain_path
        self.clusterer_path = clusterer_path
        self.metadata_path = metadata_path
        self.demo_path = demo_path

    @cached_property
    def domain_model(self):
        if not self.domain_path.is_file():
            raise ServiceUnavailable(
                "Domain classifier artifact is missing. Run python -m signaldesk.ml.train_domain_classifier."
            )
        # joblib may execute code: load only artifacts produced locally by this project.
        return load(self.domain_path)

    @cached_property
    def embedding_model(self):
        try:
            return load_embedding_model()
        except Exception as exc:
            raise ServiceUnavailable(
                "Semantic embedding model is unavailable. Check the model cache or network access."
            ) from exc

    @cached_property
    def clusterer(self):
        if not self.clusterer_path.is_file():
            raise ServiceUnavailable(
                "Semantic clusterer artifact is missing. Run python -m signaldesk.clustering.discover_issues_semantic."
            )
        return load(self.clusterer_path)

    @cached_property
    def metadata(self):
        if not self.metadata_path.is_file():
            raise ServiceUnavailable(
                "Semantic cluster metadata is missing. Run python -m signaldesk.clustering.discover_issues_semantic."
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("embedding_model") != MODEL_NAME
            or metadata.get("cluster_count") != self.clusterer.n_clusters
            or metadata.get("embedding_dimension") != self.clusterer.n_features_in_
            or set(metadata.get("descriptive_terms", {}))
            != {str(index) for index in range(self.clusterer.n_clusters)}
        ):
            raise ServiceUnavailable(
                "Semantic cluster artifacts do not match. Regenerate them together with the Phase 5 script."
            )
        return metadata

    def analyze(self, customer_text: str) -> dict:
        # Check paired artifacts before loading the larger sentence model.
        domain_model = self.domain_model
        clusterer = self.clusterer
        metadata = self.metadata
        domain = predict_text(domain_model, customer_text)
        vector = encode_full_conversations(self.embedding_model, [customer_text], verbose=False)
        cluster_id = int(clusterer.predict(vector)[0])
        similarity = cosine_similarity_value(vector[0], clusterer.cluster_centers_[cluster_id])
        return {
            "domain": domain["label"],
            "domain_confidence": domain["confidence"],
            "semantic_cluster": cluster_id,
            "cluster_similarity": similarity,
            "cluster_descriptive_terms": metadata["descriptive_terms"][str(cluster_id)],
        }

    @cached_property
    def demo_alerts(self) -> dict:
        if not self.demo_path.is_file():
            raise ServiceUnavailable(
                "Synthetic demo artifact is missing. Run python -m signaldesk.monitoring.detect_emerging_issues."
            )
        result = json.loads(self.demo_path.read_text(encoding="utf-8"))
        if result.get("temporal_mode") != "synthetic_demo" or result.get("is_real_time") is not False:
            raise ServiceUnavailable("Synthetic demo artifact is invalid. Regenerate it with the Phase 6 script.")
        return result


_service = SignalDeskService()


def get_service() -> SignalDeskService:
    """FastAPI dependency; replace this in HTTP tests without downloading models."""
    return _service
