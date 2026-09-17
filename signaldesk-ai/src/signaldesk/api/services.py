"""Load existing artifacts lazily and run the Phase 3/5/6 analysis."""

import json
import math
from functools import cached_property
from pathlib import Path

from joblib import load
import numpy as np

from signaldesk.clustering.discover_issues_semantic import (
    CLUSTERER_PATH,
    METADATA_PATH,
    MODEL_NAME,
    cosine_similarity_value,
    encode_full_conversations,
    load_embedding_model,
)
from signaldesk.ml.predict_domain import ARTIFACT_PATH, load_domain_artifact
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
    def domain_artifact(self):
        if not self.domain_path.is_file():
            raise ServiceUnavailable(
                "Domain classifier artifact is missing. Run python -m signaldesk.ml.train_domain_classifier."
            )
        try:
            # joblib may execute code: use only artifacts produced locally by this project.
            return load_domain_artifact(self.domain_path)
        except Exception as exc:
            raise ServiceUnavailable(
                "Domain classifier artifact has the wrong input contract or cannot be read. "
                "Retrain it with python -m signaldesk.ml.train_domain_classifier."
            ) from exc

    @cached_property
    def domain_model(self):
        return self.domain_artifact["pipeline"]

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
        try:
            return load(self.clusterer_path)
        except Exception as exc:
            raise ServiceUnavailable("Semantic clusterer artifact cannot be read. Regenerate it with the Phase 5 script.") from exc

    @cached_property
    def metadata(self):
        if not self.metadata_path.is_file():
            raise ServiceUnavailable(
                "Semantic cluster metadata is missing. Run python -m signaldesk.clustering.discover_issues_semantic."
            )
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            terms = metadata.get("descriptive_terms")
            valid = (
                metadata.get("embedding_model") == MODEL_NAME
                and metadata.get("cluster_count") == self.clusterer.n_clusters
                and metadata.get("embedding_dimension") == self.clusterer.n_features_in_
                and isinstance(terms, dict)
                and set(terms)
                == {str(index) for index in range(self.clusterer.n_clusters)}
                and all(isinstance(value, list) and all(isinstance(term, str) for term in value)
                        for value in terms.values())
            )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ServiceUnavailable("Semantic cluster metadata cannot be read. Regenerate it with the Phase 5 script.") from exc
        if not valid:
            raise ServiceUnavailable(
                "Semantic cluster artifacts do not match. Regenerate them together with the Phase 5 script."
            )
        return metadata

    def ready(self) -> dict[str, str]:
        """Validate local serving artifacts without loading MiniLM or running inference."""
        self.domain_artifact
        self.clusterer
        self.metadata
        return {"status": "ready"}

    def analyze(self, customer_text: str) -> dict:
        return self.analyze_many([customer_text])[0]

    def analyze_many(self, customer_texts: list[str]) -> list[dict]:
        """Run the same feature contract for one or many customer transcripts."""
        if not customer_texts or any(not isinstance(text, str) or not text.strip() for text in customer_texts):
            raise ValueError("At least one nonempty customer_text is required.")
        # Validate local artifacts before loading the larger sentence model.
        domain_artifact = self.domain_artifact
        domain_model = self.domain_model
        clusterer = self.clusterer
        metadata = self.metadata
        labels = domain_model.predict(customer_texts)
        probabilities = np.asarray(domain_model.predict_proba(customer_texts), dtype=float)
        if (
            len(labels) != len(customer_texts)
            or probabilities.ndim != 2
            or probabilities.shape[0] != len(customer_texts)
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError("Domain classifier returned invalid predictions.")

        # Preserve the encoder's float32 dtype: the fitted K-Means artifact expects it.
        vectors = np.asarray(encode_full_conversations(self.embedding_model, customer_texts, verbose=False))
        if (
            vectors.ndim != 2
            or vectors.shape != (len(customer_texts), clusterer.n_features_in_)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Semantic model returned invalid embeddings.")
        cluster_ids = clusterer.predict(vectors)
        if len(cluster_ids) != len(customer_texts):
            raise ValueError("Semantic clusterer returned the wrong number of assignments.")
        analysis_metadata = {
            "domain_model": "TF-IDF + Logistic Regression",
            "domain_input_mode": domain_artifact["input_mode"],
            "embedding_model": MODEL_NAME,
            "semantic_cluster_count": int(clusterer.n_clusters),
        }
        results = []
        for index, cluster in enumerate(cluster_ids):
            cluster_id = int(cluster)
            if not 0 <= cluster_id < clusterer.n_clusters:
                raise ValueError("Semantic clusterer returned an invalid assignment.")
            similarity = cosine_similarity_value(vectors[index], clusterer.cluster_centers_[cluster_id])
            confidence = float(max(probabilities[index]))  # predict_proba score, not calibrated certainty.
            if not math.isfinite(similarity) or not math.isfinite(confidence):
                raise ValueError("Analysis returned a nonfinite score.")
            results.append({
                "domain": str(labels[index]),
                "domain_confidence": confidence,
                "semantic_cluster": cluster_id,
                "cluster_similarity": similarity,
                "cluster_descriptive_terms": metadata["descriptive_terms"][str(cluster_id)],
                "analysis_metadata": analysis_metadata,
            })
        return results

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
