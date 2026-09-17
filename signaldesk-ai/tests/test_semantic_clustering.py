"""Model indirmeden semantic kümeleme sayısal yardımcılarını sınar."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from signaldesk.clustering.discover_issues_semantic import (
    aggregate_chunk_embeddings,
    cluster_embeddings,
    cluster_sizes,
    cosine_similarity_value,
    descriptive_top_terms,
    representative_indices,
)


def synthetic_embeddings():
    return np.array([
        [1.0, 0.0], [0.99, 0.1], [0.98, -0.1],
        [0.0, 1.0], [0.1, 0.99], [-0.1, 0.98],
    ])


def test_cosine_similarity():
    assert cosine_similarity_value([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity_value([1, 0], [0, 1]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cosine_similarity_value([0, 0], [1, 0])


def test_cluster_helper_and_sizes():
    matrix = synthetic_embeddings()
    model, labels, stats = cluster_embeddings(matrix, 2)
    assert stats["k"] == 2
    assert sum(cluster_sizes(labels, 2).values()) == 6
    assert sorted(cluster_sizes(labels, 2).values()) == [3, 3]
    assert len(representative_indices(matrix, labels, model.cluster_centers_, 0, 2)) == 2


def test_representatives_use_cosine_closeness():
    matrix = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    centers = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert representative_indices(matrix, labels, centers, 0, 2) == [0, 1]
    assert representative_indices(matrix, labels, centers, 2) == []


def test_descriptive_terms_are_posthoc():
    texts = ["internet modem broken", "internet connection slow", "flight seat change"]
    vectorizer = TfidfVectorizer()
    description_matrix = vectorizer.fit_transform(texts)
    labels = np.array([0, 0, 1])
    terms = descriptive_top_terms(description_matrix, vectorizer, labels, 0, 2)
    assert terms[0] == "internet"
    assert len(terms) == 2


def test_chunk_embeddings_are_weighted_and_normalized():
    chunks = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pooled = aggregate_chunk_embeddings(chunks, [0, 0, 1], [3, 1, 2], 2)
    assert pooled.shape == (2, 2)
    assert np.linalg.norm(pooled, axis=1) == pytest.approx([1.0, 1.0])
    assert pooled[0, 0] > pooled[0, 1]
