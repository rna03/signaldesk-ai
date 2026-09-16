"""Etiketsiz kümeleme yardımcılarını ağ olmadan sınar."""

import numpy as np
from sklearn.cluster import KMeans

from signaldesk.clustering.discover_issues import (
    cluster_counts,
    make_vectorizer,
    representative_indices,
    select_candidate,
    top_terms,
)


def test_tfidf_and_kmeans_on_small_customer_texts():
    texts = [
        "internet modem connection slow", "internet modem keeps dropping",
        "connection modem internet down", "bank account transfer delayed",
        "bank transfer account pending", "account bank transfer failed",
    ]
    vectorizer = make_vectorizer(min_df=1)
    matrix = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=2, random_state=42, n_init=5).fit(matrix)
    labels = model.labels_
    assert matrix.shape[0] == 6
    assert len(set(labels)) == 2
    assert sum(cluster_counts(labels, 2).values()) == 6
    assert len(top_terms(vectorizer, model, 0, limit=3)) == 3
    assert len(representative_indices(matrix, labels, model, 0, limit=2)) == 2


def test_cluster_counts_includes_empty_id():
    assert cluster_counts(np.array([0, 0, 2]), 3) == {0: 2, 1: 0, 2: 1}


def test_k_selection_uses_score_and_size():
    candidates = [
        {"k": 8, "silhouette": 0.040, "min_size": 10, "max_share": 0.18},
        {"k": 12, "silhouette": 0.044, "min_size": 7, "max_share": 0.19},
        {"k": 16, "silhouette": 0.050, "min_size": 1, "max_share": 0.25},
    ]
    selected, best = select_candidate(candidates)
    assert selected["k"] == 8
    assert best == 0.044
