"""Compare semantic clusters of generated issue statements with Phase 5."""

from collections import Counter

import numpy as np

from signaldesk.clustering.discover_issues import K_CANDIDATES, make_vectorizer, select_candidate
from signaldesk.clustering.discover_issues_semantic import (
    MODEL_NAME,
    cluster_embeddings,
    cluster_sizes,
    descriptive_top_terms,
    encode_full_conversations,
    load_embedding_model,
    representative_indices,
)
from signaldesk.issues.extract_issues import ARTIFACT_PATH, load_artifact


def clustering_texts(records: list[dict]) -> list[str]:
    """Use only extracted issue_statement, never the raw customer transcript."""
    texts = [row["issue_statement"] for row in records]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every record needs a nonempty issue_statement")
    return texts


def main():
    records = load_artifact()
    texts = clustering_texts(records)
    if len(texts) <= max(K_CANDIDATES):
        raise ValueError("Full extracted issue artifact is required for all K candidates")
    print(f"Artifact: {ARTIFACT_PATH}")
    print(f"Conversations: {len(records)}")
    print("Embedding input: issue_statement only; customer_text and domain are excluded")
    print(f"Embedding model: {MODEL_NAME}")
    embedding_model = load_embedding_model()
    matrix = encode_full_conversations(embedding_model, texts)
    print(f"Embedding matrix shape: {matrix.shape}")

    fitted = []
    print("\nK candidates (same settings as Phase 5):")
    for k in K_CANDIDATES:
        model, labels, stats = cluster_embeddings(matrix, k)
        fitted.append((model, labels, stats))
        print(
            f"K={k:2d} | cosine silhouette={stats['silhouette']:.4f} | "
            f"min={stats['min_size']} max={stats['max_size']} "
            f"largest_share={stats['max_share']:.1%}"
        )

    selected, best_score = select_candidate([stats for _, _, stats in fitted])
    model, labels, _ = next(item for item in fitted if item[2]["k"] == selected["k"])
    sizes = cluster_sizes(labels, selected["k"])
    print(f"\nSelected K: {selected['k']}")
    print(
        f"Selection rule: min cluster >= 5, largest share <= 20%, and smallest K "
        f"within 0.005 of best viable silhouette ({best_score:.4f})."
    )
    print(f"Selected silhouette: {selected['silhouette']:.4f}")
    print(f"Cluster size range: {selected['min_size']}–{selected['max_size']}")
    print("Cluster sizes:", sizes)

    # TF-IDF here is post-hoc interpretation; it never determines semantic labels.
    vectorizer = make_vectorizer()
    description_matrix = vectorizer.fit_transform(texts)
    for cluster_id in range(selected["k"]):
        members = np.flatnonzero(labels == cluster_id)
        domains = Counter(records[index]["domain"] for index in members)
        print(f"\nCluster {cluster_id} | size={len(members)}")
        print("Descriptive terms:", ", ".join(
            descriptive_top_terms(description_matrix, vectorizer, labels, cluster_id)
        ))
        print("Dominant domains (analysis only):", dict(domains.most_common(4)))
        print(f"Dominant domain share: {domains.most_common(1)[0][1] / len(members):.1%}")
        print("Representative issue statements (closest to centroid, max 3):")
        for index in representative_indices(matrix, labels, model.cluster_centers_, cluster_id):
            print(f"  [{records[index]['domain']}; {records[index]['issue_source']}] {texts[index]}")

    print("\nPhase 5 raw customer_text reference: K=16, cosine silhouette=0.2070, size=35–75")
    print(
        "Silhouette values compare representations without issue ground truth; "
        "domain is not a verified issue label. Inspect the statements and clusters manually."
    )


if __name__ == "__main__":
    main()
