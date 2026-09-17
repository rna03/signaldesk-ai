"""Hazır cümle embeddingleriyle müşteri görüşmelerini keşif amaçlı kümele."""

from collections import Counter

import numpy as np
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from signaldesk.clustering.discover_issues import K_CANDIDATES, make_vectorizer, select_candidate
from signaldesk.data.build_conversations import build_conversations
from signaldesk.data.inspect_dataset import DATASET_NAME


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
PHASE4_K = 16
PHASE4_SILHOUETTE = 0.0451
PHASE4_MIN_SIZE = 19
PHASE4_MAX_SIZE = 127
SANITY_SENTENCES = (
    "My internet keeps disconnecting.",
    "My connection drops every few minutes.",
    "I need to change my flight seat.",
)


def load_embedding_model():
    """Ağ/model yükleme işlemini test edilen sayısal yardımcıların dışında tut."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def encode_texts(model, texts):
    """Hazır modeli yalnızca inference için kullan; ağırlıklarını eğitme."""
    return np.asarray(
        model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )


def aggregate_chunk_embeddings(chunk_vectors, owners, token_counts, conversation_count):
    """Parça vektörlerini token sayısına göre ortala ve tekrar normalize et."""
    vectors = np.asarray(chunk_vectors)
    owners = np.asarray(owners, dtype=int)
    weights = np.asarray(token_counts, dtype=float)
    if len(vectors) != len(owners) or len(vectors) != len(weights):
        raise ValueError("Parça vektörü, sahip ve token sayısı uzunlukları eşleşmeli.")
    pooled = np.zeros((conversation_count, vectors.shape[1]), dtype=float)
    totals = np.zeros(conversation_count, dtype=float)
    np.add.at(pooled, owners, vectors * weights[:, None])
    np.add.at(totals, owners, weights)
    if np.any(totals == 0):
        raise ValueError("En az bir görüşmenin geçerli metin parçası yok.")
    pooled /= totals[:, None]
    norms = np.linalg.norm(pooled, axis=1)
    if np.any(norms == 0):
        raise ValueError("En az bir görüşmenin ortalama vektörü sıfır.")
    return (pooled / norms[:, None]).astype(np.float32)


def encode_full_conversations(model, texts):
    """Model sınırını aşan özgün metni parçalara ayırıp tek görüşme vektörüne topla."""
    special_tokens = model.tokenizer.num_special_tokens_to_add(pair=False)
    chunk_size = model.max_seq_length - special_tokens - 8  # Yeniden tokenleştirmeye küçük pay.
    if chunk_size <= 0:
        raise ValueError("Model token sınırı parçalara ayırmak için çok küçük.")
    chunks, owners, weights = [], [], []
    over_limit = 0
    max_chunks = 0
    for index, text in enumerate(texts):
        token_ids = model.tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_attention_mask=False, verbose=False,
        )["input_ids"]
        if len(token_ids) + special_tokens > model.max_seq_length:
            over_limit += 1
        pieces = [token_ids[start:start + chunk_size] for start in range(0, len(token_ids), chunk_size)]
        max_chunks = max(max_chunks, len(pieces))
        for piece in pieces:
            chunks.append(model.tokenizer.decode(piece, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            owners.append(index)
            weights.append(len(piece))
    print(f"Model sınırını aşan customer_text sayısı: {over_limit}")
    print(f"Toplam metin parçası: {len(chunks)}; görüşme başına en çok parça: {max_chunks}")
    encoded_lengths = model.tokenizer(
        chunks, add_special_tokens=True, truncation=False,
        return_length=True, return_attention_mask=False, verbose=False,
    )["length"]
    if max(encoded_lengths) > model.max_seq_length:
        raise ValueError("En az bir yeniden tokenleştirilmiş parça model sınırını aşıyor.")
    print("Parça embeddingleri üretiliyor; model ağırlıkları eğitilmiyor...")
    chunk_vectors = encode_texts(model, chunks)
    return aggregate_chunk_embeddings(chunk_vectors, owners, weights, len(texts))


def cosine_similarity_value(first, second):
    """İki vektörün kosinüs benzerliğini sıfır vektöre karşı koruyarak hesapla."""
    first, second = np.asarray(first), np.asarray(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Sıfır vektör için cosine similarity tanımsızdır.")
    return float(np.dot(first, second) / denominator)


def cluster_sizes(labels, k):
    counts = Counter(int(label) for label in labels)
    return {cluster_id: counts[cluster_id] for cluster_id in range(k)}


def cluster_embeddings(matrix, k):
    """Normalize dense vektörleri sabit tohumla kümele ve cosine silhouette hesapla."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=200)
    labels = model.fit_predict(matrix)
    sizes = cluster_sizes(labels, k)
    return model, labels, {
        "k": k,
        "silhouette": float(silhouette_score(matrix, labels, metric="cosine")),
        "min_size": min(sizes.values()),
        "max_size": max(sizes.values()),
        "max_share": max(sizes.values()) / len(labels),
    }


def representative_indices(matrix, labels, centers, cluster_id, limit=3):
    """Kendi kümesinde merkeze cosine olarak en yakın gerçek görüşmeleri seç."""
    members = np.flatnonzero(np.asarray(labels) == cluster_id)
    if len(members) == 0:
        return []
    center = centers[cluster_id]
    center_norm = np.linalg.norm(center)
    if center_norm == 0:
        raise ValueError("Sıfır küme merkezi için cosine similarity tanımsızdır.")
    member_vectors = matrix[members]
    scores = (member_vectors @ center) / (np.linalg.norm(member_vectors, axis=1) * center_norm)
    order = np.argsort(-scores, kind="stable")[:limit]
    return [int(members[position]) for position in order]


def descriptive_top_terms(tfidf_matrix, vectorizer, labels, cluster_id, limit=8):
    """Kümeleme SONRASI TF-IDF ile yalnızca yorumlama amaçlı sözcükleri bul."""
    members = np.flatnonzero(np.asarray(labels) == cluster_id)
    if len(members) == 0:
        return []
    weights = np.asarray(tfidf_matrix[members].mean(axis=0)).ravel()
    indices = np.argsort(-weights, kind="stable")[:limit]
    terms = vectorizer.get_feature_names_out()
    return [str(terms[index]) for index in indices if weights[index] > 0]


def load_conversations():
    splitler = load_dataset(DATASET_NAME, streaming=True)
    if not splitler:
        raise RuntimeError("Okunabilir split bulunamadı.")
    conversations = []
    for dataset in splitler.values():
        conversations.extend(build_conversations(dataset.decode(False)))
    return conversations


def main():
    conversations = load_conversations()
    texts = [conversation["customer_text"] for conversation in conversations]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Boş customer_text var; işlem durduruldu.")

    model = load_embedding_model()
    print(f"Conversation count: {len(conversations)}")
    print(f"Embedding model: {MODEL_NAME}")
    print("Embedding girdisi: yalnızca customer_text")
    print(f"Model max_seq_length: {model.max_seq_length}")
    matrix = encode_full_conversations(model, texts)
    print(f"Embedding matrix shape: {matrix.shape}")
    print(f"Embedding dimension: {matrix.shape[1]}")

    sanity = encode_texts(model, list(SANITY_SENTENCES))
    ab = cosine_similarity_value(sanity[0], sanity[1])
    ac = cosine_similarity_value(sanity[0], sanity[2])
    print(f"Sanity A-B cosine: {ab:.4f}")
    print(f"Sanity A-C cosine: {ac:.4f}")
    print(f"A-B > A-C: {ab > ac} (accuracy testi değildir)")

    fitted = []
    print("\nK adayları:")
    for k in K_CANDIDATES:
        cluster_model, labels, stats = cluster_embeddings(matrix, k)
        fitted.append((cluster_model, labels, stats))
        print(
            f"K={k:2d} | cosine silhouette={stats['silhouette']:.4f} | "
            f"min={stats['min_size']} max={stats['max_size']} "
            f"largest_share={stats['max_share']:.1%}"
        )

    selected, best_score = select_candidate([stats for _, _, stats in fitted])
    cluster_model, labels, _ = next(item for item in fitted if item[2]["k"] == selected["k"])
    sizes = cluster_sizes(labels, selected["k"])
    print(f"\nSeçilen K: {selected['k']}")
    print(
        "Gerekçe: en az 5 üyeli, en büyük kümesi toplamın en çok %20'si olan "
        f"adaylardan en iyi silhouette ({best_score:.4f}) değerine 0.005 yakın "
        "en küçük K seçildi. Üst terimler, alan dağılımları ve gerçek örnekler "
        "ayrıca incelenmelidir; otomatik problem etiketi verilmez."
    )
    print("Cluster büyüklükleri:", sizes)

    # Bu ayrı TF-IDF matrisi kümeleme bittikten SONRA yalnızca insan yorumu içindir.
    vectorizer = make_vectorizer()
    description_matrix = vectorizer.fit_transform(texts)
    for cluster_id in range(selected["k"]):
        members = np.flatnonzero(labels == cluster_id)
        domains = Counter(conversations[index]["domain"] for index in members)
        print(f"\nCluster {cluster_id}")
        print(f"Conversation count: {len(members)}")
        print("Descriptive top terms:", ", ".join(
            descriptive_top_terms(description_matrix, vectorizer, labels, cluster_id)
        ))
        print("Domain dağılımı (yalnızca analiz):", dict(domains.most_common()))
        print("Merkeze en yakın 3 gerçek customer_text önizlemesi:")
        for index in representative_indices(matrix, labels, cluster_model.cluster_centers_, cluster_id):
            preview = texts[index].replace("\n", " ")[:180]
            print(f"  [{conversations[index]['domain']}] {preview}...")

    print("\nPhase 4 vs Phase 5:")
    print(
        f"Phase 4 TF-IDF | K={PHASE4_K} | cosine silhouette={PHASE4_SILHOUETTE:.4f} "
        f"| min={PHASE4_MIN_SIZE} max={PHASE4_MAX_SIZE}"
    )
    print(
        f"Phase 5 embedding | K={selected['k']} | cosine silhouette={selected['silhouette']:.4f} "
        f"| min={selected['min_size']} max={selected['max_size']}"
    )
    print("Skorlar farklı temsil uzaylarında hesaplandı; doğrudan kalite veya gerçek issue doğruluğu değildir.")


if __name__ == "__main__":
    main()
