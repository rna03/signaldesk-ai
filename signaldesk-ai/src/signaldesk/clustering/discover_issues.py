"""Müşteri transcriptlerini TF-IDF ve K-Means ile keşif amaçlı kümeler."""

from collections import Counter

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

from signaldesk.data.build_conversations import build_conversations
from signaldesk.data.inspect_dataset import DATASET_NAME


K_CANDIDATES = (8, 12, 16, 20, 24, 30)
RANDOM_STATE = 42
# İngilizce konuşmadaki dolgu sözcükleri yalnızca TF-IDF sözlüğünden çıkarılır.
FILLER_WORDS = {
    "um", "uh", "ah", "yeah", "yep", "okay", "oh", "ohh", "hmm", "hm", "mhm", "mm", "er",
    "yes", "right", "know", "lah", "ll", "ve", "don", "wanna", "gonna", "brilliant",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | FILLER_WORDS)


def make_vectorizer(min_df=3):
    """Kaynak transcriptlere dokunmadan sayısal metin özellikleri oluştur."""
    return TfidfVectorizer(
        stop_words=STOP_WORDS,
        min_df=min_df,
        max_df=0.85,
        max_features=12000,
        ngram_range=(1, 2),
        norm="l2",
    )


def cluster_counts(labels, k):
    """Boş kümeler de dahil her cluster_id'nin kayıt sayısını döndür."""
    counts = Counter(int(label) for label in labels)
    return {cluster_id: counts[cluster_id] for cluster_id in range(k)}


def top_terms(vectorizer, model, cluster_id, limit=8):
    """Merkezde TF-IDF ağırlığı en yüksek terimleri göster."""
    terms = vectorizer.get_feature_names_out()
    indices = model.cluster_centers_[cluster_id].argsort()[-limit:][::-1]
    return [str(terms[index]) for index in indices]


def representative_indices(matrix, labels, model, cluster_id, limit=3):
    """Kendi kümesinde merkeze Öklid uzaklığı en az olan görüşmeleri bul."""
    members = [index for index, label in enumerate(labels) if label == cluster_id]
    if not members:
        return []
    distances = model.transform(matrix[members])[:, cluster_id]
    ordered = sorted(zip(distances, members), key=lambda item: (item[0], item[1]))
    return [index for _, index in ordered[:limit]]


def fit_candidate(matrix, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=200)
    labels = model.fit_predict(matrix)
    sizes = cluster_counts(labels, k)
    silhouette = silhouette_score(matrix, labels, metric="cosine")
    return model, labels, {
        "k": k,
        "silhouette": float(silhouette),
        "min_size": min(sizes.values()),
        "max_size": max(sizes.values()),
        "max_share": max(sizes.values()) / len(labels),
    }


def select_candidate(candidate_results):
    """Yakın silhouette değerlerinde daha küçük ve dengeli K'yi tercih et."""
    viable = [
        item for item in candidate_results
        if item["min_size"] >= 5 and item["max_share"] <= 0.20
    ]
    if not viable:
        raise ValueError("Adayların hiçbiri boyut kontrolünden geçmedi; K seçilmedi.")
    best_score = max(item["silhouette"] for item in viable)
    near_best = [item for item in viable if item["silhouette"] >= best_score - 0.005]
    return min(near_best, key=lambda item: item["k"]), best_score


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
    texts = [item["customer_text"] for item in conversations]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Boş customer_text var; kümeleme durduruldu.")
    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    print(f"Toplam conversation: {len(conversations)}")
    print("Clustering text alanı: customer_text")
    print(f"TF-IDF matrix: {matrix.shape[0]} kayıt x {matrix.shape[1]} özellik")
    print("Domain ve diğer metadata model girdisine verilmedi.")

    fitted = []
    for k in K_CANDIDATES:
        model, labels, stats = fit_candidate(matrix, k)
        fitted.append((model, labels, stats))
        print(
            f"K={k:2d} | cosine silhouette={stats['silhouette']:.4f} | "
            f"min={stats['min_size']} max={stats['max_size']} "
            f"largest_share={stats['max_share']:.1%}"
        )
        print("  Küme üst terimleri:")
        for cluster_id in range(k):
            print(f"  {cluster_id}: {', '.join(top_terms(vectorizer, model, cluster_id, 5))}")

    selected, best_score = select_candidate([stats for _, _, stats in fitted])
    model, labels, _ = next(item for item in fitted if item[2]["k"] == selected["k"])
    counts = cluster_counts(labels, selected["k"])
    print(f"\nSeçilen K: {selected['k']}")
    print(
        "Gerekçe: en az 5 üyeli ve en büyük kümesi toplamın en çok %20'si olan "
        f"adaylar içinden en iyi cosine silhouette ({best_score:.4f}) değerine "
        "0.005 yakın en küçük K seçildi. Daha az küme elle incelemeyi kolaylaştırır. "
        "Üst terimler ve gerçek örnekler aşağıda ayrıca görülebilir; skor gerçek issue etiketi değildir."
    )
    print("Cluster büyüklükleri:", counts)

    for cluster_id in range(selected["k"]):
        member_indices = [index for index, label in enumerate(labels) if label == cluster_id]
        domains = Counter(conversations[index]["domain"] for index in member_indices)
        print(f"\nCluster {cluster_id}")
        print(f"Conversation count: {len(member_indices)}")
        print("Top terms:", ", ".join(top_terms(vectorizer, model, cluster_id)))
        print("Domain dağılımı (yalnızca analiz):", dict(domains.most_common()))
        print("Merkeze en yakın 3 gerçek customer_text önizlemesi:")
        for index in representative_indices(matrix, labels, model, cluster_id):
            text = texts[index].replace("\n", " ")
            print(f"  [{conversations[index]['domain']}] {text[:180]}...")


if __name__ == "__main__":
    main()
