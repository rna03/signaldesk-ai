"""Gerçek müşteri metni + SENTETİK zamanla erken uyarı mantığını sınar."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from random import Random
from statistics import mean, pstdev

from signaldesk.clustering.discover_issues import make_vectorizer
from signaldesk.clustering.discover_issues_semantic import (
    MODEL_NAME,
    cluster_embeddings,
    descriptive_top_terms,
    encode_full_conversations,
    load_conversations,
    load_embedding_model,
)


SEMANTIC_CLUSTER_COUNT = 16  # Phase 5'te seçilip doğrulanan K.
DEMO_START = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
DEMO_HOURS = 48
BUCKET_HOURS = 1
RANDOM_SEED = 42
HISTORY_WINDOW = 12  # Saatlik, yalnızca mevcut saatten ÖNCEKİ bucket'lar.
MIN_HISTORY = 12
STD_FLOOR = 1.0
ANOMALY_THRESHOLD = 3.0
MIN_EVENT_COUNT = 8
SURGE_HOURS = 2
SURGE_EVENT_LIMIT = 30


def synthetic_events(conversations, labels, start=DEMO_START, hours=DEMO_HOURS, seed=RANDOM_SEED):
    """Yalnızca demo timestamp üret; gerçek kayıt/metin/cluster atamasını koru."""
    if len(conversations) != len(labels):
        raise ValueError("Görüşme ve cluster sayıları eşleşmeli.")
    rng = Random(seed)
    return [
        {
            "conversation_index": index,
            "cluster_id": int(label),
            "timestamp": start + timedelta(hours=rng.randrange(hours), minutes=rng.randrange(60)),
        }
        for index, label in enumerate(labels)
    ]


def choose_surge_cluster(labels, minimum_size=2 * MIN_EVENT_COUNT):
    """Yeterli kaydı olan en büyük kümeyi, eşitlikte küçük ID'yi seç."""
    sizes = Counter(int(label) for label in labels)
    eligible = [(size, cluster_id) for cluster_id, size in sizes.items() if size >= minimum_size]
    if not eligible:
        raise ValueError("Surge enjeksiyonu için yeterli büyüklükte küme yok.")
    return min(eligible, key=lambda item: (-item[0], item[1]))[1]


def inject_synthetic_surge(
    events, cluster_id, start=DEMO_START, hours=DEMO_HOURS,
    surge_hours=SURGE_HOURS, limit=SURGE_EVENT_LIMIT, seed=RANDOM_SEED + 1,
):
    """Mevcut gerçek görüşmeleri son saatlere TAŞI; yeni görüşme üretme."""
    surge_start = start + timedelta(hours=hours - surge_hours)
    eligible_positions = [
        position for position, event in enumerate(events)
        if event["cluster_id"] == cluster_id and event["timestamp"] < surge_start
    ]
    if len(eligible_positions) < limit:
        raise ValueError("Seçilen kümede taşınacak yeterli önceki olay yok.")
    rng = Random(seed)
    chosen = rng.sample(eligible_positions, limit)
    result = [event.copy() for event in events]
    for number, position in enumerate(chosen):
        hour_offset = hours - surge_hours + (number % surge_hours)
        result[position]["timestamp"] = start + timedelta(
            hours=hour_offset, minutes=rng.randrange(60)
        )
    return result, chosen


def aggregate_hourly(events, start, hours, cluster_count):
    """Boş saatleri sıfırla doldurarak her kümenin saatlik sayısını üret."""
    counts = {cluster_id: [0] * hours for cluster_id in range(cluster_count)}
    for event in events:
        cluster_id = event["cluster_id"]
        offset = event["timestamp"] - start
        hour = int(offset.total_seconds() // 3600)
        if cluster_id not in counts or not 0 <= hour < hours:
            raise ValueError("Olay kümesi veya demo zaman aralığı geçersiz.")
        counts[cluster_id][hour] += 1
    return counts


def score_bucket(series, index, window=HISTORY_WINDOW, min_history=MIN_HISTORY, std_floor=STD_FLOOR):
    """Mevcut count'u baseline'a katmadan yalnızca geçmiş saatleri kullan."""
    history = series[max(0, index - window):index]
    if len(history) < min_history:
        return None
    historical_mean = mean(history)
    historical_std = pstdev(history)
    current = series[index]
    return {
        "current_count": current,
        "historical_mean": historical_mean,
        "historical_std": historical_std,
        "anomaly_score": (current - historical_mean) / max(historical_std, std_floor),
        "increase_ratio": current / historical_mean if historical_mean > 0 else None,
    }


def detect_alerts(
    hourly_counts, start, threshold=ANOMALY_THRESHOLD,
    minimum_count=MIN_EVENT_COUNT, window=HISTORY_WINDOW,
    min_history=MIN_HISTORY, std_floor=STD_FLOOR,
):
    """Üç koşulu birlikte uygula: geçmiş, asgari hacim, z-score."""
    alerts = []
    for cluster_id, series in sorted(hourly_counts.items()):
        for index in range(len(series)):
            scored = score_bucket(series, index, window, min_history, std_floor)
            if scored is None or scored["current_count"] < minimum_count:
                continue
            if scored["anomaly_score"] >= threshold:
                alerts.append({
                    "timestamp": start + timedelta(hours=index),
                    "cluster_id": cluster_id,
                    **scored,
                })
    return sorted(alerts, key=lambda alert: (alert["timestamp"], alert["cluster_id"]))


def main():
    print("TEMPORAL DATA MODE: SYNTHETIC DEMO", flush=True)
    conversations = load_conversations()
    texts = [conversation["customer_text"] for conversation in conversations]
    model = load_embedding_model()
    embeddings = encode_full_conversations(model, texts)
    _, labels, _ = cluster_embeddings(embeddings, SEMANTIC_CLUSTER_COUNT)

    events = synthetic_events(conversations, labels)
    original_ids = [event["conversation_index"] for event in events]
    target = choose_surge_cluster(labels)
    target_size = sum(label == target for label in labels)
    surge_count = min(SURGE_EVENT_LIMIT, int(target_size * 0.4))
    if surge_count < 2 * MIN_EVENT_COUNT:
        raise ValueError("Seçilen küme anlamlı iki saatlik surge için yeterli değil.")
    injected_events, moved_positions = inject_synthetic_surge(events, target, limit=surge_count)
    if [event["conversation_index"] for event in injected_events] != original_ids:
        raise RuntimeError("Surge enjeksiyonu görüşme kimliklerini değiştirdi.")

    control_counts = aggregate_hourly(events, DEMO_START, DEMO_HOURS, SEMANTIC_CLUSTER_COUNT)
    hourly_counts = aggregate_hourly(injected_events, DEMO_START, DEMO_HOURS, SEMANTIC_CLUSTER_COUNT)
    control_alerts = detect_alerts(control_counts, DEMO_START)
    alerts = detect_alerts(hourly_counts, DEMO_START)

    # TF-IDF yalnızca SONRADAN açıklama içindir; semantic cluster atamasına girmez.
    vectorizer = make_vectorizer()
    description_matrix = vectorizer.fit_transform(texts)
    terms = descriptive_top_terms(description_matrix, vectorizer, labels, target)

    print(f"Conversation count: {len(conversations)}")
    print(f"Semantic model: {MODEL_NAME}; semantic cluster count: {SEMANTIC_CLUSTER_COUNT}")
    print(f"Synthetic time range: {DEMO_START.isoformat()} to "
          f"{(DEMO_START + timedelta(hours=DEMO_HOURS - 1)).isoformat()} (inclusive buckets)")
    print(f"Bucket size: {BUCKET_HOURS} hour")
    print(f"Baseline history window: {HISTORY_WINDOW} prior hours; minimum history: {MIN_HISTORY}")
    print(f"Alert threshold: z >= {ANOMALY_THRESHOLD}; minimum event count: {MIN_EVENT_COUNT}")
    print(f"Historical std floor: {STD_FLOOR}")
    print(f"Injected surge cluster: {target} (largest eligible cluster, {target_size} real conversations)")
    print(f"Injected surge: {len(moved_positions)} existing events moved into final {SURGE_HOURS} hours")
    print("Injected cluster descriptive terms:", ", ".join(terms))
    print(f"Control (before injection) alerts: {len(control_alerts)}")

    surge_start = DEMO_START + timedelta(hours=DEMO_HOURS - SURGE_HOURS)
    injected_cluster_alerts = [alert for alert in alerts if alert["cluster_id"] == target]
    detected_in_surge = any(alert["timestamp"] >= surge_start for alert in injected_cluster_alerts)
    print("Injected surge detected:", "YES" if detected_in_surge else "NO")
    print(f"Total alerts: {len(alerts)}")
    print(f"Injected cluster alerts: {len(injected_cluster_alerts)}")
    print(f"Other alerts: {len(alerts) - len(injected_cluster_alerts)}")

    for alert in alerts:
        print("\nEARLY WARNING")
        print(f"Time: {alert['timestamp'].isoformat()}")
        print(f"Cluster: {alert['cluster_id']} (unverified semantic group)")
        print(f"Current events: {alert['current_count']}")
        print(f"Historical mean: {alert['historical_mean']:.3f}")
        print(f"Historical std: {alert['historical_std']:.3f} (score floor: {STD_FLOOR})")
        print(f"Anomaly score: {alert['anomaly_score']:.3f}")
        ratio = alert["increase_ratio"]
        print(f"Increase ratio: {ratio:.2f}x" if ratio is not None else "Increase ratio: N/A (mean=0)")
        print("Descriptive terms:", ", ".join(
            descriptive_top_terms(description_matrix, vectorizer, labels, alert["cluster_id"])
        ))
        matching = [
            event for event in injected_events
            if event["cluster_id"] == alert["cluster_id"]
            and event["timestamp"].replace(minute=0, second=0, microsecond=0) == alert["timestamp"]
        ]
        print("Representative real customer_text previews (this bucket, max 3):")
        for event in matching[:3]:
            index = event["conversation_index"]
            print(f"  {texts[index].replace(chr(10), ' ')[:180]}...")


if __name__ == "__main__":
    main()
