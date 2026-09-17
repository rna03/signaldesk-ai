"""Gerçek dataset ve model indirmeden saatlik erken uyarı mantığını sınar."""

from datetime import datetime, timedelta, timezone

from signaldesk.monitoring.detect_emerging_issues import (
    aggregate_hourly,
    detect_alerts,
    inject_synthetic_surge,
    score_bucket,
    synthetic_events,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_hourly_aggregation_includes_zero_buckets():
    events = [
        {"timestamp": START + timedelta(minutes=3), "cluster_id": 0},
        {"timestamp": START + timedelta(minutes=40), "cluster_id": 0},
        {"timestamp": START + timedelta(hours=2, minutes=9), "cluster_id": 1},
    ]
    assert aggregate_hourly(events, START, 3, 2) == {0: [2, 0, 0], 1: [0, 0, 1]}


def test_current_bucket_is_excluded_from_baseline_and_std_zero_is_safe():
    series = [2] * 12 + [15]
    result = score_bucket(series, 12)
    assert result["historical_mean"] == 2
    assert result["historical_std"] == 0
    assert result["anomaly_score"] == 13  # std floor = 1
    assert result["increase_ratio"] == 7.5


def test_minimum_history_and_zero_mean():
    assert score_bucket([0] * 11 + [10], 11) is None
    result = score_bucket([0] * 12 + [8], 12)
    assert result["anomaly_score"] == 8
    assert result["increase_ratio"] is None


def test_minimum_event_count_and_normal_series_do_not_alert():
    counts = {0: [2] * 48, 1: [0] * 20 + [7] + [0] * 27}
    assert detect_alerts(counts, START) == []


def test_controlled_spike_alerts():
    counts = {0: [2] * 20 + [15] + [2] * 27}
    alerts = detect_alerts(counts, START)
    assert len(alerts) == 1
    assert alerts[0]["timestamp"] == START + timedelta(hours=20)
    assert alerts[0]["current_count"] == 15
    assert alerts[0]["historical_mean"] == 2
    assert alerts[0]["anomaly_score"] == 13


def test_synthetic_timestamps_deterministic_and_injection_moves_only():
    conversations = [{"conversation_id": f"real_{index}", "customer_text": "original"} for index in range(20)]
    labels = [0] * 20
    first = synthetic_events(conversations, labels, start=START, hours=48, seed=42)
    second = synthetic_events(conversations, labels, start=START, hours=48, seed=42)
    assert first == second
    assert all(START <= event["timestamp"] < START + timedelta(hours=48) for event in first)
    moved, positions = inject_synthetic_surge(first, 0, start=START, hours=48, limit=8, seed=43)
    assert len(moved) == len(first)
    assert len(positions) == 8
    assert [event["conversation_index"] for event in moved] == list(range(20))
    assert sum(event["timestamp"] != first[index]["timestamp"] for index, event in enumerate(moved)) == 8
    assert all(moved[position]["timestamp"] >= START + timedelta(hours=46) for position in positions)
