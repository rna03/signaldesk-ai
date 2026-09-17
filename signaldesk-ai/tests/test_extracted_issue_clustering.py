"""Offline checks that semantic clustering reads generated issues, not transcripts."""

import pytest

from signaldesk.clustering.discover_issues import make_vectorizer
from signaldesk.clustering.discover_extracted_issues_semantic import clustering_texts
from signaldesk.issues.extract_issues import load_artifact


def test_clustering_input_is_issue_statement_only():
    records = [
        {"customer_text": "flight flight flight", "issue_statement": "modem disconnects"},
        {"customer_text": "airline airline airline", "issue_statement": "internet slows"},
        {"customer_text": "seat seat seat", "issue_statement": "router fails"},
    ]
    texts = clustering_texts(records)
    assert texts == ["modem disconnects", "internet slows", "router fails"]
    features = make_vectorizer(min_df=1).fit(texts).get_feature_names_out()
    assert "flight" not in features
    assert "modem" in features


def test_empty_issue_and_missing_artifact_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonempty issue_statement"):
        clustering_texts([{"issue_statement": "  ", "customer_text": "valid raw text"}])
    with pytest.raises(FileNotFoundError, match="extract_issues --full"):
        load_artifact(tmp_path / "missing.json")
