"""Phase 3 veri hazırlama ve tahmin yardımcılarını ağ olmadan sınar."""

import pytest

from joblib import dump

from signaldesk.ml.predict_domain import INPUT_MODE, load_domain_artifact, predict_text
from signaldesk.ml.train_domain_classifier import make_pipeline, prepare_examples, split_examples


def conversation(number, domain):
    return {
        "conversation_id": f"call_{number}",
        "domain": domain,
        "customer_text": "(Um) Customer original",
        "agent_text": "(uh) Agent original",
        "customer_speaker_id": "leak_customer",
        "agent_speaker_id": "leak_agent",
    }


def test_prepare_uses_only_original_text_and_domain():
    texts, labels, ids = prepare_examples([conversation(1, "banking")])
    assert texts == ["(Um) Customer original"]
    assert labels == ["banking"]
    assert ids == ["call_1"]
    assert "leak_customer" not in texts[0]
    assert "Agent original" not in texts[0]


def test_prepare_requires_customer_but_not_agent_text():
    record = conversation(1, "banking")
    del record["agent_text"]
    assert prepare_examples([record])[0] == ["(Um) Customer original"]
    record["customer_text"] = "   "
    with pytest.raises(ValueError, match="customer_text"):
        prepare_examples([record])


def test_split_is_deterministic_and_conversations_do_not_overlap():
    conversations = [conversation(i, "banking" if i < 4 else "retail") for i in range(8)]
    texts, labels, ids = prepare_examples(conversations)
    train, test = split_examples(texts, labels, ids, test_size=0.5)
    assert (train, test) == split_examples(texts, labels, ids, test_size=0.5)
    assert {ids[i] for i in train}.isdisjoint({ids[i] for i in test})
    assert sorted(labels[i] for i in train) == ["banking", "banking", "retail", "retail"]
    assert list(make_pipeline().named_steps) == ["tfidf", "classifier"]


def test_duplicate_conversation_id_is_rejected():
    with pytest.raises(ValueError):
        prepare_examples([conversation(1, "banking"), conversation(1, "retail")])


class FakeModel:
    def predict(self, texts):
        return ["telecom"]

    def predict_proba(self, texts):
        return [[0.2, 0.8]]


def test_prediction_output_shape():
    assert predict_text(FakeModel(), "Phone problem") == {"label": "telecom", "confidence": 0.8}
    with pytest.raises(ValueError):
        predict_text(FakeModel(), " ")


def test_artifact_requires_explicit_customer_input_mode(tmp_path):
    path = tmp_path / "domain_classifier.joblib"
    pipeline = make_pipeline()
    dump({"pipeline": pipeline, "input_mode": INPUT_MODE}, path)
    assert load_domain_artifact(path)["pipeline"].steps[0][0] == "tfidf"
    for legacy_artifact in (pipeline, {"pipeline": pipeline, "input_mode": "customer_plus_agent"}):
        dump(legacy_artifact, path)
        with pytest.raises(ValueError, match="customer_text"):
            load_domain_artifact(path)
