"""Offline extraction tests with a fake tokenizer and generation model."""

import numpy as np
import pytest

from signaldesk.issues.extract_issues import (
    IssueExtractor,
    build_prompt,
    extract_conversations,
    fallback_statement,
    load_artifact,
    normalize_issue,
    prepare_prompt,
    select_focus_text,
    select_domain_sample,
    write_artifact,
)


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))

    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(range(len(text.split()) + int(add_special_tokens)))

    def decode(self, ids, **kwargs):
        return " ".join("word" for _ in ids)

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        width = max(len(self.encode(prompt)) for prompt in prompts)
        return {"input_ids": np.zeros((len(prompts), width), dtype=int)}

    def batch_decode(self, ids, **kwargs):
        return ["Issue: internet connection fails", " \n "][:len(ids)]


class FakeModel:
    config = type("Config", (), {"n_positions": 512})()

    def generate(self, input_ids, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 32
        assert kwargs["num_beams"] == 1
        return [[index] for index in range(len(input_ids))]


def test_prompt_and_minimal_normalization():
    prompt = build_prompt("My modem drops.")
    assert "My modem drops." in prompt
    assert prompt.endswith("Issue:")
    assert "invent details" in prompt
    assert normalize_issue("  Issue:   modem   keeps  dropping \n") == "modem keeps dropping"
    assert normalize_issue(" N/A ") == ""
    assert normalize_issue("(uh)") == ""
    assert normalize_issue("Nope, nope") == ""
    assert normalize_issue("HTL 9-8 three four") == ""
    assert normalize_issue("No, I'm not sure.") == ""
    assert normalize_issue("Is there anything else I can help you with?") == ""
    assert normalize_issue("A one B two C three") == ""


def test_token_limit_is_checked_without_changing_original():
    text = " ".join(f"word{index}" for index in range(100))
    tokenizer = FakeTokenizer()
    prompt, truncated = prepare_prompt(tokenizer, text, max_input_tokens=40)
    assert truncated is True
    assert len(tokenizer.encode(prompt)) <= 40
    assert prompt.endswith("Issue:")
    assert text.endswith("word99")  # Input was not mutated.
    short_prompt, short_truncated = prepare_prompt(tokenizer, "modem fails", max_input_tokens=40)
    assert short_truncated is False
    assert "modem fails" in short_prompt


def test_focus_window_uses_first_request_cue_without_mutating_transcript():
    text = "Hello, my name is Jo. I wanted to change my flight seat. Thanks."
    focused, shifted = select_focus_text(text)
    assert shifted is True
    assert "change my flight seat" in focused
    assert text.startswith("Hello")
    assert select_focus_text("No explicit cue here.") == ("No explicit cue here.", False)
    assert fallback_statement("Hello Jo. I'm looking for a new electricity service. Yes, that's right.") == (
        "I'm looking for a new electricity service."
    )


def test_fake_model_output_and_explicit_fallback():
    extractor = IssueExtractor(tokenizer=FakeTokenizer(), model=FakeModel())
    conversations = [
        {"conversation_id": "call1", "domain": "telecom", "customer_text": "My internet goes out."},
        {"conversation_id": "call2", "domain": "retail", "customer_text": "The jacket does not fit."},
    ]
    records = extract_conversations(conversations, extractor)
    assert records[0]["issue_statement"] == "internet connection fails"
    assert records[0]["issue_source"] == "flan_t5"
    assert records[1]["issue_statement"] == "The jacket does not fit."
    assert records[1]["issue_source"] == "fallback"
    assert records[1]["customer_text"] == conversations[1]["customer_text"]


def test_sample_is_deterministic_and_uses_distinct_domains():
    conversations = [
        {"conversation_id": f"{domain}-{index}", "domain": domain, "customer_text": "text"}
        for domain in ("telecom", "retail", "banking", "travel", "energy", "insurance", "finance", "food", "health", "education", "technology")
        for index in range(3)
    ]
    first = select_domain_sample(conversations)
    second = select_domain_sample(list(reversed(conversations)))
    assert [row["conversation_id"] for row in first] == [row["conversation_id"] for row in second]
    assert len({row["domain"] for row in first}) == 10


def test_artifact_round_trip_and_invalid_artifact(tmp_path):
    records = [{
        "conversation_id": "call1", "domain": "telecom", "customer_text": "My modem fails.",
        "issue_statement": "modem fails", "issue_source": "flan_t5", "input_truncated": False,
    }]
    path = tmp_path / "extracted_issues.json"
    write_artifact(records, path)
    assert load_artifact(path) == records
    path.write_text('{"conversation_count": 1, "records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_artifact(path)
