"""İnternet kullanmadan görüşme eşleştirmesini sınar."""

import pytest

from signaldesk.data.build_conversations import build_conversations, pair_records
from signaldesk.data.profile_dataset import extract_call_id


def record(channel, role, text, call="audio/en_AU_Agriculture_123", domain="agriculture"):
    return {
        "file_name": f"{call}_channel{channel}.wav",
        "role": role,
        "text": text,
        "domain": domain,
        "speaker_id": f"speaker_{role}",
    }


def test_same_call_id_and_role_based_pairing():
    customer = record(1, "customer", "(Um) My delivery is late.")
    agent = record(2, "agent", "(uh) I can check.")
    assert extract_call_id(customer["file_name"])[0] == extract_call_id(agent["file_name"])[0]
    conversation = build_conversations([customer, agent])[0]
    assert conversation["conversation_id"] == "audio/en_AU_Agriculture_123"
    assert conversation["customer_text"] == "(Um) My delivery is late."
    assert conversation["agent_text"] == "(uh) I can check."
    assert conversation["customer_speaker_id"] == "speaker_customer"


@pytest.mark.parametrize("rows", [
    [record(1, "agent", "hello")],
    [record(1, "agent", "hello"), record(2, "agent", "again")],
    [record(1, "agent", "hello"), record(1, "customer", "hi")],
    [record(1, "agent", "hello"), record(2, "customer", "hi", call="audio/other_456")],
    [record(1, "agent", "hello"), record(2, "customer", "hi", domain="banking")],
])
def test_invalid_pair_is_rejected(rows):
    with pytest.raises(ValueError):
        pair_records(rows)


def test_malformed_file_name_is_rejected():
    with pytest.raises(ValueError):
        extract_call_id("audio/call_without_channel.wav")
