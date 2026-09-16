"""Doğrulanmış AppTek kanallarını görüşme seviyesinde birleştirir."""

import json
from collections import defaultdict

from datasets import load_dataset

from signaldesk.data.inspect_dataset import DATASET_NAME, readable_value
from signaldesk.data.profile_dataset import extract_call_id


def pair_records(rows):
    """Aynı çağrının tam iki kaydını, rollere göre ve metni değiştirmeden eşleştir."""
    if len(rows) != 2:
        raise ValueError(f"Bir görüşme tam 2 kayıt içermeli; gelen: {len(rows)}")
    ids_and_channels = [extract_call_id(row.get("file_name")) for row in rows]
    if ids_and_channels[0][0] != ids_and_channels[1][0]:
        raise ValueError("Dosya adları farklı görüşmelere ait.")
    if {channel for _, channel in ids_and_channels} != {1, 2}:
        raise ValueError("Görüşmede channel1 ve channel2 birlikte bulunmalı.")
    by_role = {row.get("role"): row for row in rows}
    if len(by_role) != 2 or set(by_role) != {"customer", "agent"}:
        raise ValueError("Görüşmede bir customer ve bir agent bulunmalı.")
    if rows[0].get("domain") != rows[1].get("domain"):
        raise ValueError("Eşleşen kayıtların domain değerleri farklı.")

    customer = by_role["customer"]
    agent = by_role["agent"]
    return {
        "conversation_id": ids_and_channels[0][0],
        "domain": customer.get("domain"),
        "customer_text": customer.get("text"),
        "agent_text": agent.get("text"),
        "customer_speaker_id": customer.get("speaker_id"),
        "agent_speaker_id": agent.get("speaker_id"),
    }


def build_conversations(rows):
    """Tüm grupları doğrula; bozuk grup varsa kısmi sonuç döndürme."""
    grouped = defaultdict(list)
    for row in rows:
        call_id, _ = extract_call_id(row.get("file_name"))
        grouped[call_id].append(row)
    return [pair_records(group) for group in grouped.values()]


def main():
    try:
        splitler = load_dataset(DATASET_NAME, streaming=True)
        if not splitler:
            raise RuntimeError("Okunabilir split bulunamadı.")
        conversations = []
        for split_name, dataset in splitler.items():
            built = build_conversations(dataset.decode(False))
            # Birden fazla split olursa aynı dosya adı farklı splitlerde çakışmaz.
            if len(splitler) > 1:
                for conversation in built:
                    conversation["conversation_id"] = f"{split_name}:{conversation['conversation_id']}"
            conversations.extend(built)
    except Exception as error:
        print(f"Eşleştirme durduruldu: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    print(f"Doğrulanıp oluşturulan görüşme sayısı: {len(conversations)}")
    print("İlk 2 görüşme (yalnızca terminal önizlemesindeki metinler kısaltılır):")
    for conversation in conversations[:2]:
        preview = {key: readable_value(value) for key, value in conversation.items()}
        print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
