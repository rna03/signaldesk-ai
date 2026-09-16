"""AppTek kayıtlarının dağılımlarını ve dosya adına dayalı eşleşmeleri inceler."""

from collections import Counter, defaultdict
from statistics import mean, median
import re

from datasets import load_dataset

from signaldesk.data.inspect_dataset import DATASET_NAME


FILE_NAME_PATTERN = re.compile(r"^(?P<call_id>.+)_channel(?P<channel>[12])\.wav$")


def extract_call_id(file_name):
    """Beklenen dosya adından görüşme kimliği ve kanal numarası çıkar."""
    match = FILE_NAME_PATTERN.fullmatch(file_name or "")
    if match is None:
        raise ValueError(f"Beklenmeyen file_name biçimi: {file_name!r}")
    return match.group("call_id"), int(match.group("channel"))


def profile_dataset():
    splitler = load_dataset(DATASET_NAME, streaming=True)
    if not splitler:
        raise RuntimeError("Okunabilir split bulunamadı.")

    counts = {name: Counter() for name in ("role", "domain", "accent", "gender")}
    channel_roles = Counter()
    groups = defaultdict(list)
    malformed = []
    speaker_ids = set()
    text_lengths = []
    split_counts = Counter()
    null_text = null_audio = 0
    columns = []

    for split_name, dataset in splitler.items():
        dataset = dataset.decode(False)
        if not columns:
            columns = list(dataset.features or [])
        for row in dataset:
            split_counts[split_name] += 1
            for name, counter in counts.items():
                counter[row.get(name)] += 1
            text = row.get("text")
            if text is None or not str(text).strip():
                null_text += 1
            else:
                text_lengths.append(len(text))
            if row.get("audio") is None:
                null_audio += 1
            if row.get("speaker_id") is not None:
                speaker_ids.add(row["speaker_id"])

            try:
                call_id, channel = extract_call_id(row.get("file_name"))
            except ValueError:
                malformed.append(row.get("file_name"))
                continue
            # Split adı da kimliğe dahil: başka splitlerde aynı ad çakışmaz.
            groups[(split_name, call_id)].append((channel, row.get("role"), row.get("domain"), row.get("accent")))
            channel_roles[(channel, row.get("role"))] += 1

    sizes = Counter(len(rows) for rows in groups.values())
    complete = sum(
        len(rows) == 2
        and {row[0] for row in rows} == {1, 2}
        and {row[1] for row in rows} == {"agent", "customer"}
        and len({row[2] for row in rows}) == 1
        for rows in groups.values()
    )
    anomalies = {
        "record_count_not_2": sum(len(rows) != 2 for rows in groups.values()),
        "duplicate_or_missing_channel": sum(len(rows) == 2 and {row[0] for row in rows} != {1, 2} for rows in groups.values()),
        "role_mismatch": sum(len(rows) == 2 and {row[1] for row in rows} != {"agent", "customer"} for rows in groups.values()),
        "domain_mismatch": sum(len(rows) == 2 and len({row[2] for row in rows}) != 1 for rows in groups.values()),
    }
    return {
        "splits": split_counts,
        "total": sum(split_counts.values()),
        "columns": columns,
        "counts": counts,
        "null_text": null_text,
        "null_audio": null_audio,
        "unique_speakers": len(speaker_ids),
        "text_lengths": {
            "min": min(text_lengths) if text_lengths else None,
            "max": max(text_lengths) if text_lengths else None,
            "mean": mean(text_lengths) if text_lengths else None,
            "median": median(text_lengths) if text_lengths else None,
        },
        "unique_calls": len(groups),
        "group_sizes": sizes,
        "exactly_two": sizes[2],
        "complete": complete,
        "broken": len(groups) - complete + len(malformed),
        "malformed": malformed,
        "anomalies": anomalies,
        "channel_roles": channel_roles,
    }


def main():
    try:
        report = profile_dataset()
    except Exception as error:
        print(f"Profil çıkarılamadı: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    print(f"Veri kümesi: {DATASET_NAME}")
    print("Split'ler:", dict(report["splits"]))
    print("Toplam kayıt:", report["total"])
    print("Kolonlar:", ", ".join(report["columns"]))
    for name, count in report["counts"].items():
        print(f"\n{name} dağılımı ({len(count)} değer):")
        for value, number in sorted(count.items(), key=lambda item: str(item[0])):
            print(f"  {value}: {number}")
    print("\nBoş/null text:", report["null_text"])
    print("Boş/null audio:", report["null_audio"])
    print("Benzersiz speaker_id:", report["unique_speakers"])
    print("Text uzunluğu (karakter):", report["text_lengths"])
    print("\nBenzersiz görüşme kimliği:", report["unique_calls"])
    print("Görüşme başına kayıt sayısı dağılımı:", dict(sorted(report["group_sizes"].items())))
    print("Tam olarak 2 kayıtlı görüşme:", report["exactly_two"])
    print("Güvenilir eşleşme (2 kanal, iki rol, aynı domain):", report["complete"])
    print("Bozuk/eksik eşleşme veya bozuk dosya adı:", report["broken"])
    print("Bozuk dosya adları:", report["malformed"][:10], f"(toplam {len(report['malformed'])})")
    print("Anomaliler:", report["anomalies"])
    print("Kanal/rol dağılımı:")
    for (channel, role), number in sorted(report["channel_roles"].items()):
        print(f"  channel{channel} / {role}: {number}")


if __name__ == "__main__":
    main()
