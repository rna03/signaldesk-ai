"""Görüşme metninden domain tahmini için ilk klasik ML baseline'ını eğitir."""

from collections import Counter
from pathlib import Path

from datasets import load_dataset
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from signaldesk.data.build_conversations import build_conversations
from signaldesk.data.inspect_dataset import DATASET_NAME
from signaldesk.ml.predict_domain import predict_text


ARTIFACT_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "domain_classifier.joblib"
RANDOM_STATE = 42


def prepare_examples(conversations):
    """Yalnızca iki özgün transcripti feature, domain'i label olarak hazırla."""
    texts, labels, ids = [], [], []
    for conversation in conversations:
        customer = conversation["customer_text"]
        agent = conversation["agent_text"]
        label = conversation["domain"]
        if not customer or not agent or not label:
            raise ValueError(f"Eksik metin veya domain: {conversation['conversation_id']}")
        texts.append(customer + " " + agent)
        labels.append(label)
        ids.append(conversation["conversation_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Tekrarlanan conversation_id var; train/test sızıntısı riski.")
    return texts, labels, ids


def split_examples(texts, labels, ids, test_size=0.2):
    """Görüşmeleri sabit tohumla ve domain oranlarını koruyarak ayır."""
    train_indices, test_indices = train_test_split(
        range(len(texts)), test_size=test_size, random_state=RANDOM_STATE, stratify=labels
    )
    if {ids[index] for index in train_indices} & {ids[index] for index in test_indices}:
        raise ValueError("Aynı görüşme iki veri bölümünde bulundu.")
    return train_indices, test_indices


def make_pipeline():
    """TF-IDF yalnızca fit sırasında verilen train metinlerinden öğrenilir."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def load_conversations():
    splitler = load_dataset(DATASET_NAME, streaming=True)
    if not splitler:
        raise RuntimeError("Okunabilir split bulunamadı.")
    conversations = []
    for split_name, dataset in splitler.items():
        built = build_conversations(dataset.decode(False))
        if len(splitler) > 1:
            for conversation in built:
                conversation["conversation_id"] = f"{split_name}:{conversation['conversation_id']}"
        conversations.extend(built)
    return conversations


def main():
    conversations = load_conversations()
    texts, labels, ids = prepare_examples(conversations)
    train_indices, test_indices = split_examples(texts, labels, ids)
    x_train = [texts[index] for index in train_indices]
    y_train = [labels[index] for index in train_indices]
    x_test = [texts[index] for index in test_indices]
    y_test = [labels[index] for index in test_indices]

    print(f"Toplam conversation: {len(conversations)}")
    print(f"Train conversation: {len(train_indices)}")
    print(f"Test conversation: {len(test_indices)}")

    model = make_pipeline()
    model.fit(x_train, y_train)  # TF-IDF sözlüğü ve sınıflandırıcı yalnızca train ile öğrenir.
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))

    classes = sorted(set(labels))
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    print("Confusion matrix (satır=gerçek, sütun=tahmin):")
    print("Sıra:", ", ".join(classes))
    for label, row in zip(classes, matrix):
        print(f"  {label:16} " + " ".join(f"{value:3d}" for value in row))
    mistakes = Counter((actual, predicted) for actual, predicted in zip(y_test, predictions) if actual != predicted)
    print("\nEn sık karıştırılan ilk 10 yönlü domain çifti:")
    for (actual, predicted), count in mistakes.most_common(10):
        print(f"  {actual} -> {predicted}: {count}")

    print("\nİlk 10 gerçek test örneği:")
    for index in range(min(10, len(x_test))):
        result = predict_text(model, x_test[index])
        print_prediction(y_test[index], result, x_test[index])
    print("\nYanlış tahminlerden örnekler:")
    for index in [i for i, (actual, predicted) in enumerate(zip(y_test, predictions)) if actual != predicted][:5]:
        print_prediction(y_test[index], predict_text(model, x_test[index]), x_test[index])

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, ARTIFACT_PATH)
    print(f"\nModel artifact: {ARTIFACT_PATH}")


def print_prediction(actual, result, text):
    preview = text.replace("\n", " ")[:160]
    print(f"Gerçek: {actual} | Tahmin: {result['label']} | Confidence: {result['confidence']:.2f}")
    print(f"Text preview: {preview}...")


if __name__ == "__main__":
    main()
