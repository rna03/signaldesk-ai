"""Yerel olarak eğitilmiş baseline modeliyle elle metin tahmini yapar."""

import argparse
from pathlib import Path

from joblib import load
from sklearn.pipeline import Pipeline


ARTIFACT_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "domain_classifier.joblib"
EXAMPLE_TEXT = "I cannot log in to my mobile phone account. Please help me reset the password."
INPUT_MODE = "customer_text"


def load_domain_artifact(path=ARTIFACT_PATH):
    """Yalnızca müşteri metniyle eğitilmiş, sözleşmesi belirtilmiş modeli yükle."""
    # joblib dosyaları yalnızca güvenilen yerel kaynaktan yüklenmelidir.
    artifact = load(path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("input_mode") != INPUT_MODE
        or not isinstance(artifact.get("pipeline"), Pipeline)
    ):
        raise ValueError(
            "Domain artifact input_mode='customer_text' sözleşmesini karşılamıyor; "
            "python -m signaldesk.ml.train_domain_classifier ile yeniden eğitin."
        )
    return artifact


def predict_text(model, text):
    """Tahmin edilen etiketi ve en yüksek sınıf skorunu döndür."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Tahmin için boş olmayan bir metin girin.")
    label = str(model.predict([text])[0])
    confidence = float(max(model.predict_proba([text])[0]))
    return {"label": label, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Eğitilmiş domain modeliyle metin tahmini")
    parser.add_argument("--text", default=EXAMPLE_TEXT, help="Tahmin edilecek görüşme metni")
    args = parser.parse_args()
    if not ARTIFACT_PATH.exists():
        raise SystemExit(f"Model bulunamadı: {ARTIFACT_PATH}. Önce train_domain_classifier çalıştırın.")
    artifact = load_domain_artifact()
    result = predict_text(artifact["pipeline"], args.text)
    print(f"Metin: {args.text}")
    print(f"Tahmin: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
