"""AppTek çağrı merkezi veri kümesinin yapısını küçük bir örnekle gösterir."""

from datasets import load_dataset


DATASET_NAME = "apptek-com/apptek_callcenter_dialogues"
PREVIEW_COUNT = 3
TEXT_LIMIT = 300


def readable_value(value):
    """Uzun metinleri ve ses verisini terminal için güvenli biçimde kısalt."""
    if isinstance(value, dict) and "bytes" in value:
        return {"path": value.get("path"), "bytes": "<gösterilmiyor>"}
    if isinstance(value, bytes):
        return "<ikili veri gösterilmiyor>"
    if isinstance(value, str) and len(value) > TEXT_LIMIT:
        return value[:TEXT_LIMIT] + "… [kısaltıldı]"
    return value


def main():
    print(f"Veri kümesi: {DATASET_NAME} (default yapılandırması)")
    try:
        # streaming=True tüm ses arşivini indirmeden satırları sırayla okumayı sağlar.
        splitler = load_dataset(DATASET_NAME, streaming=True)
        if not splitler:
            raise RuntimeError("Veri kümesinde okunabilir split bulunamadı.")

        print("\nMevcut split'ler:")
        toplam = 0
        for ad, veri in splitler.items():
            bilgi = veri.info.splits.get(ad) if veri.info and veri.info.splits else None
            sayi = bilgi.num_examples if bilgi else None
            if sayi is None:
                # Metaveri eksikse akışı say; kayıtları bellekte tutma, sesi açma.
                sayi = sum(1 for _ in veri.decode(False))
            print(f"- {ad}: {sayi} örnek")
            toplam += sayi
        print(f"Toplam örnek sayısı: {toplam}")

        # İlk mevcut split yalnızca önizleme için kullanılır; kaynak veri değişmez.
        ad, veri = next(iter(splitler.items()))
        veri = veri.decode(False)  # Ses dosyasını açma veya çözümleme.
        ornekler = list(veri.take(PREVIEW_COUNT))
        kolonlar = list(veri.features) if veri.features else list(ornekler[0]) if ornekler else []
        print(f"\nİncelenen split: {ad}")
        print("Kolonlar:", ", ".join(kolonlar) if kolonlar else "bulunamadı")

        print("\nKolon ve tip özeti (şemadaki tip / ilk örnekteki Python tipi):")
        for kolon in kolonlar:
            sema_tipi = str(veri.features[kolon]) if veri.features else "bilinmiyor"
            python_tipi = type(ornekler[0].get(kolon)).__name__ if ornekler else "örnek yok"
            print(f"- {kolon}: {sema_tipi} / {python_tipi}")

        print(f"\nİlk {len(ornekler)} örnek (uzun metinler kısaltılır; ses içeriği gösterilmez):")
        for sira, ornek in enumerate(ornekler, 1):
            print(f"\nÖrnek {sira}:")
            for kolon, deger in ornek.items():
                print(f"  {kolon}: {readable_value(deger)}")
    except Exception as hata:
        print(f"\nVeri kümesi yüklenemedi: {type(hata).__name__}: {hata}")
        print("Erişim, bağlantı ve veri kümesi sayfasını kontrol edin; başka veri kümesine geçilmedi.")
        raise SystemExit(1) from hata


if __name__ == "__main__":
    main()
