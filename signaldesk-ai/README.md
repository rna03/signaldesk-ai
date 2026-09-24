# SignalDesk AI

Çağrı merkezi görüşmelerini incelemek, müşteri metnini hizmet alanlarına ayırmak, benzer konuları anlamsal kümelerde toplamak ve sorun kümelerindeki olağandışı artışları açıklanabilir bir demo üzerinden göstermek için geliştirilmiş bir AI/ML öğrenme projesidir.

> Bu sürüm canlı çağrı akışı izlemez. Kaynak veri kümesinde gerçek olay zamanları bulunmadığı için erken uyarı ekranı **sentetik zaman verisi** kullanır.

![SignalDesk AI Türkçe kullanıcı arayüzü](docs/signaldesk-dashboard.jpg)

## İçindekiler

- [Mevcut özellikler](#mevcut-özellikler)
- [Sistem akışı](#sistem-akışı)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Veri ve model dosyalarını hazırlama](#veri-ve-model-dosyalarını-hazırlama)
- [Uygulamayı çalıştırma](#uygulamayı-çalıştırma)
- [API endpointleri](#api-endpointleri)
- [Test ve doğrulama](#test-ve-doğrulama)
- [Sınırlamalar](#sınırlamalar)

## Mevcut özellikler

- Hugging Face üzerindeki `apptek-com/apptek_callcenter_dialogues` veri kümesini streaming modunda inceleme ve profil çıkarma.
- Aynı çağrıya ait müşteri ve temsilci kanallarını `file_name` ile eşleştirme; tarafı kanal numarası yerine `role` alanından belirleme.
- Müşteri metninden TF-IDF ve Logistic Regression ile `domain` tahmini.
- `sentence-transformers/all-MiniLM-L6-v2` embedding modeli ve K-Means ile anlamsal sorun kümeleri üretme.
- Küme merkezine kosinüs benzerliği ve kümeyi açıklamaya yardımcı terimler döndürme.
- Önceki saatleri temel alan Z-Score yaklaşımıyla sentetik erken uyarı demosu.
- FastAPI üzerinden tekli analiz, en fazla 50 metinlik toplu analiz, durum ve demo uyarı endpointleri.
- FastAPI tarafından servis edilen responsive Türkçe HTML/CSS/Vanilla JavaScript arayüzü.
- FLAN-T5-small ile ayrı ve isteğe bağlı bir sorun ifadesi çıkarma deneyi.

`domain`, veri kümesindeki hizmet alanıdır; doğrulanmış müşteri sorunu etiketi değildir. Anlamsal kümeler de insan tarafından onaylanmış bir sorun taksonomisi değildir.

## Sistem akışı

```text
Müşteri metni
    │
    ├── TF-IDF + Logistic Regression ──> hizmet alanı + model skoru
    │
    └── MiniLM embedding + K-Means ────> sorun kümesi + merkez benzerliği
                                              │
                                              └──> açıklayıcı küme terimleri

Ayrı demo yolu:
anlamsal kümeler + sentetik saatler ──> geçmiş pencere + Z-Score ──> erken uyarılar
```

Model eğitimi ve artifact üretimi çevrimdışı komutlarla yapılır. FastAPI, istek sırasında modeli yeniden eğitmez; daha önce üretilmiş yerel dosyaları yükler.

## Teknolojiler

| Alan | Kullanılan teknoloji |
|---|---|
| Dil | Python 3.11+ |
| Veri | Hugging Face Datasets, pandas |
| Sınıflandırma | scikit-learn, TF-IDF, Logistic Regression |
| Anlamsal temsil | Sentence Transformers, `all-MiniLM-L6-v2` |
| Kümeleme | K-Means |
| İsteğe bağlı deney | Transformers, `google/flan-t5-small` |
| API | FastAPI, Pydantic, Uvicorn |
| Arayüz | HTML, CSS, Vanilla JavaScript |
| Test | pytest |

## Proje yapısı

```text
signaldesk-ai/
├── frontend/                  # Türkçe web arayüzü
├── docs/                      # README ekran görüntüsü
├── src/signaldesk/
│   ├── api/                   # FastAPI endpointleri ve analiz servisi
│   ├── clustering/            # TF-IDF ve MiniLM kümeleme deneyleri
│   ├── data/                  # Veri inceleme, profil ve görüşme eşleştirme
│   ├── issues/                # İsteğe bağlı FLAN-T5 deneyi
│   ├── ml/                    # Domain modelini eğitme ve kullanma
│   └── monitoring/            # Sentetik erken uyarı üretimi
├── tests/                     # İnternet gerektirmeyen birim/API testleri
├── .gitignore
├── requirements.txt
└── README.md
```

## Kurulum

Windows PowerShell üzerinde:

```powershell
git clone https://github.com/rna03/signaldesk-ai.git signaldesk-ai-repo
cd .\signaldesk-ai-repo\signaldesk-ai

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

Python 3.11 veya daha yeni bir sürüm gerekir. Bu yayın hazırlığı Python 3.12.2 ile doğrulandı. Proje `src` dizin yapısını kullandığı için her yeni PowerShell oturumunda `PYTHONPATH` satırını yeniden çalıştırın.

Veri kümesi herkese açıktır; Hugging Face token veya API anahtarı gerekmez. İlk veri ve model indirmeleri internet bağlantısı ister.

## Veri ve model dosyalarını hazırlama

### İsteğe bağlı veri incelemesi

```powershell
python -m signaldesk.data.inspect_dataset
python -m signaldesk.data.profile_dataset
python -m signaldesk.data.build_conversations
```

Bu komutlar kaynak transcriptleri değiştirmez. Ses alanını indirip yazıya çevirmez; veri kümesindeki mevcut metin ve metadata üzerinde çalışır.

### Ana uygulama için gerekli artifactler

Aşağıdaki komutları sırayla çalıştırın:

```powershell
python -m signaldesk.ml.train_domain_classifier
python -m signaldesk.clustering.discover_issues_semantic
python -m signaldesk.monitoring.detect_emerging_issues
```

Komutların ürettiği yerel dosyalar:

| Dosya | Görevi |
|---|---|
| `artifacts/domain_classifier.joblib` | Müşteri metniyle eğitilmiş TF-IDF + Logistic Regression modeli |
| `artifacts/semantic_clusterer.joblib` | MiniLM vektörleriyle eğitilmiş K-Means modeli |
| `artifacts/semantic_cluster_metadata.json` | Embedding modeli, küme sayısı ve açıklayıcı terimler |
| `artifacts/early_warning_demo.json` | Sentetik zamanlı erken uyarı demo verisi |

Bu üretilmiş dosyalar Git deposuna eklenmez. Domain modeli, K-Means modeli veya semantic metadata eksik ya da uyumsuzsa `/ready` ve analiz endpointleri `503` döndürebilir. Erken uyarı demo dosyası eksikse `/api/v1/alerts/demo` kullanılamaz. İlk MiniLM kullanımı modeli Hugging Face üzerinden indirebilir.

### İsteğe bağlı FLAN-T5 deneyi

```powershell
python -m signaldesk.issues.extract_issues --sample
python -m signaldesk.issues.extract_issues --full
python -m signaldesk.clustering.discover_extracted_issues_semantic
```

Bu deney ana API analiz yolunda kullanılmaz. `--full`, Git dışında tutulan `artifacts/extracted_issues.json` dosyasını üretir ve daha fazla indirme/işlem süresi gerektirir.

## Uygulamayı çalıştırma

Artifactler hazırlandıktan sonra:

```powershell
python -m uvicorn signaldesk.api.main:app --reload
```

Ardından:

- Arayüz: <http://127.0.0.1:8000/>
- Swagger API dokümantasyonu: <http://127.0.0.1:8000/docs>

PowerShell ile açmak için:

```powershell
Start-Process "http://127.0.0.1:8000/"
Start-Process "http://127.0.0.1:8000/docs"
```

## API endpointleri

| Metot | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | Web arayüzünü döndürür. |
| `GET` | `/health` | Sürecin yanıt verdiğini kontrol eder; model yüklemez. |
| `GET` | `/ready` | Yerel model ve metadata dosyalarının temel sözleşmesini kontrol eder. |
| `GET` | `/api/v1/info` | API sürümü ve kullanılan model bilgilerini döndürür. |
| `POST` | `/api/v1/analyze` | Tek bir müşteri metnini analiz eder. |
| `POST` | `/api/v1/analyze/batch` | 1–50 müşteri metnini girdi sırasıyla analiz eder. |
| `GET` | `/api/v1/alerts/demo` | Sentetik erken uyarı verisini döndürür. |
| `GET` | `/docs` | Swagger arayüzünü açar. |

Tekli analiz örneği:

```powershell
$body = @{
    customer_text = "My modem keeps losing connection."
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/analyze" `
    -ContentType "application/json" `
    -Body $body
```

Yanıt; `domain`, `domain_confidence`, `semantic_cluster`, `cluster_similarity`, `cluster_descriptive_terms` ve `analysis_metadata` alanlarını içerir. `domain_confidence`, sınıflandırıcının `predict_proba` çıktısıdır ve kalibre edilmiş kesinlik değildir. `cluster_similarity` olasılık değil, atanan küme merkezine kosinüs benzerliğidir.

Toplu analiz örneği:

```powershell
$body = @{
    items = @(
        @{ customer_text = "My modem keeps losing connection." },
        @{ customer_text = "My bank transfer has not arrived." }
    )
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/analyze/batch" `
    -ContentType "application/json" `
    -Body $body
```

## Test ve doğrulama

Python testleri internetten veri veya model indirmeden çalışır:

```powershell
python -m pytest -q
```

Node.js kuruluysa JavaScript sözdizimi ayrıca kontrol edilebilir:

```powershell
node --check frontend\js\api.js
node --check frontend\js\app.js
```

Bu yayın hazırlığında doğrulanan sonuçlar:

- Python 3.12.2 ortamında `pip check`: başarılı.
- `python -m pytest -q`: **61 test geçti**.
- Her iki JavaScript dosyasında `node --check`: başarılı.
- Çalışan uygulamada `/health`, `/ready`, tekli analiz ve toplu analiz akışları: başarılı.
- Türkçe arayüz ve dar ekran görünümü: tarayıcıda doğrulandı.

Bağımlılıklardan gelen deprecation uyarıları testleri başarısız kılmaz.

## Güvenlik ve depo içeriği

- API anahtarı veya Hugging Face token kaynak koda yazılmaz.
- `.env`, sanal ortam, önbellek, yerel model dosyaları, üretilmiş transcript/deney çıktıları ve paketleme çıktıları `.gitignore` kapsamındadır.
- Arayüz, API’den gelen metinleri HTML olarak enjekte etmek yerine `textContent` ile gösterir.
- Uygulamada kullanıcı kimlik doğrulaması ve yetkilendirme bulunmaz; internete açık üretim servisi olarak kullanılmamalıdır.

## Sınırlamalar

- Kaynak veri İngilizce çağrı merkezi transcriptlerinden oluşur; canlı müşteri trafiği olarak değerlendirilmemelidir.
- Ses kaydını yazıya çevirme uygulanmamıştır. Whisper veya başka bir speech-to-text sistemi kullanılmaz.
- Veri kümesinde gerçek çağrı zaman damgaları yoktur. Erken uyarı demosu sentetik saatler ve kontrollü artışlar kullanır; gerçek zamanlı izleme veya alarm gönderimi yapmaz.
- Kalıcı veritabanı, uyarı geçmişi, kullanıcı hesabı, yetkilendirme ve Docker yapılandırması yoktur.
- `domain` hizmet alanıdır; onaylanmış sorun etiketi değildir. K-Means kümeleri ve açıklayıcı terimler insan doğrulamasından geçmiş bir sorun taksonomisi değildir.
- Model skorları karar garantisi değildir. `domain_confidence` kalibre edilmiş olasılık, `cluster_similarity` doğruluk skoru değildir.
- `/ready`, yerel artifact sözleşmesini kontrol eder; MiniLM önbelleğini veya gelecekteki ağ erişimini tam olarak garanti etmez.
- Toplu analiz en fazla 50 öğe kabul eder. Tek bir metin için ayrıca tanımlanmış uzunluk sınırı yoktur.
- FLAN-T5 deneyi ana API ve arayüzün analiz yoluna bağlı değildir; model fine-tuning uygulanmamıştır.
