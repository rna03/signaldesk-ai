# SignalDesk AI — Phase 1–9

SignalDesk AI'ın uzun vadeli amacı, müşteri destek görüşmelerindeki tekrar eden ve çözülmeyen sorunları fark edip olağandışı artışları erken göstermektir. İlk iki aşama yalnızca gerçek veri kümesinin yapısını inceler ve aynı görüşmenin iki kaydını bir araya getirir; model eğitmez.

## Bu aşamada kullandığımız veri

[AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) seçildi. Herkese açık sayfasında müşteri ve temsilci konuşmalarının İngilizce dökümleri, ses dosyaları ve `role`, `domain`, `accent` gibi alanlar bulunuyor. Varsayılan yapılandırmanın tek split'i `test`; veri kartına göre 1.746 konuşmacı kanalı kaydı içeriyor. Görüşmeler rol yapılarak kaydedilmiş: gerçek müşteri şikâyetlerinin sıklığı veya zaman içindeki artışı bu veriden doğrudan çıkarılamaz. Veri kartı bu veriyi özellikle değerlendirme ve analiz için tanımlıyor. Lisans: CC BY-SA 4.0.

Hugging Face, veri kümelerinin ve modellerin paylaşılabildiği bir ekosistemdir. **Hugging Face Hub**, bu veri kümesinin sayfası, açıklaması ve dosyalarının bulunduğu çevrim içi yerdir. **Dataset (veri kümesi)**, burada her satırı bir konuşmacı kanalına ait olan düzenli kayıtlar topluluğudur. Python'daki **`datasets` kütüphanesi** Hub üzerindeki bu kayıtları okur. **`load_dataset()`**, verdiğimiz `apptek-com/apptek_callcenter_dialogues` adını Hub'da bulup veri kümesini Python'da erişilebilir hale getirir. Betikteki `streaming=True`, büyük ses verisini topluca indirmeden ilk birkaç satıra bakmamızı sağlar.

**Split**, veri kümesinin adlandırılmış bölümüdür. Yaygın olarak `train` öğrenme, `validation` geliştirme sırasında kontrol, `test` son değerlendirme için kullanılır. Bu kümenin varsayılan yapılandırmasında yalnızca `test` vardır; betik olmayan `train` veya `validation` bölümleri varmış gibi göstermez. **Feature/column (özellik/sütun)**, her kayıttaki bir alanı ifade eder. Örneğin `text` konuşma dökümü, `role` konuşanın müşteri mi temsilci mi olduğunu belirtir. Betik her sütunun veri kümesi şemasındaki tipini ve ilk örnekteki Python tipini yan yana gösterir.

Phase 1'de model kullanmıyoruz; önce kayıtların hangi alanları içerdiğini ve hangi sınırları olduğunu anlamamız gerekiyor. İleride `text` ve `role` alanları müşteri sorunlarını anlamaya yardımcı olabilir. Ancak bu veri kümesinde gerçek olay tarihi ve çözüm durumu alanları gösterilmiyor; erken uyarı ve çözümsüzlük değerlendirmesi için ayrıca uygun veri gerekir.

## Phase 2: veri profili ve görüşme eşleştirmesi

**Dataset profiling**, bütün kayıtları tarayıp veri kümesinin şeklini sayılarla anlamaktır. Burada kaç `customer` ve `agent` satırı olduğu, `domain` dağılımı, boş `text` sayısı ve metin uzunlukları buna örnektir. Modelden önce bunu yapıyoruz: iki konuşmacı kaydını yanlış birleştirirsek sonraki analizler de yanlış görüşmeyi anlatır.

**Row/record (satır/kayıt)** bu kümede tek konuşmacı kanalının dökümüdür. **Conversation (görüşme)** ise aynı çağrıdaki müşteri ve temsilci kayıtlarının birlikte tutulduğu yapıdır. **Conversation pairing**, `file_name` sonundaki `_channel1.wav` veya `_channel2.wav` bölümünü çıkarıp kalan çağrı kimliğine göre iki satırı bulma işlemidir. Dosya adının biçimini varsaymıyoruz: bütün veri kümesinde her kimliğin iki farklı kanalı, bir `customer` ve bir `agent` rolü ve aynı `domain` değeri olup olmadığını kontrol ediyoruz. Kanal numarası rolü belirlemez; gerçek veride `channel1` bazen müşteridir.

**Metadata**, metnin yanında gelen açıklayıcı alanlardır. `role` konuşanın müşteri mi temsilci mi olduğunu gösterir ve metni doğru alana koymak için kullanılır. `domain` görüşmenin hizmet alanını (örneğin `banking`) belirtir; iki tarafta aynı olmasını bekleriz. `accent` konuşmacının İngilizce aksan grubudur. `speaker_id` aynı konuşmacının farklı kayıtlarını tanımaya yarayan takma kimliktir; çağrı kimliği değildir. `audio` bu yükleme biçiminde tüm satırlarda `None` göründü. Ses dosyası yolları `file_name` içinde bulunur; bu aşamada ses indirilmez veya çözülmez.

Görüşme yapısındaki `customer_text` ve `agent_text`, kaynak dökümlerin birebir değerleridir. `(Um)` ve `(uh)` gibi ifadeleri silmiyoruz, küçük harfe çevirmiyoruz ve stopword çıkarmıyoruz: önce özgün kaydı koruyup ne içerdiğini anlamak istiyoruz. Terminaldeki iki örneğin **görünümü** uzunluk için kısaltılır; Python'da oluşturulan görüşmedeki metin kısaltılmaz. Phase 2 sonunda **model eğitilmiş olmayacak**.

Gerçek veriyle doğrulanan özet: `test` split'inde 1.746 satır, 873 benzersiz görüşme kimliği, 873 tam eşleşme, 0 bozuk/eksik eşleşme; 873 `customer`, 873 `agent` satırı. `channel1`: 807 agent ve 66 customer; `channel2`: 66 agent ve 807 customer. `text` boşluğu 0, `audio` için `None` sayısı 1.746. `build_conversations.py` hiçbir CSV/JSON dosyası yazmaz; veri kaynağı Hugging Face olarak kalır.

## Dosyalar ve önemli satırlar

- `src/signaldesk/__init__.py`: `signaldesk` klasörünü Python paketi yapar.
- `src/signaldesk/data/__init__.py`: veri keşfi kodunun bulunduğu alt paketi tanımlar.
- `src/signaldesk/data/inspect_dataset.py`: `DATASET_NAME` ile veri kaynağını sabitler. `load_dataset(..., streaming=True)` mevcut split'leri açar. `veri.info.splits` yayımlanmış örnek sayılarını okur; sayı yoksa `sum(1 for _ in veri.decode(False))` satırları bellekte biriktirmeden sayar. `veri.decode(False)` sesin çözülmesini engeller. `take(3)` yalnızca ilk üç satırı önizler. `readable_value()` uzun metni kısaltır ve ikili ses içeriğini ekrana basmaz. Hata yakalama bölümü gerçek hata türünü ve mesajını gösterir, başka veri kümesine geçmez.
- `src/signaldesk/data/profile_dataset.py`: tüm satırları akış halinde tarar; `Counter` ile dağılımları, `statistics` ile metin uzunluklarını hesaplar. `extract_call_id()` dosya adını beklenen biçimde ayrıştırır; uymayanı sessizce eşleştirmez. Görüşme başına kayıt, kanal, rol ve domain tutarlılığını sayar.
- `src/signaldesk/data/build_conversations.py`: `pair_records()` tam iki kanalı, iki farklı rolü ve ortak domain'i zorunlu kılar. Müşteri ve temsilci metnini kanal numarasına göre değil `role` değerine göre yerleştirir. `build_conversations()` bozuk bir grup varsa kısmi sonuç döndürmez; kaynak metni değiştirmez.
- `tests/test_environment.py`: paketimizi ve üç temel bağımlılığı import eder; ağ bağlantısı veya dataset indirmesi istemez.
- `tests/test_pairing.py`: küçük sentetik kayıtlarda çağrı kimliğini, ters kanal/rol düzenini ve eksik/bozuk eşleşmelerin reddini internet olmadan sınar.
- `src/signaldesk/ml/__init__.py`: klasik makine öğrenmesi kodunun Python paketi olduğunu belirtir.
- `src/signaldesk/ml/train_domain_classifier.py`: doğrulanmış görüşmeleri yeniden `build_conversations()` ile oluşturur. Phase 3'te iki transcripti birleştiren ilk deneyi yaptı; Phase 9'da müşteri metni girdisiyle aynı TF-IDF + Logistic Regression yöntemini değerlendirir. `split_examples()` görüşmeleri sabit tohum ve `stratify` ile 80/20 ayırır. `model.fit(x_train, y_train)` yalnızca train verisini görür. Test tahminleri, metrikler ve karışıklık matrisi yazdırılır; model ve girdi sözleşmesi yerel artifact olarak saklanır.
- `src/signaldesk/ml/predict_domain.py`: kaydedilmiş modeli yükleyip verilen metnin domain tahminini ve en yüksek sınıf skorunu gösterir. Yalnızca kendi eğittiğiniz, güvenilir yerel joblib dosyasını açın.
- `tests/test_domain_classifier.py`: metin/etiket hazırlamayı, görüşme ayrımını ve tahmin çıktısını sentetik verilerle sınar; gerçek veri indirmez.
- `src/signaldesk/clustering/__init__.py`: etiketsiz keşif kodunun Python paketi olduğunu belirtir.
- `src/signaldesk/clustering/discover_issues.py`: doğrulanmış görüşmeleri yeniden oluşturur, yalnızca `customer_text` ile TF-IDF matrisi üretir ve farklı K-Means adaylarını karşılaştırır. `make_vectorizer()` kaynak metni değiştirmeden sayısal özellik oluşturur. `cluster_counts()` dengeyi, `top_terms()` merkezde ağır basan sözcükleri, `representative_indices()` merkeze yakın gerçek görüşmeleri bulur. `select_candidate()` silhouette yakınlığını ve küme büyüklüğünü birlikte dikkate alır. `domain` yalnızca sonuçlar üretildikten sonra dağılım göstermek için okunur.
- `tests/test_issue_clustering.py`: sentetik müşteri metinleriyle TF-IDF, K-Means, sayım, üst terimler ve K seçimi yardımcılarını internet olmadan sınar.
- `src/signaldesk/clustering/discover_issues_semantic.py`: hazır `all-MiniLM-L6-v2` modelini yükleyip yalnızca `customer_text` için embedding üretir; aynı K adaylarını dener, gerçek metin örneklerini ve sonradan hesaplanan domain dağılımını gösterir. `load_embedding_model()` ağ/model yüklemeyi sayısal yardımcılardan ayırır. `encode_full_conversations()` uzun metinleri model sınırına sığan parçalara ayırır; `aggregate_chunk_embeddings()` parça vektörlerini tek görüşme vektörüne dönüştürür. `descriptive_top_terms()` kümeleme bittikten sonra ayrı TF-IDF temsiliyle insanın inceleyebileceği terimleri bulur; bu temsil K-Means'e verilmez.
- `tests/test_semantic_clustering.py`: sentetik vektörlerle cosine similarity, K-Means, küme büyüklüğü, merkez yakınlığı ve betimleyici terimleri sınar; modeli indirmez.
- `src/signaldesk/monitoring/__init__.py`: zaman temelli erken uyarı denemelerinin Python paketini tanımlar.
- `src/signaldesk/monitoring/detect_emerging_issues.py`: Phase 5'in gerçek görüşme, embedding ve K=16 kümeleme işlevlerini yeniden kullanır. `synthetic_events()` yalnızca demo zamanı ekler; `inject_synthetic_surge()` mevcut olayları son saatlere taşır. `aggregate_hourly()` boş saatleri sıfırla doldurur. `score_bucket()` mevcut saati dışarıda tutarak geçmiş ortalama ve standart sapmayı hesaplar. `detect_alerts()` geçmiş, asgari olay sayısı ve z-score koşullarını birlikte uygular.
- `tests/test_emerging_issue_detection.py`: saatlik sayım, geçmiş sızıntısı, sıfır standart sapma, eşikler, normal seri, sıçrama ve deterministik sentetik zamanları internetsiz sınar.
- `src/signaldesk/api/__init__.py`: API alt paketini tanımlar.
- `src/signaldesk/api/schemas.py`: Pydantic istek ve yanıt şekillerini tanımlar. Boş müşteri metnini, boş batch'i ve 50 öğeyi aşan batch'i reddeder; geçerli transcripti değiştirmez. `analysis_metadata` alanının şeklini de belirtir.
- `src/signaldesk/api/services.py`: domain modelini, hazır MiniLM'yi, fitted K-Means'i ve küme metadata'sını gerektiğinde yükleyip önbellekte tutar. Tekli ve çoklu analiz aynı akışı kullanır; `demo_alerts` yalnızca Phase 6'nın yapılandırılmış sentetik sonucunu okur.
- `src/signaldesk/api/main.py`: `/health`, `/ready` ve `/api/v1/` HTTP route'larını tanımlar. Route servis çağrısını ve güvenli HTTP hata dönüşünü yapar; model mantığı route içine yazılmaz.
- `tests/test_api.py`: `TestClient` ve sahte servisle HTTP sözleşmesini, girdi reddini, readiness'i ve kontrollü 503/500 yanıtını internet olmadan sınar.
- `requirements.txt`: veri/ML bağımlılıklarına ek olarak API için yalnızca `fastapi` ve `uvicorn` ekler. Pydantic FastAPI bağımlılığı olarak gelir.
- `.gitignore`: sanal ortamı, önbelleği, sır içerebilen `.env` dosyalarını ve yeniden üretilebilir yerel model/demo artifact'lerini Git dışında tutar.

## Windows PowerShell'de çalıştırma

Python 3.11 veya üzeri kurulu olmalıdır. Komutları bu klasörün **üst dizininde** sırayla çalıştırın:

```powershell
cd .\signaldesk-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m signaldesk.data.inspect_dataset
python -m signaldesk.data.profile_dataset
python -m signaldesk.data.build_conversations
python -m signaldesk.ml.train_domain_classifier
python -m signaldesk.ml.predict_domain
python -m signaldesk.ml.predict_domain --text "I need help with my bank transfer."
python -m signaldesk.clustering.discover_issues
python -m signaldesk.clustering.discover_issues_semantic
python -m signaldesk.monitoring.detect_emerging_issues
python -m pytest -q
python -m uvicorn signaldesk.api.main:app --reload
```

Bilgisayarınızda Python 3.12 yoksa ve 3.11 varsa `py -3.11 -m venv .venv` kullanın. `PYTHONPATH`, `src` içindeki paketin kurulum paketi oluşturmadan import edilmesini sağlar. Yeni bir PowerShell oturumunda testi veya betiği tekrar çalıştırırken bu satırı tekrar girin.

Dataset sayfası herkese açık ve bu makinede token olmadan ilk üç kayıt Python ile okundu. Kısıtlı ağ ortamında ilk deneme `ConnectionError: Couldn't reach 'apptek-com/apptek_callcenter_dialogues' on the Hub (LocalEntryNotFoundError)` hatası verdi; ağ erişimi açıldığında aynı komut başarılı oldu. Bir bağlantı veya izin sorunu olursa betik gerçek hata türünü ve mesajını basıp başarısız çıkış koduyla durur. İlk çalıştırma internet gerektirir; pytest internet gerektirmez. Veri kartı: [kaynak ve kullanım açıklaması](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues).

## Phase 3: ilk makine öğrenmesi baseline'ı

**Machine Learning (makine öğrenmesi)**, örneklerden bir örüntü öğrenip yeni bir örnek hakkında tahmin yapmaktır. Burada her görüşmenin müşteri ve temsilci dökümü bir örnek; görüşmenin `domain` değeri doğru cevaptır. Doğru cevapları eğitim sırasında verdiğimiz için bu **supervised learning (denetimli öğrenme)** örneğidir. Sonuç, `banking`, `telecom` gibi sınıflardan biri olduğu için görev **classification (sınıflandırma)**dır. Bu model müşteri sorununun çözüldüğünü veya yeni bir sorun dalgası başladığını söylemez; yalnızca domain tahmin eder. **Baseline**, ilerideki yöntemlerin karşılaştırılacağı ilk ve basit sonuçtur.

**Feature (girdi özelliği)** modelin tahmin için kullandığı bilgidir. Bu çalışmada yalnızca `customer_text + " " + agent_text` kullanılır. **Label/target (etiket/hedef)** ise öğrenmeye çalıştığımız `domain`dur. `speaker_id`, `file_name`, `conversation_id`, `accent` ve `gender` metnin içine eklenmez. Bu alanlar tesadüfi ipuçları verebilir; test başarısını olduğundan yüksek gösterebilir. Buna **data leakage (veri sızıntısı)** denir. Aynı görüşmeyi hem train hem test bölümüne koymak da sızıntıdır; kod bunu ayrıca kontrol eder.

Bilgisayar ham metni doğrudan sayısal modele veremez. **TF-IDF**, metindeki kelime ve iki kelimelik dizileri sayısal özelliklere çevirir. Bir görüşmede sık görünen, ama bütün görüşmelerde aynı derecede sık olmayan terimler daha ayırt edici olur. Örneğin `account` bankacılık için yararlı olabilir; hemen her konuşmadaki `hello` daha az ayırt edicidir. TF-IDF sözlüğü yalnızca **train set** üzerinde öğrenilir. **Logistic Regression**, adında “Regression” geçmesine rağmen sınıflandırma yapabilen bir modeldir: sayısal metin özelliklerini kullanarak 16 domain için skor üretir ve en yüksek skorlu sınıfı seçer. Bu aşamada Deep Learning kullanmadık; basit model veri akışını ve hataları daha kolay incelememizi sağlar.

TF-IDF kendi sayısal temsilini oluştururken kelimeleri küçük harfe dönüştürebilir; bu işlem yalnızca modelin içindeki özellik çıkarma adımıdır. Kaynak dataset ve görüşme yapısındaki özgün transcriptler değiştirilmez.

**Train set**, modelin öğrendiği görüşmelerdir; **test set**, eğitim bittiğinde yeni görüşme gibi kullanılan ayrı bölümdür. Kod görüşmeleri %80/%20 ayırır, `random_state=42` ile ayrımı tekrar üretilebilir yapar ve `stratify` ile domain oranlarını yaklaşık korur. `model.fit()` TF-IDF sözlüğünü ve Logistic Regression ağırlıklarını yalnızca train üzerinde öğrenir. `model.predict()` eğitimde görmediği test metinlerine domain etiketi verir. Test verisi eğitime girseydi sonuç, gerçek yeni görüşmelere genelleme hakkında güvenilir bilgi vermezdi.

**Accuracy**, tüm test görüşmeleri içinde doğru tahmin oranıdır. **Precision**, bir domain diye tahmin edilenlerin ne kadarının gerçekten o domain olduğu; **Recall**, gerçekten o domaindeki görüşmelerin ne kadarının bulunduğudur. **F1 score**, precision ve recall'u birlikte özetler. **Macro F1**, 16 domainin her birine eşit ağırlık verip F1 değerlerini ortalar; çok kayıtlı domainlerin az kayıtlıları gölgelemesini önler. **Confusion matrix (karışıklık matrisi)**, gerçek domainlerin hangi domain olarak tahmin edildiğini satır ve sütunlarla gösterir. Betik en sık karışan on farklı domain çiftini ayrıca yazar.

`predict_proba()` çıktısındaki en yüksek değer **confidence** olarak gösterilir. Bu, modelin kendi sınıfları arasındaki skorudur; otomatik olarak kalibre edilmiş gerçek olasılık veya “kesinlik” değildir. Yüksek skor yanlış tahminde de görülebilir. Manuel olarak yazdığınız kısa bir metin, eğitimdeki tam görüşmelerden farklı olduğundan sonucu özellikle dikkatle yorumlayın.

Hugging Face burada **dataset sağlar**. Hugging Face'ten hazır bir model kullanmıyoruz. Bu Phase'deki Logistic Regression modelini **biz**, bu verinin train bölümünde eğitiyoruz. Küçük baseline modeli `artifacts/domain_classifier.joblib` dosyasına yerel olarak kaydedilir. `.gitignore` bu dosyayı commit dışında tutar: kaynak veri ve koddan yeniden üretilebilir; ayrıca model dosyası eğitim verisinden izler taşıyabilir. Veri kümesinin tamamı da repoya yazılmaz.

### Doğrulanan ilk çalışma

873 görüşme 698 train ve 175 test görüşmesine ayrıldı. Test sonucunda Accuracy **0.9257**, Macro Precision **0.9505**, Macro Recall **0.9093**, Macro F1 **0.9237** çıktı. Yanlış tahminlerde en sık karışan yönlü domain çiftlerinin her biri birer kez görüldü; örneğin `hospitality -> entertainment`, `finance -> banking`, `telecom -> entertainment`. Tam sınıf raporu, karışıklık matrisi ve test örnekleri eğitim komutunun terminal çıktısında yer alır. Varsayılan manuel metin tahmini `banking`, confidence **0.1029** oldu; kısa metin için bu düşük skoru kesin sonuç saymayın. Çevrimdışı pytest sonucu: **12 passed**. Sonuçlar veri ve kütüphane sürümüne göre değişebilir.

## Phase 4: etiketsiz müşteri konusu keşfi

Phase 3'te **supervised learning** kullandık: her görüşmenin doğru `domain` etiketi vardı. Burada **unsupervised learning (etiketsiz öğrenme)** kullanıyoruz; veri kümesinde müşterinin *problem türünü* söyleyen doğrulanmış bir etiket yok. **Clustering (kümeleme)**, benzer metinleri gruplamayı dener. Kümelerin ne anlama geldiğini bir insanın örneklerden incelemesi gerekir. Bu nedenle Phase 3'teki Accuracy/F1 değerlerini burada hesaplamıyoruz: karşılaştırılacak gerçek problem etiketleri yok.

**K-Means**, metinlerin sayısal temsilini K adet gruba ayırır. **K**, istediğimiz küme sayısıdır; bir problem türü sayısının gerçek cevabı değildir. Her grubun **centroid (merkez)** denen ortalama bir TF-IDF vektörü vardır. Algoritma vektörleri yakın merkezlere yerleştirir. TF-IDF'yi tekrar kullanabiliyoruz çünkü bu yöntem etiket gerektirmeden metindeki terimlerden sayısal vektör oluşturur. Burada yalnızca `customer_text` kullanıyoruz: müşterinin anlattığı ihtiyaca odaklanmak ve temsilcinin standart yanıtlarının etkisini azaltmak için. `agent_text`, `domain`, konuşmacı kimliği, dosya adı, aksan ve cinsiyet model girdisine verilmez. `domain` yalnızca **kümeleme bittikten sonra** kümelerin dağılımını anlamak için kullanılır.

Vektörler L2 normlu olduğundan metinler arasındaki **cosine similarity**, sözcük ağırlıklarının yönce benzerliğini anlatır: ortak ayırt edici terimler arttıkça değer yükselir. **Cosine distance** bu benzerliğin tersine işleyen uzaklıktır. K-Means merkezlere Öklid uzaklığı kullanır; bu betikte **Silhouette Score** ayrıca cosine distance ile hesaplanır. Silhouette, bir görüşmenin kendi kümesine, diğer kümelere göre ne kadar yakın olduğunu özetler. Yüksek değer tek başına iyi *problem* kümeleri bulduğumuzu kanıtlamaz; konuşma tarzı, tekrar eden kelimeler veya geniş domain konuları da ayrışabilir. [scikit-learn metin kümeleme örneği](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html) de merkez terimlerinin yalnızca yorumlama ipucu olduğunu gösterir.

Vektörleştirici İngilizce yaygın sözcükleri ve ilk denemede kümeleri domine ettiği görülen `um`, `uh`, `yes`, `ohh`, `hm` gibi konuşma dolgularını **yalnızca TF-IDF sözlüğünden** çıkarır. `min_df=3`, tekil rastlantısal terimleri azaltır; `max_df=0.85`, neredeyse her görüşmedeki sözcükleri sınırlar; en çok 12.000 özellik ve tek/ikili sözcük dizileri kullanılır. Özgün transcriptler yerinde kalır. **Top terms**, küme merkezinde TF-IDF ağırlığı yüksek terimlerdir; kümenin yaklaşık içeriğini anlatır fakat otomatik ve kesin bir problem etiketi değildir. **Merkeze yakın örnek**, o kümenin merkez vektörüne Öklid uzaklığı en az olan gerçek müşteri dökümüdür. Bu örnekler yalnızca terminal gösteriminde kısaltılır.

### Gerçek veriyle K seçimi ve sınırlar

873 müşteri dökümü için 10.326 TF-IDF özelliği oluştu. Sabit `random_state=42` ile denenen adaylar:

| K | Cosine silhouette | En küçük küme | En büyük küme |
|---:|---:|---:|---:|
| 8 | 0.0310 | 52 | 218 |
| 12 | 0.0414 | 35 | 260 |
| 16 | 0.0451 | 19 | 127 |
| 20 | 0.0382 | 18 | 113 |
| 24 | 0.0455 | 12 | 90 |
| 30 | 0.0365 | 11 | 65 |

**K=16** seçildi. K=24'ün silhouette değeri yalnızca 0.0004 daha yüksek; K=16 daha az kümeyle incelemeyi kolaylaştırıyor ve en büyük kümesi verinin %14,5'i. K=12'nin 260 görüşmelik genel kümesi (%29,8) çok geniş. Kod, en az 5 üyeli ve en büyük kümesi toplamın en fazla %20'si olan adaylardan, en iyi silhouette değerine 0.005 yakın en küçük K'yi seçer. Bu eşikler keşif için bilinçli bir denge kuralıdır; gerçek problem türü sayısını kanıtlamaz. K=16 üst terimlerinde teslim edilmeyen paketler, uçuş değişikliği, banka transferi, doktor randevusu ve ürün iadesi gibi incelenebilir konular görülüyor.

Sonuç yine de sınırlı. Cluster 7 (`need, guys, think...`, 106 görüşme) 16 domaini karıştırıyor. Cluster 9 (`package, perfect, good...`, 127 görüşme) seyahat, teslimat ve perakendeyi birlikte tutuyor; `package` farklı anlamlarda kullanılabiliyor. Cluster 14 (`zero, zero zero, flight...`, 45 görüşme) konuşmalarda söylenen sayıların etkisini gösteriyor. Cluster 10 teknoloji ve telekomu (36 ve 30 görüşme) birleştiriyor. Bazı kümeler problemden çok geniş hizmet alanını yakalıyor; örneğin `flight/seat` veya `energy/solar`. **0.0451 düşük bir silhouette değeridir** ve bu karma kümelerle birlikte TF-IDF'nin yalnızca sözcük benzerliğine dayanmasının sınırını gösterir. Küme numaraları veya üst terimleri “gerçek problem etiketi” olarak kullanmayın. Bu aşama zaman içinde artış veya çözülmeyen sorun tespiti de yapmaz.

Betik her kümenin büyüklüğünü, üst terimlerini, sonradan hesaplanan domain dağılımını ve merkeze en yakın üç gerçek `customer_text` önizlemesini terminale basar. Hiçbir CSV/JSON ya da model artifact dosyası oluşturmaz. Phase 4 doğrulamasında **15 pytest testi geçti**.

## Phase 5: hazır modelden anlamsal embedding ile keşif

Phase 4'te TF-IDF, her kelime veya kelime çiftine ayrı bir özellik ayırdı; çoğu görüşmede bu özelliklerin büyük kısmı sıfırdır. Buna **sparse vector (seyrek vektör)** denir. **Semantic embedding (anlamsal gömme)** ise hazır bir modelin metni daha kısa bir **dense vector (yoğun vektör)** içine dönüştürmesidir. Bu boyutlar tek tek kelimelerin sayaçları değildir; modelin eğitim sırasında öğrendiği birleşik dil örüntülerini temsil eder. Örneğin “My internet keeps disconnecting.” ile “My connection drops every few minutes.” az sözcük paylaşsa da benzer bir durumu anlatabilir. TF-IDF bu kelime farkında zorlanabilir; embedding bu iki anlamı yakın vektörlere koymayı amaçlar. Üçüncü örnek “I need to change my flight seat.” farklı bir konudur. Betik bu üç cümlenin gerçek model çıktıları arasında cosine similarity hesaplar; beklenen sonucu koda yazmaz.

**Pretrained model**, daha önce başka verilerle eğitilmiş ve hazır ağırlıklarla yayımlanmış modeldir. Burada [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) kullanılır. **Bu aşamada modeli eğitmiyoruz ve fine-tune etmiyoruz.** Model yalnızca `customer_text` için vektör üretir; bu kullanım **inference (çıkarım)**dır. `normalize_embeddings=True`, her vektörün uzunluğunu 1'e ölçekler. **Cosine similarity**, iki vektörün yönce yakınlığını ölçer; benzer anlamlı ifadelerin embedding uzayında daha yakın olması hedeflenir. Yüksek cosine similarity **aynı gerçek müşteri problemi** olduklarını garanti etmez.

K-Means'e verilen girdiler yalnızca müşteri metninin embeddingleridir. `agent_text`, `domain`, `speaker_id`, `conversation_id`, `gender`, `accent` ve `file_name` verilmez. Domain dağılımı ancak kümeleme bittikten sonra yorumlama için hesaplanır; domain gerçek issue etiketi değildir. Embedding boyutları kelime olmadığı için model merkezinden doğrudan “top terms” okumuyoruz. Bunun yerine her kümeye atanmış özgün müşteri metinleri üzerinde **ayrı bir TF-IDF analiz katmanı** kurup betimleyici terimleri gösteriyoruz. Bu TF-IDF matrisi kümeleri oluşturmaz ve cluster ID'lerini değiştirmez.

Bu genel amaçlı model SignalDesk çağrılarına özel eğitilmedi. [Model kartına göre](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), uzun girdiler varsayılan olarak 256 word piece sonrasında kesilir. Gerçek veride görüşmelerin çoğu bu sınırı aştığından betik metni token sınırına sığan parçalara ayırır, her parçayı hazır modelle encode eder, parça uzunluğuna göre vektörleri ortalar ve görüşme vektörünü tekrar normalize eder. Böylece metnin sonu sessizce kaybolmaz; fakat basit ortalama uzun bir görüşmenin cümleler arası bağlamını bütünüyle koruyamaz. Kaynak transcript hiç değiştirilmez. Bu temsil, Phase 4'ün tam metin TF-IDF sonucuyla aynı tür özellikler üretmediğinden silhouette değerleri doğrudan kalite garantisi değildir. Domain dağılımı doğrulanmış problem etiketi değildir ve kümeler doğrulanmış müşteri problem kategorileri sayılmamalıdır.

Hazır model dosyaları varsayılan Hugging Face önbelleğinde tutulur; repoya eklenmez. Eğer önbellek proje altında oluşturulursa `.gitignore` içindeki `.cache/` kuralı onu dışlar. Phase 7 için betik, seçilen fitted K-Means'i `artifacts/semantic_clusterer.joblib` ve aynı koşudaki yalnızca betimleyici terimleri `artifacts/semantic_cluster_metadata.json` olarak yazar. Müşteri transcriptleri veya embedding matrisi kalıcı CSV/JSON'a yazılmaz. İki artifact birlikte yeniden üretilir ve birlikte kullanılmalıdır; Git ikisini de dışlar.

### Gerçek çalışma ve Phase 4 karşılaştırması

873 `customer_text` için model 384 boyutlu görüşme vektörleri üretti: matris **(873, 384)**. Model sınırını aşan **849** döküm, toplam **3.053** parçaya ayrıldı; bir görüşmede en fazla 10 parça vardı. Her parçanın yeniden tokenleştirilmiş uzunluğu sınır altında kontrol edildi. Sanity check'te internet bağlantısını anlatan A–B cosine similarity **0.6723**, internet ile uçak koltuğunu anlatan A–C **0.1817** çıktı; A–B daha yüksek. Bu küçük kontrol bir accuracy testi değildir.

| K | Phase 5 cosine silhouette | En küçük küme | En büyük küme |
|---:|---:|---:|---:|
| 8 | 0.1745 | 40 | 200 |
| 12 | 0.1944 | 39 | 169 |
| 16 | 0.2070 | 35 | 75 |
| 20 | 0.2107 | 8 | 69 |
| 24 | 0.1927 | 14 | 62 |
| 30 | 0.1773 | 7 | 69 |

**K=16** seçildi: K=20'nin silhouette değeri yalnızca 0.0037 daha yüksek, fakat en küçük kümesi 8 görüşme. K=16'da bütün kümeler 35–75 aralığında; 16 kümenin terim ve örneklerini elle incelemek 20 kümeye göre daha kolay. Kod, en az 5 üyeli ve en büyük kümesi toplamın en çok %20'si olan adaylardan en iyi silhouette değerine 0.005 yakın en küçük K'yi seçer. Bu bir keşif tercihidir, gerçek problem sayısının kanıtı değildir.

| Çalışma | Temsil | Seçilen K | Cosine silhouette | Küme aralığı | Gözlemsel yorum |
|---|---|---:|---:|---:|---|
| Phase 4 | Tam metin TF-IDF | 16 | 0.0451 | 19–127 | Bazı belirgin terimler var; 7, 9 ve 14 numaralı kümeler farklı konuları veya sayı/dolgu sözcüklerini karıştırıyor. |
| Phase 5 | Tüm metnin parça ortalamalı MiniLM embeddingi | 16 | 0.2070 | 35–75 | Daha dengeli ve birçok kümeyle ilgili metin örneği var; yine de karışık kümeler sürüyor. |

Phase 5'te örneğin cluster 6 internet hızı/modem metinlerini (**38 telecom / 48**), cluster 7 laptop onarımını (**37 technology / 39**), cluster 11 kaza/sigorta talebini (**40 insurance / 41**) topluyor. Cluster 4 ise **31 banking ve 15 hospitality** dahil 11 domaini birleştiriyor; para transferi, hesap ve rezervasyon iptali/geri ödeme ifadeleri aynı gruba düşebiliyor. Cluster 8 **33 banking ve 27 finance** görüşmesini birleştiriyor. Bu ortak konu sözcükleri veya semantik yakınlık, aynı somut müşteri problemi anlamına gelmez. Phase 5'in silhouette değeri daha yüksek, ancak skorlar **farklı temsil uzaylarında** hesaplandı; gerçek issue etiketleri veya insan doğrulaması olmadan “embedding kesin daha iyi problem kümeleri buldu” sonucu çıkaramayız.

Betik seçilen her kümenin sayısını, betimleyici üst terimlerini, domain dağılımını ve merkeze cosine olarak en yakın üç gerçek müşteri metninin kısaltılmış önizlemesini terminalde gösterir. Son çevrimdışı test sonucu: **20 passed**.

## Phase 6: sentetik zamanda erken uyarı denemesi

**Önemli: AppTek verisinde gerçek görüşme zaman damgası yoktur.** Bu bölüm gerçek geçmişte bir olay yaşandığını veya SignalDesk'in canlı izleme yaptığını iddia etmez. Gerçek `customer_text` kayıtları ve Phase 5'in semantic cluster atamaları korunur. Yalnızca **temporal evaluation katmanı sentetiktir**: saatler ve kontrollü artış bellekte üretilir; yeni müşteri metni veya yeni görüşme oluşturulmaz.

```text
Gerçek müşteri görüşmeleri
        ↓
Hazır modelden semantic embeddingler
        ↓
Semantic kümeler (K=16; doğrulanmış problem etiketi değil)
        ↓
SENTETİK demo zaman damgaları
        ↓
Saatlik küme sayıları
        ↓
Yalnızca önceki saatlerden baseline
        ↓
Anomaly score
        ↓
Açıklanabilir erken uyarı denemesi
```

**Event stream (olay akışı)** burada her gerçek görüşmenin küme ID'si ve *sentetik* zamanından oluşur. **Time series (zaman serisi)** bu olayların saat sırasındaki sayılarıdır. **Time bucket (zaman dilimi)** bir saatlik aralıktır; hiç görüşme olmayan saat de 0 sayılır. **Baseline**, bir kümenin yakın geçmişteki olağan sayısıdır. **Rolling mean (kayan ortalama)** önceki 12 saatlik sayıların ortalaması; **rolling standard deviation (kayan standart sapma)** bu geçmiş sayıların ne kadar değişken olduğudur. **Z-score**, mevcut saatin geçmiş ortalamadan kaç standart sapma uzaklaştığını yaklaşık gösterir. Standart sapma 0 veya çok küçükse bölme işlemi için 1,0 alt sınırı kullanılır. Bu koruma düşük hacimli kümelerde skoru etkiler ve gerçek istatistiksel belirsizliği çözmez.

**Anomaly detection (anomali tespiti)** beklenmedik sayı artışını işaretler. **Emerging issue**, gerçekten yeni veya hızla yaygınlaşan belirli bir müşteri problemi olabilir; her anomali emerging issue değildir. Örneğin mevsimsel kampanya veya rastlantısal yoğunluk da artış yaratabilir. **False positive (yanlış alarm)**, incelendiğinde gerçek ve eyleme değer bir problem artışı çıkmayan uyarıdır. Bu basit kurala göre uyarı için **en az 12 geçmiş saat**, **mevcut saatte en az 8 görüşme** ve **z-score en az 3,0** birlikte gerekir. Asgari sayı, 0–1 civarındaki küçük oynamaların alarm üretmesini azaltır. Artış oranı `current / historical_mean` yalnızca yardımcı açıklamadır; ortalama 0 ise `N/A` gösterilir ve bu oran tek başına alarm kararı vermez.

**Data leakage (veri sızıntısı)** burada mevcut saatin sayısını kendi normal ortalamasına katmakla oluşurdu. Kod geçmiş dizisini `series[index - 12:index]` mantığıyla alır: sağ uçtaki mevcut saat hariçtir. Önceki bir uyarı saati daha sonraki saatin geçmişine girebilir; 14:00 değerlendirmesinde 13:00 artık gerçekten geçmiş olduğu için bu doğrudur. Baseline'ın sıçrama sonrası yükselmesi ikinci uyarının skorunu düşürebilir.

**Synthetic timestamp**, gerçek zaman alanı olmadığı için algılayıcı mantığını deneyebilmemizi sağlar. Sabit seed 42 ile her görüşmeye 1–2 Ocak 2026 UTC içindeki 48 saatten bir demo zamanı verilir. **Synthetic surge injection**, yeterli kaydı olan en büyük kümeyi seçip 30 *mevcut* görüşmenin demo zamanını son iki saate taşır; metin, embedding ve cluster ataması aynı kalır. Bu yapay artış kolayca yakalanması amaçlanan kontrollü bir deneydir; normal trafikteki gerçek false positive oranını ölçmez. Bir küme de doğrulanmış problem türü olmadığı için uyarı henüz gerçek bir “çözülmeyen sorun” anlamına gelmez.

### Gerçek kod koşusundaki sentetik demo sonucu

- Görüşme: **873**; semantic küme: **16**. Sentetik aralık: **2026-01-01 00:00–2026-01-02 23:00 UTC**, saatlik bucket.
- Baseline: önceki **12 saat**; minimum geçmiş **12 saat**; standart sapma alt sınırı **1,0**. Uyarı: **z ≥ 3,0** ve **en az 8 olay**.
- En büyük uygun küme **cluster 5** seçildi (**75** gerçek görüşme). **30** mevcut olay son iki saate taşındı. Betimleyici terimler: `size, store, jacket, return, perfect, exchange, order, fit`.
- Enjeksiyon öncesi kontrol akışında **0** uyarı vardı. Enjeksiyon sonrası **2** uyarı: ikisi cluster 5'te, diğer kümelerde **0**. **Injected surge detected: YES**.
- **2026-01-02 22:00 UTC:** 16 olay; geçmiş ortalama **0,667**, standart sapma **0,943**, z-score **15,333**, artış oranı **24,00×**.
- **2026-01-02 23:00 UTC:** 16 olay; geçmiş ortalama **1,917**, standart sapma **4,349**, z-score **3,239**, artış oranı **8,35×**.

Cluster 5'in gerçek örnekleri sezonluk ürün stok sorgusu, kıyafet bedeni ve iade gibi farklı perakende konularını içerebilir. Bu yüzden `size/jacket/return` terimleri kümeyi kesin bir “iade problemi” etiketi yapmaz. Ayrıca sentetik zamanlar gerçek trafik ritmini, toplam çağrı hacmindeki değişimi, hafta/gün mevsimselliğini veya olay çözüm durumunu temsil etmez. 12 saatlik basit baseline ve kontrollü enjeksiyonla elde edilen iki uyarı **production monitoring başarısı** değildir. Gerçek zaman damgalı görüşmeler ve insan doğrulamalı problem grupları olmadan erken uyarı kalitesi ölçülemez.

## Phase 7: FastAPI ile model serving temeli

**API (uygulama programlama arayüzü)**, başka bir programın SignalDesk'ten belirli bir biçimde sonuç istemesidir. **REST API**, bu örnekte HTTP adresleri ve yöntemleriyle çalışan basit arayüzdür. **Endpoint**, çağrılan adres ve yöntem ikilisidir: `GET /health` ile `POST /api/v1/analyze` farklı işlemlerdir. **HTTP GET**, bilgi okumak için kullanılır; `/api/v1/info` yapılandırmayı, `/api/v1/alerts/demo` sentetik uyarı sonucunu okur. **HTTP POST**, sunucuya yeni bir analiz girdisi gönderir; `/api/v1/analyze` verilen müşteri metnini analiz eder, fakat veritabanına kaydetmez.

**Request (istek)**, istemcinin yolladığı HTTP çağrısıdır. Analyze isteğinin gövdesi örneğin `{"customer_text":"My internet keeps disconnecting."}` biçimindedir. **Response (yanıt)**, sunucunun döndürdüğü sonuç ve HTTP durum kodudur. **JSON**, bu anahtar/değer yapısının hem istek hem yanıt için kullanılan metin biçimidir. Analyze yanıtındaki `domain`, `semantic_cluster` ve `cluster_similarity` değerleri gerçek Phase 3/5 modellerinden hesaplanır.

**FastAPI**, Python fonksiyonlarını HTTP endpointlerine bağlar ve Pydantic ile veri şekillerini denetler. **Uvicorn**, FastAPI uygulamasını yerel HTTP sunucusu olarak çalıştırır. **Pydantic validation**, gelen `customer_text` alanının bir metin olmasını ve yalnızca boşluk içermemesini denetler; yanlış istek model koduna geçmez. **HTTP 200** isteğin başarıyla işlendiğini, **422** gönderilen verinin şemaya uymadığını, **503** gerekli yerel artifact veya hazır modelin kullanılamadığını gösterir. Beklenmeyen işlem hataları güvenli **500** yanıtı verir; istemciye stack trace, yerel yol veya ortam bilgisi gönderilmez.

**Model serving**, daha önce hazırlanmış modeli yeni girdiler için erişilebilir tutmaktır. Bu **inference API** yeni bir müşteri metninin domainini tahmin eder ve semantic kümesini belirler; burada yeniden eğitim veya K-Means fit işlemi yoktur. Phase 7'de domain modeli `customer_text + agent_text` ile öğrenilmişken API yalnızca `customer_text` alıyordu. Bu girdi farkının Phase 9'da nasıl ele alındığı aşağıda anlatılıyor. **Lazy loading**, modeli ilk ihtiyaç duyulduğunda yüklemektir; `/health` ve `/api/v1/info` modelleri yüklemez. **Caching**, yüklenen Python nesnesini sonraki isteklerde yeniden kullanmaktır. Böylece her istekte diskteki joblib dosyasını veya büyük MiniLM modelini tekrar okumayız. Servis yeniden başlarsa önbellek de yeniden başlar.

Yeni metin Phase 5 ile **aynı** parçalara ayırma ve vektör birleştirme işlevinden geçer. Kaydedilmiş fitted K-Means `predict()` ile küme seçer; request sırasında 873 görüşmeyi yeniden kümelemez. `cluster_similarity`, yeni metin vektörü ile seçilen merkezin cosine similarity değeridir; **probability değildir**. `cluster_descriptive_terms`, aynı Phase 5 koşusunda ayrı TF-IDF analizinden üretilen açıklayıcı sözcüklerdir; **semantic cluster doğrulanmış issue label değildir**. `domain_confidence` de kalibre edilmiş kesinlik değildir. Phase 6 endpointi `temporal_mode: "synthetic_demo"` ve `is_real_time: false` döndürür; gerçek zamanlı izleme değildir.

```text
Client
  ↓ HTTP request
FastAPI route
  ↓ Pydantic validation
SignalDesk service layer
  ├── Phase 3 domain classifier (yerel artifact)
  ├── Phase 5 hazır MiniLM embedding modeli
  ├── Phase 5 fitted semantic clusterer + descriptive metadata
  └── Phase 6 structured synthetic early-warning demo
  ↓
JSON response
```

### Yerel artifact'leri üretme ve API'yi çalıştırma

Proje kökünde PowerShell açın. İlk iki analiz betiği Hugging Face verisine erişir; MiniLM'nin ilk kullanımı model indirmeyi gerektirebilir. Artık bir kez üretilen yerel artifact'ler servis başlangıcında gerekmez, yalnızca ilgili endpoint ilk çağrıldığında okunur.

```powershell
cd "C:\Users\Rana\Documents\ChatGPT\call center\signaldesk-ai"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m signaldesk.ml.train_domain_classifier
python -m pytest -q
python -m uvicorn signaldesk.api.main:app --reload
```

Mevcut Phase 5 K-Means/metadata ve Phase 6 demo artifact'leri yerindeyse onları yeniden üretmeniz gerekmez. Eksiklerse yukarıdaki ilgili Phase 5/6 komutlarıyla oluşturun. `/ready` yerel analiz artifact'lerinin durumunu söyler; sentetik demo dosyası ayrı endpoint için gereklidir.

Phase 9'a yükseltirken eski müşteri+temsilci domain artifact'ini kullanmaya devam etmeyin; `train_domain_classifier` komutunu yeniden çalıştırarak müşteri metni sözleşmesine uygun artifact üretin. Phase 5 betiği `semantic_clusterer.joblib` ve `semantic_cluster_metadata.json` dosyalarını **aynı koşuda** üretir; bunlardan birini başka koşudaki dosyayla eşleştirmeyin, çünkü küme ID'leri değişebilir. Phase 6 betiği küçük `early_warning_demo.json` dosyasına yalnızca sentetik zamanlı uyarı ölçülerini yazar; müşteri transcripti yazmaz. Bu üç yerel dosya ve domain modeli `artifacts/` altında tutulur ve Git tarafından ignore edilir. `.joblib` dosyalarını yalnızca kendi çalıştırdığınız güvenilir betikten yükleyin. Dosyalar eksikse ilgili endpoint 503 ile üretme komutunu söyler; uygulamanın açılması ve `/health` çalışması için bu dosyalar gerekmez.

Sunucu çalışırken ayrı bir PowerShell'de:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/info
$body = @{ customer_text = "My internet keeps disconnecting every few minutes and the modem keeps losing connection." } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/analyze -Method Post -ContentType "application/json" -Body $body
Invoke-RestMethod http://127.0.0.1:8000/api/v1/alerts/demo
```

Tarayıcıda [Swagger arayüzünü](http://127.0.0.1:8000/docs) açabilirsiniz. **OpenAPI**, endpointlerin istek ve yanıt şemalarını açıklayan makine tarafından okunabilir tanımdır; **Swagger UI** bunu sayfa olarak gösterir. Böylece endpointleri görebilir, örnek request gönderebilir ve response'u frontend yapmadan inceleyebilirsiniz. Bu API'de henüz authentication, kalıcı veritabanı veya real-time event ingestion yoktur; herkese açık CORS da eklenmedi. `--reload` geliştirme içindir.

## Phase 8: görüşmeden yapılandırılmış müşteri problemi çıkarma

Phase 5, müşterinin uzun ve dolgu ifadeleri içeren dökümünü doğrudan embedding'e verdi. Bu, bazı kümelerde belirli problem yerine geniş hizmet alanını yakalayabiliyor. Phase 8'de **aynı görüşmeler için** önce kısa bir `issue_statement` üretiyoruz, sonra MiniLM ve K-Means'e yalnızca bu ifadeyi verip kontrollü karşılaştırma yapıyoruz. Kaynak `customer_text` aynen korunur. Bu çalışma yalnızca çevrimdışı deneydir; Phase 7'nin `POST /api/v1/analyze` davranışı değiştirilmedi.

```text
Phase 5: customer_text → MiniLM → embedding → K-Means

Phase 8: customer_text → FLAN-T5 → issue_statement
                                      ↓
                                  MiniLM
                                      ↓
                              384-D embedding
                                      ↓
                                  K-Means
                                      ↓
                                issue cluster
```

**LLM (büyük dil modeli)**, çok sayıda metin örneğinden dil örüntüleri öğrenmiş model ailesidir; burada kullandığımız küçük FLAN-T5 de dil üreten bir transformer modelidir. **Generative model**, verilen girdiye karşılık yeni metin üretir: müşteri görüşmesinden örneğin “modem connection keeps dropping” gibi bir cümle yazması beklenir. **Transformer**, metindeki tokenların birbirleriyle ilişkisini işlemeye yarayan model mimarisidir. **Token**, tokenizer'ın metni böldüğü küçük sayısal birimdir; bir sözcük bir veya birkaç token olabilir. **Tokenizer**, metni modelin işleyebileceği token ID'lerine çevirir ve çıktıyı yeniden metne dönüştürür. **Hugging Face Transformers**, hazır model ve tokenizer'ı yükleyip `generate()` ile yerelde çalıştırmamızı sağlayan Python kütüphanesidir.

**Prompt**, modele verilen açık yönerge ve müşteri metnidir. `build_prompt()` aynı kısa yönergeyi her görüşme için kurar: ana problemi tek cümleyle belirtmesini, çözüm/temsilci yanıtı eklememesini ve bilgi uydurmamasını ister. **Inference**, önceden öğrenilmiş ağırlıkları kullanarak yeni girdi için çıktı üretmektir. **Pretrained model**, ağırlıkları önceden hazırlanmış modeldir. **Fine-tuning**, bu ağırlıkları bizim veri kümesinde yeniden eğitmek olurdu. **Phase 8 does not train or fine-tune FLAN-T5.** [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) seçildi: yönerge takip eden, yerelde CPU ile çalıştırılabilen küçük text-to-text modelidir; büyük model veya bulut API gerektirmez. Yine de 873 uzun görüşmede CPU çalışması zaman alır.

**FLAN-T5 ile MiniLM farklı iş yapar:** FLAN-T5 yeni bir issue cümlesi **üretir**; MiniLM metni 384 boyutlu sayısal **embedding** vektörüne çevirir. Text generation çıktısı okunabilir kelimelerdir, embedding boyutları ise tek tek kelime veya doğrulanmış problem etiketi değildir. FLAN-T5'in küçük olması karmaşık konuşmaları kaçırmasına yol açabilir. **Hallucination (uydurma)**, modelin dökümde bulunmayan bir problem ayrıntısı yazmasıdır. Ayrıca gerçekten söylenen önemli bir bilgiyi atlayabilir. Bu nedenle üretilmiş `issue_statement`, **ground truth (doğrulanmış doğru cevap)** değildir; insan incelemesi gerekir.

**Deterministic generation** için sampling kapalı (`do_sample=False`), greedy çözümleme (`num_beams=1`) ve üst çıktı sınırı 32 token kullanılır. Temperature verilmez. Aynı model, girdi ve ortamda sonuçları tekrar üretmeye yardımcı olur; farklı kütüphane/model sürümleri veya donanım mutlak bit eşitliği garantilemez. Modelin 512 tokenlık konum sınırının altında, bu deney için bilinçli bir **128 token girdi penceresi** kullanılır. Betik `calling about`, `looking for`, `I'd like to` gibi ilk açık istek ifadesini bulursa model girdisini o ifadenin hemen önünden başlatıp en fazla yaklaşık 45 sözcüklük parçayı alır; böyle bir ifade yoksa konuşmanın başından ilerler. Bu basit seçim yanlış ifadeye odaklanabilir veya daha sonra söylenen asıl sorunu kaçırabilir. Tokenizer, pencereyi **truncation kapalıyken** ölçer, prompta yer ayırır ve gerçekten modele verilen son dizinin sınıra sığdığını denetler. Kaynak transcript aynen korunur; model girdisinden bir kısım çıkarıldıysa `input_truncated=true` kaydedilir. Model boş, yalnız dolgu/sayı kodu içeren, çok kısa veya açıkça genel yanıt kalıbı olan çıktı verirse ilk seçilen kaynak cümleden en çok 180 karakterlik alıntı kullanılır ve `issue_source="fallback"` yazılır; bu FLAN-T5 üretimi gibi sunulmaz. Diğer kayıtlarda `issue_source="flan_t5"` olur. Çıktıya yalnızca baş/son boşluk ve fazla boşluk temizliği uygulanır; stemming, stopword çıkarma veya agresif temizleme yapılmaz.

### Dosyalar ve yeniden üretme

- `src/signaldesk/issues/__init__.py`: issue çıkarımı paketini tanımlar.
- `src/signaldesk/issues/extract_issues.py`: `build_prompt()` yönergeyi tek yerde tutar. `prepare_prompt()` token sınırını kontrol eder. `IssueExtractor` tokenizer/modeli ilk kullanımda yükleyip aynı nesneleri sonraki batch'lerde kullanır. `select_domain_sample()` 10 farklı domainden tekrarlanabilir örnek seçer. `extract_conversations()` özgün metni ve `issue_source` alanını kayda koyar. `write_artifact()` yalnız tam koşuda yerel JSON yazar; `load_artifact()` eksik veya uyumsuz dosyayı reddeder.
- `src/signaldesk/clustering/discover_extracted_issues_semantic.py`: kayıtların **yalnızca `issue_statement`** alanını Phase 5'in aynı MiniLM, chunking, K adayları, K-Means ve seçim kuralıyla kümelemeye verir. Üst terimler ayrı TF-IDF hesabıyla kümeleme bittikten sonra bulunur; domain yalnızca yorumlama içindir.
- `tests/test_issue_extraction.py` ve `tests/test_extracted_issue_clustering.py`: sahte tokenizer/model ve küçük sentetik kayıtlarla promptu, sınırı, fallback'i, örnek seçimini, artifact'i ve doğru kümeleme girdisini internetsiz denetler.
- `requirements.txt`: doğrudan kullanılan `transformers` paketini ekler. PyTorch zaten `sentence-transformers` bağımlılığıyla kurulduğu için tekrar eklenmedi.
- `.gitignore`: `artifacts/extracted_issues.json` dosyasını Git dışında tutar. Bu JSON tam müşteri dökümlerini de içerir; repoya commit edilmez.

Proje kökünde PowerShell:

```powershell
cd "C:\Users\Rana\Documents\ChatGPT\call center\signaldesk-ai"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest -q
python -m signaldesk.issues.extract_issues --sample
python -m signaldesk.issues.extract_issues --full
python -m signaldesk.clustering.discover_extracted_issues_semantic
```

İlk örnek koşusu Hugging Face'ten public veri ve hazır model indirebilir; API key gerekmez. `--sample` dosya yazmaz. Örneklerin üretilen problem cümlelerini gözle inceleyin, sonra `--full` ile 873 kaydı `artifacts/extracted_issues.json` içine yeniden üretin. Dosyada model, prompt sürümü ve generation ayarları da bulunur. Son komut yalnız bu artifact'i okuyarak issue tabanlı kümeleme yapar; FLAN-T5'i tekrar çalıştırmaz. Kümeleme etiketsizdir: silhouette temsil uzayındaki ayrımı ölçer, gerçek problem doğruluğunu ölçmez. `domain` etiketleri issue etiketi değildir. Phase 6 zaman damgaları hâlâ sentetiktir. Bu bir production LLM sistemi değildir.

### Gerçek veri koşusu ve örnek inceleme

Bu makinede PyTorch CPU sürümüyle son `--full` koşusu, **model yüklemesi dahil 119,80 saniye** sürdü. **873** görüşmenin **610** `issue_statement` değeri FLAN-T5 çıktısı, **263** değeri açıkça işaretli kaynak metin fallback'i oldu. **0** boş issue var. 128 tokenlık odak penceresi nedeniyle **873** kaydın hepsinde `input_truncated=true`; bu, özgün `customer_text` alanının kesildiği anlamına gelmez. Artifact yaklaşık **2,63 MB** ve Git dışında. Bu sayılar çıkarımın çalıştığını gösterir, ifadelerin doğru problem etiketi olduğunu kanıtlamaz.

Sabit tohumlu 10-domain örnek incelemesinden bazı sonuçlar (müşteri metninin yalnız kısa açıklaması; tam metin `--sample` terminal çıktısında görünür):

| Domain | Kaynak müşteri talebinin kısa görünümü | Issue statement | Kaynak |
|---|---|---|---|
| agriculture | Kuraklığa dayanıklı tohum stokunu soruyor | `i need to know if you have any of those drought resistant seed` | flan_t5 |
| aviation | Diz ameliyatı nedeniyle daha fazla bacak alanlı koltuk istiyor | `I'd like to request a seat with extra leg leg leg room` | flan_t5 |
| banking | Hesaplar arasında otomatik aylık transfer kurmak istiyor | `I want to set up an au automatic monthly transfer between the two` | flan_t5 |
| deliveryservice | Paket iki gün önce gelmeliydi | `a package that I had purchased was supposed to arrive two days ago` | flan_t5 |
| energy | Yeni eve elektrik hizmeti istiyor | `I'm just looking for a new electricity service as a new home owner.` | fallback |
| entertainment | VIP geçiş bilgisini çevrim içi bulamıyor | `passes, (uh) can't find anything about that online, just the basic ticket.` | fallback |
| finance | Emeklilik yatırım hesabı açmayı soruyor | `How to open a retirement savings account` | flan_t5 |
| food | Doğum günü için deluxe yemek paketi soruyor | `a deluxe package` | flan_t5 |
| health | Doktorla takip randevusu istiyor | `I, I'm calling about (uh) sc~ scheduling a follow-up appointment with my doctor, Dr Payne.` | fallback |
| hospitality | Erken giriş istiyor | `was just wondering if I could request an early check-in.` | fallback |

Bu 10 örnekte 6 model çıktısı ve 4 fallback vardı. `flan_t5` alanı yalnızca çıktının kaynağını belirtir; örneğin `leg leg leg` ve `au automatic` tekrarları kalite kusurudur. İlk denemelerde küçük model zaman zaman “No, I'm not sure” gibi problem olmayan yanıtlar verdi; açık kalıplar fallback'e yönlendirildi. Açıkça geçersiz olmayan ama yanlış/eksik bir ifade hâlâ kalabilir. Otomatik doğrulayıcı gerçek issue doğruluğunu ölçemez.

### Aynı K adaylarıyla kümeleme karşılaştırması

Issue metinleri MiniLM ile **(873, 384)** vektöre dönüştürüldü. Aşağıdaki sayılar son artifact ve aynı Phase 5 aday ayarlarıyla gerçek koşudan geldi:

| K | Cosine silhouette | En küçük küme | En büyük küme |
|---:|---:|---:|---:|
| 8 | 0.0782 | 68 | 192 |
| 12 | 0.0804 | 27 | 159 |
| 16 | 0.0951 | 21 | 105 |
| 20 | 0.0972 | 19 | 69 |
| 24 | 0.1001 | 16 | 82 |
| 30 | **0.1074** | 11 | 50 |

Phase 5 ile aynı denge kuralı uygulandı: en küçük küme en az 5 kayıt, en büyük küme toplamın en çok %20'si; uygun en iyi silhouette değerinin 0.005 yakınındaki **en küçük K** seçilir. Bu koşuda **K=30**, çünkü K=24'ün 0.1001 skoru en iyi 0.1074'ün 0.005 altında değil. Seçilen küme boyutları **11–50**. K=30 gerçek issue türü sayısı değildir.

Her seçili kümeden ilk betimleyici terimler, en çok görülen domainler ve merkeze yakın bir **gerçek issue_statement** aşağıdadır. Bunlar otomatik doğrulanmış konu adları değildir. Betik çalıştırıldığında her küme için üç temsilciyi ve daha fazla terimi basar.

| Küme | Boyut | Betimleyici terimler | Baskın domainler | Temsilci issue statement |
|---:|---:|---|---|---|
| 0 | 19 | event, special, small | food 8; hospitality 5 | `a small office event` |
| 1 | 29 | plan, like, want | finance 7; telecom 7 | `I'd like to upgrade my plan` |
| 2 | 27 | cancel, reservation, hotel | hospitality 16; travel 4 | `I need to cancel the reservation.` |
| 3 | 30 | catering, corporate, party | food 29; hospitality 1 | `I'm looking for catering services for a birthday party.` |
| 4 | 29 | appointment, schedule, doctor | health 28; food 1 | `I'm calling to schedule up a follow-up appointment with my doctor.` |
| 5 | 50 | account, savings, transfer | banking 42; finance 6 | `id like to transfer my savings to my checking account` |
| 6 | 36 | supposed, early, arrive | deliveryservice 5; hospitality 5 | `It was supposed to arrive yesterday.` |
| 7 | 36 | service, internet, electricity | energy 16; telecom 10 | `I was calling to make sure that I could get new service at my new home.` |
| 8 | 35 | issue, didn't, work | retail 7; technology 7 | `No, it's not working.` |
| 9 | 44 | need, want, size | retail 9; energy 4 | `I need them` |
| 10 | 17 | organic, farm, certification | agriculture 17 | `if we would like to do a farm that is totally organic?` |
| 11 | 13 | irrigation, water, drip | agriculture 12; energy 1 | `I'm just calling about the (uh,) subsidies for drip drip irrigation.` |
| 12 | 37 | laptop, problem, crashing | technology 19; retail 6 | `I have a problem with my laptop.` |
| 13 | 44 | tickets, vip, passes | entertainment 40; travel 2 | `i'm looking for a ticket for the VIP seating` |
| 14 | 22 | package, send, delivery | deliveryservice 22 | `I need to send the package to my cousin in (uh) other city.` |
| 15 | 38 | calling, jacket, return | retail 21; deliveryservice 14 | `I'm just calling because I've got a request to return (uh) a jacket I recently purchased.` |
| 16 | 21 | drive, house, london | realestate 6; energy 4 | `I'm just calling to inquire about an apartment in the Glasgow area` |
| 17 | 20 | flight, change, date | aviation 16; hospitality 2 | `I need to change my return flight to a later date.` |
| 18 | 16 | vegetarian, meal, request | food 6; aviation 5 | `i want to request a vegetarian meal` |
| 19 | 34 | apartment, rental, bedroom | realestate 28; travel 3 | `Hello, I'm looking for a furnished apartment.` |
| 20 | 11 | seat, change, upgrade | aviation 8; entertainment 2 | `i need to change my seat` |
| 21 | 18 | insurance, claim, car | insurance 17; aviation 1 | `I'm calling to file a new car insurance claim.` |
| 22 | 41 | looking, stock, buy | retail 20; deliveryservice 4 | `I was looking for is out of stock.` |
| 23 | 37 | need, pay, charge | finance 7; banking 6 | `I need to pay the bills` |
| 24 | 50 | sure, sorry, issue | entertainment 6; retail 5 | `I'm not sure what I'm doing.` |
| 25 | 29 | attend, anymore, make | entertainment 11; hospitality 4 | `I can't attend anymore.` |
| 26 | 13 | room, upgrade, special | hospitality 8; insurance 2 | `I need to upgrade my room` |
| 27 | 21 | condition, doctor, medical | health 10; insurance 3 | `I have a chronic condition` |
| 28 | 38 | book, package, vacation | travel 25; aviation 4 | `I want to book a vacation package to the Biggs Theme Park` |
| 29 | 18 | add, extra, bag | aviation 11; insurance 3 | `I need to add a second bag.` |

| Karşılaştırma | Phase 5: ham `customer_text` | Phase 8: çıkarılmış `issue_statement` |
|---|---|---|
| MiniLM ve K-Means girdisi | Tam konuşma, token parçalarının ortalaması | Kısa üretilmiş ifade veya açık fallback alıntısı |
| Seçilen K | 16 | 30 |
| Seçilen cosine silhouette | **0.2070** | **0.1074** |
| Küme boyutu | 35–75 | 11–50 |
| Belirgin örnek | Perakende kümesinde beden/stok/iade birlikte | Otel iptali (2), uçuş tarihi değişikliği (17), ek bagaj (29) daha dar görünüyor |
| Sorunlu örnek | Geniş domain/konu karışımları | Genel “I'm not sure” cümleleri küme 24'te; 8, 9 ve 15 de farklı sorunları karıştırıyor |

Issue ifadeleri bazı dar talepleri görünür kılıyor; örneğin koltuk değişikliği ile ek bagaj farklı kümelerde. Buna karşılık **seçilen silhouette daha düşük** ve K=16'da issue temsili 21–105 ile ham metnin 35–75 aralığından daha dengesiz. Kümelerin bir kısmı hâlâ geniş konuları veya modelin yanlış/genel çıktısını yakalıyor. **Farklı temsillerde silhouette farkı gerçek problem doğruluğu veya “LLM daha iyi” kanıtı değildir.** Doğrulanmış issue etiketleri ve insan değerlendirmesi olmadan hangi kümenin gerçek tekrarlayan problem olduğunu bilemeyiz. 263 fallback'in de kısa ama kısmen ham transcript alıntısı olduğunu unutmayın; sonuç saf FLAN-T5 çıktısı değildir.

## Phase 9: tutarlı analiz servisi

**Training (eğitim)** sırasında TF-IDF, eğitim metinlerinde hangi sözcüklerin bulunduğunu öğrenir; Logistic Regression bu sözcüklerden `domain` tahmini yapacak ağırlıkları öğrenir. **Inference/serving (çıkarım/sunma)** sırasında API yeni müşteri metnini bu hazır modelden geçirir. Tek bir API isteği modeli yeniden eğitmez. Modelin eğitimde gördüğü girdi türü ile API'de aldığı girdi türü farklıysa buna **training-serving skew** veya **training-serving mismatch** denir. Phase 7'de model müşteri ve temsilci metinlerini birlikte görürken API'de yalnız müşteri metni geliyordu. Müşteri tarafındaki sorunu erken anlamak istediğimiz için Phase 9 aynı TF-IDF + Logistic Regression yöntemini yalnız `customer_text` kullanacak şekilde yeniden değerlendirir ve eğitir. Önceki iki taraflı modelin sonuçları tarihsel karşılaştırma olarak korunur; farklı girdilerle elde edilen skorları aynı deneymiş gibi yorumlamayın.

Gerçek AppTek verisindeki **873 görüşme**, `random_state=42` ve domain oranlarını koruyan aynı **698 eğitim / 175 test** bölünmesiyle karşılaştırıldı. Her iki koşuda da TF-IDF ve Logistic Regression ayarları aynıydı; yalnızca metin girdisi değişti.

| Girdi | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Phase 3: `customer_text + agent_text` | 0.9257 | 0.9505 | 0.9093 | 0.9237 |
| Phase 9: yalnız `customer_text` | 0.8400 | 0.8450 | 0.7939 | 0.8031 |

Müşteri metniyle sonuç **daha düşük**; bunu gizlemiyoruz. Testte `finance` sınıfının 6 örneğinden **0**'ı, `telecom` sınıfının 8 örneğinden **4**'ü doğru bulundu. `finance → banking` ve `finance → food` karışıklıklarının her biri 2 kez, `telecom → retail` 2 kez görüldü. Bu küçük sınıf örnekleri genelleme garantisi vermez; domain çıktısı tek başına kesin karar olarak kullanılmamalıdır. Yine de ana API için müşteri metniyle eğitilmiş artifact seçildi, çünkü API temsilci yanıtını almadan müşteri tarafındaki sorunu analiz etmek üzere tasarlandı; giriş sözleşmesinin doğru olması ölçümü ve hatayı görünür kılar.

**Feature contract (girdi sözleşmesi)**, artifact'in hangi alanla eğitildiğini ve API'nin ona hangi alanı vereceğini açıkça belirtir. Buradaki sözleşme `input_mode = "customer_text"` olmalıdır. Eski `customer_text + agent_text` artifact'i yanlışlıkla kullanılırsa API sessizce tahmin üretmek yerine kontrollü bir kullanılabilirlik hatası vermelidir. Böyle bir kontrol, aynı dosya adını taşıyan iki farklı eğitim sürümünü güvenle ayırmaya yardımcı olur. **Model artifact**, eğitimin sonunda diske yazılmış TF-IDF sözlüğü ve sınıflandırıcı gibi hazır model durumudur; API onu yükler ve `predict()` için kullanır. K-Means artifact'i de 16 kümenin öğrenilmiş merkezlerini taşır. Küme numaralarının açıklayıcı terimleri ayrı metadata dosyasında tutulur; metadata ile K-Means aynı koşudan gelmelidir. Üretilen büyük dosyalar Git'e eklenmez.

**Liveness**, sunucu işleminin yanıt verebildiği anlamına gelir: `GET /health` bunun hafif kontrolüdür ve model yüklemez. **Readiness**, analiz için gerekli yerel bileşenlerin hazır olup olmadığını anlatır: `GET /ready` domain modeli, semantic K-Means artifact'i ve ona bağlı metadata gibi zorunlu parçaları kontrol eder. Hazırsa `200` ve `{"status":"ready"}`, eksik veya uyumsuz parça varsa güvenli açıklamayla `503` beklenir. Bu kontrol ağır müşteri metni çıkarımı veya K-Means yeniden eğitimi yapmaz. MiniLM'nin ilk kullanımı önbellekte model yoksa ayrıca indirme gerektirebilir; dolayısıyla hafif readiness kontrolü her olası çalışma zamanı hatasını önceden kanıtlamaz. `/health` başarılıyken `/ready` başarısız olabilir: süreç çalışıyor, fakat henüz analiz veremiyordur.

Tekli `POST /api/v1/analyze` yanıtında `domain`, `domain_confidence`, `semantic_cluster`, `cluster_similarity` ve `cluster_descriptive_terms` korunur. Ek `analysis_metadata`, kullanılan domain modelinin adını, `domain_input_mode` değerini, MiniLM modelini ve küme sayısını belirtir; yerel dosya yolu veya secret içermez. `domain_confidence`, Logistic Regression sınıflandırıcısının `predict_proba` çıktısında seçilen sınıfa verdiği skordur. **Kalibre edilmiş kesinlik** değildir: örneğin `0.8`, her on tahminden sekizinin doğru olacağını tek başına kanıtlamaz. `cluster_similarity`, yeni metnin MiniLM vektörü ile **atanan K-Means kümesinin merkezi** arasındaki cosine similarity değeridir. Vektörlerin yönsel yakınlığını ölçer; olasılık, doğrulanmış issue etiketi veya anomali skoru değildir. Geçersiz ya da sonlu olmayan değer güvenli biçimde reddedilmelidir.

**Batch inference**, birden fazla bağımsız müşteri metnini tek HTTP isteğinde analiz etmektir. `POST /api/v1/analyze/batch`, `items` dizisindeki her metin için tekli endpoint ile aynı sonuç yapısını `results` dizisinde, girdi sırasıyla döndürür. En fazla **50** öğeye izin verilir; boş dizi, sınırı aşan dizi ve yalnız boşluk içeren metin `422` ile reddedilir. Tek tek 50 HTTP isteği göndermek yerine bir batch göndermek ağ ve istek işleme maliyetini azaltabilir; MiniLM de metinleri birlikte encode edebilir. Bu işlem **streaming**, Kafka kuyruğu veya arka planda sürekli izleme değildir. Model ve artifact'leri batch içindeki her öğe için yeniden yüklemek yerine servis önbellekte tutar. Beklenmedik sunucu hataları istemciye stack trace, yerel yol, ortam değişkeni veya secret sızdırmadan güvenli `500` döndürür.

**API versioning**, adreslerdeki `/api/v1/` bölümüdür. İleride istek veya yanıt yapısı uyumsuz biçimde değişirse yeni bir sürüm açarak eski istemcilerin mevcut sözleşmeyi kullanmasına imkân verir. Bu sürüm numarası modelin eğitim sürümüyle aynı şey değildir. `/health` ve `/ready` işletim kontrolleridir; analiz endpointleri `/api/v1/` altında kalır.

### Ana sunum akışı ve ayrı deneyler

```text
Client
  ↓
FastAPI /api/v1
  ↓
Pydantic validation
  ↓
Analysis Service
  ├── customer_text → customer-only TF-IDF + Logistic Regression → domain
  └── customer_text → MiniLM embedding → K-Means K=16
                                          ↓
                              atanan merkezle cosine similarity
  ↓
Structured JSON response + analysis_metadata

Ayrı sentetik demo: GET /api/v1/alerts/demo → Phase 6 uyarı artifact'i
Ayrı çevrimdışı deney: customer_text → FLAN-T5 → issue_statement → MiniLM → K-Means
```

Ana API, Phase 5'teki **ham müşteri metni → MiniLM → K-Means K=16** akışını kullanır. Phase 8 FLAN-T5 çalışması ayrı bir çevrimdışı deney olarak korunur ve `/api/v1/analyze` içine bağlanmaz. Bu kararda yalnız silhouette sayısına bakılmadı: 873 kaydın 263'ünde fallback gerekmesi, bazı üretilmiş ifadelerin genel ya da tekrarlı olması, belirli sorunları daha dar gruplama kazanımı ve API'ye yeni bir üretici model koymanın işlem yükü birlikte değerlendirildi. Daha iyi gerçek issue tespiti iddiası için doğrulanmış etiket ve insan incelemesi gerekir. Erken uyarı demo zamanları da sentetiktir; dataset'te gerçek çağrı zaman damgası yoktur.

### Phase 9'u PowerShell'de çalıştırma

Proje kökünden çalışın. İlk artifact üretimi Hugging Face verisine, MiniLM'nin ilk kullanımı model indirmeye ihtiyaç duyabilir. Domain artifact'ini eski iki taraflı modelden müşteri metni sözleşmesine geçirmek için eğitim komutunu yeniden çalıştırın.

```powershell
cd "C:\Users\Rana\Documents\ChatGPT\call center\signaldesk-ai"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m signaldesk.ml.train_domain_classifier
python -m signaldesk.clustering.discover_issues_semantic
python -m signaldesk.monitoring.detect_emerging_issues
python -m pytest -q
python -m uvicorn signaldesk.api.main:app --reload
```

Sunucu açıkken **ayrı** bir PowerShell penceresinde aynı `PYTHONPATH` ayarını yapmanız gerekmez; HTTP isteklerini gönderin:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/api/v1/info

$single = @{ customer_text = "My internet keeps disconnecting every few minutes and restarting the modem does not fix it." } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/analyze -Method Post -ContentType "application/json" -Body $single

$batch = @{
    items = @(
        @{ customer_text = "My modem keeps disconnecting from the internet." },
        @{ customer_text = "I cannot transfer money between my bank accounts." },
        @{ customer_text = "I need to change the date of my flight." }
    )
} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8000/api/v1/analyze/batch -Method Post -ContentType "application/json" -Body $batch

Invoke-RestMethod http://127.0.0.1:8000/api/v1/alerts/demo
Start-Process http://127.0.0.1:8000/docs
```

`/docs` sayfası endpointleri ve istek/yanıt şemalarını etkileşimli gösterir. Batch yanıtının her öğesini kendi girdi sırasıyla karşılaştırın. Buradaki üç örnek sırasıyla telecom, banking ve aviation konusunda yazılmıştır; model tahmininin bu etiketleri kesin vermesi garanti değildir. API'de gerçek zamanlı olay alımı, kalıcı veritabanı, authentication veya canlı uyarı akışı bulunmaz.
