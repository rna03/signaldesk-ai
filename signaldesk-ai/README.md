# SignalDesk AI — Phase 1, 2, 3 ve 4

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
- `src/signaldesk/ml/train_domain_classifier.py`: doğrulanmış görüşmeleri yeniden `build_conversations()` ile oluşturur. `prepare_examples()` yalnızca iki transcripti birleştirip `domain` etiketini alır; kimlik ve konuşmacı metadata'sını model girdisine koymaz. `split_examples()` görüşmeleri sabit tohum ve `stratify` ile 80/20 ayırır. `make_pipeline()` TF-IDF ile Logistic Regression'ı birleştirir. `model.fit(x_train, y_train)` yalnızca train verisini görür. Test tahminleri, metrikler ve karışıklık matrisi yazdırılır; model yerel artifact olarak kaydedilir.
- `src/signaldesk/ml/predict_domain.py`: kaydedilmiş modeli yükleyip verilen metnin domain tahminini ve en yüksek sınıf skorunu gösterir. Yalnızca kendi eğittiğiniz, güvenilir yerel joblib dosyasını açın.
- `tests/test_domain_classifier.py`: metin/etiket hazırlamayı, görüşme ayrımını ve tahmin çıktısını sentetik verilerle sınar; gerçek veri indirmez.
- `src/signaldesk/clustering/__init__.py`: etiketsiz keşif kodunun Python paketi olduğunu belirtir.
- `src/signaldesk/clustering/discover_issues.py`: doğrulanmış görüşmeleri yeniden oluşturur, yalnızca `customer_text` ile TF-IDF matrisi üretir ve farklı K-Means adaylarını karşılaştırır. `make_vectorizer()` kaynak metni değiştirmeden sayısal özellik oluşturur. `cluster_counts()` dengeyi, `top_terms()` merkezde ağır basan sözcükleri, `representative_indices()` merkeze yakın gerçek görüşmeleri bulur. `select_candidate()` silhouette yakınlığını ve küme büyüklüğünü birlikte dikkate alır. `domain` yalnızca sonuçlar üretildikten sonra dağılım göstermek için okunur.
- `tests/test_issue_clustering.py`: sentetik müşteri metinleriyle TF-IDF, K-Means, sayım, üst terimler ve K seçimi yardımcılarını internet olmadan sınar.
- `requirements.txt`: bu aşamada gereken `datasets`, `pandas`, `pytest` paketlerini listeler. Pandas henüz veri işlemese de sonraki veri keşfi çalışmaları için erişimi test edilir.
- `.gitignore`: sanal ortam, Python önbelleği ve yerel `.env` dosyalarını Git dışında tutar.

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
python -m pytest -q
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
