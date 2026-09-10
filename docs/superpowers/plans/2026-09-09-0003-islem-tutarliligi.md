# Spec 0003 — İşlem Tutarlılığı Uygulama Planı

> Developer şapkası; plan inceleme/onay içindir. Uygulama kodu ve test kodu yazılmadı.
> Uygulayıcı: onay sonrasında `superpowers:executing-plans` ile dilim dilim ilerle; insan checkpoint'lerini koru.

**Goal:** Pozisyona göre işlem eylemini, geçmiş testi ve sanal takibi aynı onaylı iş kuralları altında tutarlı hâle getirmek.

**Architecture:** Mevcut Python/Streamlit yapısında veri uygunluğu, pozisyon/eylem kuralları ve gerçekleşme hesabı ayrılacak. Ortak değerlendirme akışı ekran, geçmiş test ve sanal takip tarafından kullanılacak; gerçek pozisyon yalnız kullanıcı teyidiyle değişecek. .NET/React dönüşümü veya yeni API planlanmıyor.

**Tech Stack:** Mevcut Python, Streamlit, pandas, pytest; arayüz kanıtı için mevcut test altyapısına uygun uygulama testi ve tarayıcı ekran görüntüsü.

**Spec:** `specs/0003-islem-kararlari-ve-performans-tutarliligi.md`, Revizyon 18; R01–R30 ve 104 kabul kriteri (AC14–AC20, AC26–AC122).

## Güncel plan etkileri

Revizyon 18: yeni AC105–AC122 mevcut veri uygunluğu, karar/gerçekleşme, kayıt, arayüz ve bütünleşik test dilimlerine aşağıdaki test matrisiyle eklenmiştir. Yeni üretim dosyası gerekmiyor. Performans süre hedefi ölçüm sonrası onaylanacaktır.

İş kurallarının tek kaynağı spec'teki R01–R30 ve AK kararlarıdır. Önceki revizyonlara ait tekrar eden delta notları kaldırıldı; kapanmış sorular yeniden onaya sunulmaz.

- Adım 1/4: R19–R23 uyarınca sabit başlangıç stopu, karar mumuna bağlı ATR, kullanıcı teyitli yükseltme, zarar sonrası iki mum bekleme ve maliyet revizyonundan bağımsız karar geçmişi uygulanır. Otomatik hedef satışı ve otomatik iz süren stop kapalıdır.
- Adım 3/4: sanal AL/SAT sonraki günlük açılıştadır; stop emri bu bekleme kuralından ayrılır. Sıradaki mum yoksa bekleyen sinyal gösterilir.
- Adım 4/5: gerçek işlem/stop güncellemelerinde kayıt zamanı ile geçerlilik zamanı ayrılır. Gerçek teyitli miktar/fiyat sanal varsayımlarla değiştirilmez.
- Adım 4/6: her varlık için bağımsız ve değiştirilebilir sanal tutarlar; komisyon hariç bütçe, ek ücret/nakit kontrolü, piyasa bazlı sonradan girilebilir makas/kayma ve bilinmiyor etiketleri uygulanır.
- Adım 4: miktar geçerli adıma aşağı yuvarlanır, bütçenin kalanı nakitte kalır. Adım bilinmiyorsa veya bütçe asgari miktara yetmiyorsa sanal giriş oluşturulmaz. Yuvarlama gerçek kullanıcı kaydını değiştirmez.
- Adım 2/7: kripto spot, BIST, altın ve Amerikan hisseleri günlük veriyle ayrı doğrulanır; özel eklenen semboller ve mevcut Yahoo/XAU eşlemeleri kapsamda kalır. Bir piyasanın başarısı diğerlerinin kanıtı sayılmaz.
- AK01–AK06 kapalıdır; AK05 sayısal süre hedefleri ölçüm sonrası onaylanmıştır.

## Kaynak durumu ve onay sınırı

Güncel spec diskten okundu; yalnız dosyada bulunan 104 AC resmi eşleme tablosundadır. Kaynak ekte bulunmayan eski AC numaraları geri eklenmedi. Devam eden CLARIFY kararlarıyla sonradan eklenen AC74–AC122 de tabloda yer alır; R01–R30'daki diğer ayrıntılar tamamlayıcı testlerle kapsanır.

`docs/architecture.md`, `docs/conventions.md`, `docs/testing.md` ve Developer rol kartı 2026-09-10 tarihinde diskten okundu; önceki eksik dosya durumu sona erdi. Belgeler .NET/React/xUnit iskeletini anlatıyor; bu feature'ın onaylı AK06 istisnası Python/Streamlit ve mevcut Python test altyapısıdır. Para/yüzde decimal, bağımsız test ve görsel kanıt kuralları geçerlidir; genel belge düzeltmesi ayrı iştir.

AK01–AK06, 0003 numarası, eski/teyitsiz kayıt geçişi ve uygulama planı onaylıdır. Uygulama bu plan üzerinden yürütülmüştür.

## Global Constraints

- V1 yalnız yol haritası adım 1–3; adım 4–7 sonraki spec'lere aittir.
- Günlük mumla kripto spot, BIST, altın ve Amerikan hisseleri; kullanıcının bu gruplarda eklediği varlıklar dahil. Diğer mevcut aralıklar kaldırılmaz, doğrulanmamış kapsam olarak gösterilir. Kaynak ve takvim sınırları piyasa bazında doğrulanır.
- Elle işlem, tam çıkış, gerçek emir gönderimi yok; yeni strateji, yeni gösterge ve eşik/ağırlık optimizasyonu yok.
- Para ve yüzde için konvansiyonlardaki decimal kuralı esas alınır. Analitik ara hesap istisnası konvansiyonlardaki onaylı sınırla uygulanır.
- Mevcut kullanıcı portföyü ve sanal işlem kayıtları test verisi olarak değiştirilmez; geçişler kopyalar üzerinde doğrulanır.
- Çalışma sırasında Telegram veya başka dış kanala gerçek mesaj gönderilmez; dış etkiler testte taklit edilir.
- Her AC bağımsız test adıyla eşlenir; görünür kriterlere ayrıca görsel kanıt eklenir.
- Uygulama öncesi spec numaralı izole branch/worktree; main'e doğrudan commit yok. Commit/PR onayı proje kurallarına göre alınır.

## Dosya haritası

### Eklenecek üretim dosyaları

| Yol | Sorumluluk |
|---|---|
| `trading_contracts.py` | Değerlendirme bağlamı, pozisyon durumu, eylem, veri/hata durumu ve maliyet/gerçekleşme varsayımlarının ortak tanımı; finansal hassasiyet sınırı. |
| `market_validation.py` | Kapanış takvimi, güncellik, geçerli geçmiş ve zorunlu alan/bileşen uygunluğu; açık mumla geçersiz veriyi ayırma. |
| `trade_decisions.py` | Pozisyona göre AL/BEKLE/SAT eylemi, koruma önceliği, ilk/güncellenmiş stop geçerliliği ve zarar sonrası bekleme. |
| `trade_execution.py` | Tetik ile gerçekleşmeyi ayırma, manuel/emir varsayımı, temas ve boşluklar, maliyet/nakit hesabı ve tek simüle işlem geçişi. |
| `trading_service.py` | Doğrulanmış bağlamla ortak sinyal/eylem/değerlendirme akışı; karşılaştırılabilirlik ve hata sonuçları. |
| `position_journal.py` | Gerçek işlem teyidi, tekrar güvenliği, düzeltme geçmişi; sanal durumdan ayrı kullanıcı pozisyon kaydı. |
| `trading_ui.py` | Karar paneli, pozisyon teyit/düzeltme akışı ve durum mesajları; app.py'nin ek iş mantığıyla büyümesini sınırlama. |
| `indicator_calculations.py` | Uygulama açılışındaki `pandas_ta`/numba gecikmesini kaldıran pandas tabanlı eşdeğer gösterge ara hesapları. |

### Değişecek mevcut dosyalar

| Yol | Planlanan değişiklik |
|---|---|
| `config.py` | Onaylı V1 kapsamı ve kuralları için tek ayar kaynağı; seçilmemiş AK değerleri yerine uydurma varsayılan konmaz. |
| `data_fetchers.py` | Kaynak, saat dilimi/kapanış bilgisi ve hata nedenini koruyan veri alımı; sessiz yanlış piyasa/aralık dönüşünü önleme. |
| `signal_engine.py` | Koşulsuz son satır atmayı kaldırma; doğrulanmış geçmişi kullanma; canlı/backtest durum geçmişi ve ML dahil bileşen kullanımını ortak değerlendirmeyle eşleştirme. |
| `technical_analysis.py` | `run_strategy_backtest`, çıkış ve iz süren stop hesaplarını ortak karar/gerçekleşme davranışına bağlama; BEKLE'de otomatik çıkış çelişkisini giderme. |
| `paper_trading.py` | `_step_book` ve güncellemeyi aynı değerlendirmeye bağlama; OHLC temasları, maliyetler, tekrar işleme ve kayıt hataları. |
| `portfolio.py` | Gerçek teyit akışını bağlama; `check_active_positions_auto_close` davranışını gerçek pozisyonu kendiliğinden kapatmayacak şekilde ayırma. |
| `app.py` | Yeni karar görünümü ve teyit akışını bağlama; geçmiş test/sanal takip varsayımlarını aktarma; otomatik gerçek pozisyon kapatma çağrılarını ayırma. |
| `theme.py` | Eylem, eski/bilinmeyen durum, koruma gerekçesi ve uyum puanı açıklamasının açık sunumu. |
| `scanner.py` | Karar bağlamı ve kapsam uygunluğu kullanımı; tarayıcı sonucunu kişisel koşulsuz işlem talimatı gibi sunmama. |
| `tests/conftest.py` | Sabit saat, kapanış sınırları, boşluklu OHLC ve ayrılmış gerçek/sanal pozisyon senaryoları; ağdan bağımsız veri. |
| `tests/test_signal_engine.py` | Değişen BEKLE/son mum davranışının regresyonları ve ortak sinyal geçmişi. |
| `tests/test_signal_cache.py` | Aynı kesit tekrarının ve veri düzeltmesinin doğru sonucu üretmesi. |
| `tests/test_portfolio.py` | Gerçek pozisyon teyidi ile eski otomatik kapatma davranışının ayrılması. |
| `tests/test_backtest_weights.py` | Aynı bileşen/ayar bağlamında backtest çalışmasının korunması. |
| `tests/test_scanner.py` | Kapsam etiketi ve bilgi/eylem ayrımı regresyonu. |
| `requirements-dev.txt` | Yalnız seçilen arayüz/görsel test aracı mevcut değilse gerekli geliştirme bağımlılığı. |
| `.github/workflows/tests.yml` | Birim/entegrasyon, uygulama açılış kontrolü ve arayüz kanıtlarını ekleme; ağlı performans bütçesini standart birim testlerine karıştırmama. |

### Eklenecek test ve kanıt dosyaları

- `tests/test_trade_decisions.py` — D grubundaki eylem ve koruma testleri.
- `tests/test_trade_execution.py` — E grubundaki fiyat, zaman, maliyet testleri.
- `tests/test_market_validation.py` — V grubundaki kapanış/veri testleri.
- `tests/test_position_journal.py` — P grubundaki gerçek teyit/düzeltme testleri.
- `tests/test_trading_parity.py` — I grubundaki bağımsız beklenen sonuç ve modlar arası eşitlik testleri.
- `tests/test_trading_ui.py` — U grubundaki gerçek karar görünümü testleri.
- `tests/test_trading_performance.py` — B grubundaki onaylı referans ölçümler; normal testten ayrı işaretli çalıştırma.
- `tests/test_trading_errors.py` — X grubundaki hata ve sahte başarı engelleme testleri.
- `tests/test_trading_smoke.py` — kullanıcı teyidiyle uçtan uca kritik akış ve yeniden açılış.
- `tests/fixtures/trading_v1/` — onaylı kuralların somut sınır örnekleri; AK01–AK04 sonrasında oluşturulacak veri dosyaları.
- `docs/evidence/0003/` — kriter numarasıyla ekran görüntüleri, ölçüm koşulları ve test raporları; uygulama doğrulaması sırasında oluşturulur.

`advanced_analysis.py` ve `optimize_weights.py` için strateji değişikliği planlanmıyor. Uygulama sırasında `ml_models.py` içindeki zorunlu göstergelerin `pandas_ta` yüklemesine bağlı olduğu ve ilk çalışmayı 120 saniyenin üstüne taşıdığı doğrulandı; ML kapatılmadan aynı standart gösterge ara hesaplarına bağlandı.

## Adım sırası ve katmanlar

### 0. Kaynak ve kararların kesinleştirilmesi

- [ ] Okunmuş kural belgelerinin AK06 istisnasıyla uygulandığını doğrula; iki spec'in 0002 numara çakışmasını yönetici kararıyla çöz.
- [ ] Spec'in güncel açık karar envanterindeki AK01–AK03 ve AK05 ayrıntılarını teknik inceleme/ölçümle paketle ve gerekli seçimleri kesinleştir. AK04'ün kapanmış kurallarını yeniden açma.
- [ ] Decimal kapsamını, eski portföy kayıtlarının gerçek/sanal niteliğini ve zorunlu ML/bileşen listesini netleştir.
- [ ] Mevcut çalışma kopyasını koruyarak `feature/0003-islem-tutarliligi` için izolasyon hazırla; ancak uygulama onayından sonra.
- [ ] Mevcut ilgili testlerin ve uygulama açılışının başlangıç durumunu kaydet; önceden var olan arızayı yeni değişiklikten ayır.

Çıkış: uygulanabilir kaynak spec ve ölçülebilir AK değerleri; bu adım tamamlanmadan bağımlı üretim koduna geçilmez.

### 1. İş kuralları ve ortak finansal sözleşme

Dosyalar: `trading_contracts.py`, `trade_decisions.py`, `config.py`; test: D/E grupları.

- [ ] Matristen pozisyon durumları, AL/BEKLE/SAT, stop/iz süren koruma, bekleme ve geçersiz değer testlerini yaz.
- [ ] Testleri çalıştırıp beklenen davranış eksikliği nedeniyle başarısız olduklarını doğrula.
- [ ] Onaylı durumları ve saf eylem kurallarını oluştur; fiyatlama veya kullanıcı kaydı bu aşamada yan etki yaratmasın.
- [ ] İlgili testleri geçir; bağımsız QA incelemesi için davranış ve kanıtı sun.

Girdi: onaylı pozisyon/koruma/finansal hassasiyet kuralları. Çıktı: eylem, neden, koruma ve bekleme durumunu taşıyan tek karar sözleşmesi.

### 2. Veri alımı sınırı ve piyasa doğrulaması

Dosyalar: `market_validation.py`, `data_fetchers.py`, `tests/conftest.py`; test: V/X grupları.

- [ ] Kapanıştan hemen önce/tam sınırda/sonra, kapalı seans, eksik mum, çelişkili OHLC, sağlayıcı hatası testlerini yazıp başarısızlığı doğrula.
- [ ] Sağlayıcının kaynak ve zaman bilgilerini koru; takvim ve veri niteliğini doğrula. Salt açık son mum yüzünden geçerli geçmişi engelleme.
- [ ] Sırasız/yinelenmiş veri için AK03'te onaylı davranışı uygula; çelişkili veriyi sessizce doğru sayma.
- [ ] İlgili testleri geçir ve kaynak sınırlarının kanıtını incelemeye sun.

Girdi: sağlayıcı verisi, değerlendirme anı, AK01/03. Çıktı: kapanmış kullanılabilir geçmiş veya nedenli uygunluk/hata sonucu.

### 3. Ortak sinyal ve değerlendirme akışı

Dosyalar: `trading_service.py`, `signal_engine.py`; test: I grubu, mevcut sinyal/cache testleri.

- [ ] Aynı tarih kesitinin canlı, geçmiş ve yeniden değerlendirmede farklılaşmasını yakalayan testler yazıp başarısızlığı doğrula.
- [ ] Son satırı koşulsuz silme yerine doğrulanmış geçmişi geçir. Durum makinesini aynı tarihsel girdiler ve zorunlu bileşenlerle ilerlet.
- [ ] Her karar için veri kesiti, ayar/kural sürümü ve bileşen uygunluğunu taşı; eksik bileşeni nötr başarılı sonuç gibi sunma.
- [ ] Ortak akışı ve veri düzeltmesi sonrası yeniden değerlendirmeyi test et; ML nedeniyle doğan maliyeti kaydet.

Girdi: 1 ve 2'nin sonuçları. Çıktı: aynı geçmiş ve pozisyonla tekrarlanabilir karar; eksik bileşen durumunda açık karşılaştırılamaz sonucu.

### 4. Gerçekleşme, geçmiş test ve sanal takip

Dosyalar: `trade_execution.py`, `technical_analysis.py`, `paper_trading.py`; test: E/I/X grupları.

- [ ] Manuel gecikme, stop/hedef eşit temas, boşluk, giriş öncesi temas, aynı mumda iki temas, maliyet/nakit sınırı ve tekrar işleme testlerini yazıp başarısızlığı doğrula.
- [ ] Tetikten ayrı gerçekleşme modeli oluştur; AK02 seçimine göre yalnız bilinebilen fiyatları veya açıkça etiketlenen varsayımları kullan.
- [ ] Backtest ve sanal takibi aynı karar/gerçekleşme geçişine bağla; BEKLE'yi çıkış sayma. Sanal kayıt hatasını başarılı kayıttan ayır.
- [ ] Küçük, elle hesaplanmış işlem dizisinin beklenen fiyat/zaman/net sonucuyla iki modu da ayrı ayrı doğrula; ardından birbirleriyle karşılaştır.
- [ ] Aynı mumun tekrarı ve uygulama yeniden açılışında ikinci işlem oluşmadığını doğrula.

Girdi: 3'ün kararı, OHLC/zaman, başlangıç simülasyon durumu, AK02/04. Çıktı: bir adet simüle geçiş veya gerçekleşmemiş/belirsiz sonuç.

### 5. Gerçek pozisyon teyidi ve kayıt geçişi

Dosyalar: `position_journal.py`, `portfolio.py`; test: P grubu ve mevcut portföy testleri.

- [ ] Teyitsiz açma/kapama, tekrar teyidi, fiyat düzeltmesi, sanal/gerçek ayrımı ve kayıt bozulması testlerini yazıp başarısızlığı doğrula.
- [ ] Yalnız açık kullanıcı teyidinin gerçek pozisyonu değiştirmesini sağla; stopa temas gerçek satış olmuş sayılmasın.
- [ ] Eski kayıtları kopyada incele; gerçek/sanal niteliği doğrulanamayanları gerçek teyitli pozisyon olarak otomatik aktarma. Kaynak kayıtları koru ve belirsizliği göster.
- [ ] Aynı teyidin tekrarında tek etkiyi, kaydetme başarısızlığında sahte başarı verilmediğini ve yeniden açılış tutarlılığını doğrula.

Girdi: kullanıcı teyidi veya düzeltmesi. Çıktı: doğrulanmış gerçek pozisyon; simülasyon durumu bu kaydı değiştiremez.

### 6. Arayüz ve tüm tüketicilerin bağlanması

Dosyalar: `trading_ui.py`, `app.py`, `theme.py`, `scanner.py`, `portfolio.py`; test: U grubu ve smoke.

- [ ] Bilinmeyen/geçersiz/açık/boş pozisyon, veri hataları ve kapsam etiketlerinin gerçek görünüm testlerini yazıp başarısızlığı doğrula.
- [ ] Karar panelini eylem/gerekçe/koruma ayrımıyla bağla; açık kullanıcı teyidi ve düzeltme akışını sun.
- [ ] Ana panel, kenar sinyalleri, tarayıcı ve bildirim metni üretiminde koşulsuz kişisel al/sat talimatını engelle; testte dışarı mesaj gönderme.
- [ ] Ekran görüntülerini görünür AC numaralarıyla kaydet; mevcut masaüstü/mobil grafik davranışı regresyonlarını çalıştır.
- [ ] Pozisyon yok → AL → alım teyidi → BEKLE ile tutma → koruyucu uyarı/SAT → satış teyidi → yeniden açılış akışını kanıtla.

Girdi: 3–5'in sonuçları. Çıktı: teknik hatayı sızdırmayan, gerçek işlemle uyarıyı ayıran görünüm.

### 7. Kabul, performans ve bağımsız QA

Dosyalar: `tests/test_trading_performance.py`, `.github/workflows/tests.yml`, koşullu `requirements-dev.txt`, `docs/evidence/0003/`.

- [ ] Kaynak spec'teki 104 AC'nin eşlemesini tekrar karşılaştır; R01–R30 için AC'lerde olmayan ek testleri de kontrol et.
- [ ] Hedefli testler ardından mevcut tam pytest grubunu bir kez çalıştır; hata varsa nedenini çözmeden başarı ilan etme.
- [ ] AK05'e göre ekran ve geçmiş değerlendirmeyi, ilk/sonraki çalışma ve toplam/hesaplama ayrımıyla ölç; donanım/veri/tekrar protokolünü raporla.
- [ ] Onaylı ortamda arayüz smoke ve görsel kanıtı üret; ağ hatası senaryolarını kontrollü gecikmelerle ayrı doğrula.
- [ ] Bağımsız QA'ya yalnız spec, diff ve test/ölçüm kanıtını sun. Developer'ın test çalıştırması bağımsız QA onayı değildir.
- [ ] Takım Yöneticisi kararı sonrası proje commit/PR/kapanış disiplinini uygula; SCORECARD'ı gerçek sonuçlarla güncelle.

## Riskler ve emin olunmayan noktalar

| Kimlik | Risk / belirsizlik | Öneri ve gerekçe |
|---|---|---|
| RİSK01 | Belgeler mevcut ve okundu; genel yığın tanımı çalışan uygulamadan farklı. | Feature için onaylı AK06 istisnasını esas al; genel belge düzeltmesini ayrı tut. Teknoloji dönüşümü açma. |
| RİSK02 | Dört ana başlığın alt ayrıntıları açık: AK01, AK02, AK03, AK05. | Teknik olarak bulunabilecek kaynak/sınırları önce incele; kalan kullanıcı seçimlerini toplu ve gerekçeli sun. Onaylı stop/bekleme kararlarını yeniden sorma. |
| RİSK03 | Para/yüzde decimal kuralı ile mevcut pandas/ML/float analizi çelişiyor. | Finansal kayıt, gerçekleşme ve yüzde sonuçlarında Decimal zorunlu olsun; analitik girdilerde konvansiyonlardaki onaylı dönüşüm sınırı uygulansın. Tüm analiz hattını sessizce dönüştürmek model davranışını ve maliyeti değiştirebilir; onaysız float istisnası da kurala aykırıdır. |
| RİSK04 | Canlı son 15 mumu oynatıyor, ML yalnız son mumda; backtest daha uzun geçmişi farklı bileşenlerle ilerletiyor. | Aynı tarihsel karar girdisini ortak akışta üret; dar pencereyi doğruluk kanıtı olmadan koruma. ML'yi kapatıp eşitlik elde etmek spec'i karşılamaz. |
| RİSK05 | Günlük OHLC stop emrinin mum içi kesin gerçekleşme sırasını vermiyor. | AK02 onaylı stop gerçekleşme modelini uygula; gerçek kullanıcı kayıtlarını teyitli fiyattan izle. Onaylı sonraki açılış kuralı yalnız sanal AL/SAT için geçerlidir. |
| RİSK06 | Eski portföy kaydı otomatik kapanıyor; kayıtların gerçek mi sanal mı olduğu belirsiz. | Kopya üzerinde geçiş ve kullanıcı teyidiyle sınıflama. Otomatik kayıtları gerçek satış kanıtı sayma; kaynak veriyi silme. |
| RİSK07 | Tekrar/eşzamanlı kayıt, yalnız atomik dosya değiştirmeyle çözülmüyor. | Aynı işlemi yeniden uygulamayı engelleyen kayıt kimliği ve yazım sıralaması planla; iki yazar/çökme senaryosunu sınayarak kayıp güncellemeyi önle. |
| RİSK08 | Eşitlik testinde iki hatalı yol aynı sonucu verebilir. | Önce bağımsız elle hesaplanmış küçük örneğe karşı doğrula, sonra modları karşılaştır. Sadece aynı yardımcı fonksiyonu çağırmaları kanıt değildir. |
| RİSK09 | Sağlayıcılar kapanış işareti/saat dilimini kaybediyor; seans bilgisi piyasaya bağlı. | AK01 seçimiyle kaynak metadata'sını koru, saat/takvim sınırlarını testte sabitle. Evrensel son satır çıkarma kuralı kullanma. |
| RİSK10 | Performans bütçesi bilinmiyor; ML tarihsel eşitliği pahalı olabilir. | Doğruluğu önce kur, aynı kararın tekrar kullanımını kanıtla; AK05 hedefini ölçümle belirle. Yeni model veya optimizasyon araştırması açma. |
| RİSK11 | Güncel 104 AC, R01–R30'in tüm ayrıntılarını tek başına kapsamıyor. | Resmi AC eşlemesine kayıp numaraları geri ekleme; devam eden gereksinimleri ayrı test et. Böylece kullanıcının dosyası değişmeden iş davranışı korunur. |
| RİSK12 | Mevcut pytest başlangıç durumu bu turda ölçülmedi; geçmiş turdaki çalışma sonuçsuz durdurulmuştu. | Uygulama başında hedefli başlangıç çalışmasını raporla. Eski kanıtı yeni ortamda geçmiş test sonucu sayma. |
| RİSK13 | İki farklı feature 0002 numarası taşıyor. | Onaylı 0003 numarasını uygulama öncesi branch/PR/kanıt atıflarında tekilleştir; hangi işin denetlendiği belirsiz kalmasın. Mevcut dosyaları onaysız yeniden adlandırma. |

## Kriter ↔ test eşlemesi

Dosya kısaltmaları: D=`tests/test_trade_decisions.py`, E=`tests/test_trade_execution.py`, V=`tests/test_market_validation.py`, P=`tests/test_position_journal.py`, I=`tests/test_trading_parity.py`, U=`tests/test_trading_ui.py`, B=`tests/test_trading_performance.py`, X=`tests/test_trading_errors.py`.

Her satırda test adı ve bağımsız beklenen sonuç vardır. U testleri gerçek görünüm akışını çalıştırır; yalnız HTML metnini aramakla yetinilmez. Kullanıcıya görünür sonuçlarda test raporuna `docs/evidence/0003/ACnn` önekli görüntü eklenir. AK'ye bağlı sınırlar test geliştirilmeden önce onaylı somut değerlere çevrilir.

| AC | Dosya :: test adı | Bağımsız kanıt |
|---|---|---|
| 14 | I :: test_ac14_net_parity | Aynı işlem/nakit/miktar/maliyetle iki net sonuç beklenen tutardır. |
| 15 | U :: test_ac15_missing_component_warning | Eksik bileşenle karşılaştırılamaz görünür. |
| 16 | V :: test_ac16_open_candle_mutation_immunity | Yalnız açık mumu değiştirince yayımlanan karar değişmez. |
| 17 | V :: test_ac17_exact_close_validation_boundary | AK03 kapanış koşulu tam sağlanınca mum dahil edilir. |
| 18 | V :: test_ac18_keep_last_closed_candle | Tamamlanmış son mum sırf son diye atılmaz. |
| 19 | U :: test_ac19_empty_data_state | Veri yok görünür; eylem görünmez. |
| 20 | V :: test_ac20_history_n_minus_one | Onaylı N−1 geçerli mumda alım engellenir. |
| 26 | U :: test_ac26_negative_fee_message | Negatif komisyon değerlendirme öncesi doğrulama mesajı verir. |
| 27 | U :: test_ac27_unknown_fee_no_verified_net | Bilinmeyen komisyonla doğrulanmış net sonuç gösterilmez. |
| 28 | E :: test_ac28_no_fill_before_signal_known | İşlem sinyalin bilinebildiği andan önce oluşmaz. |
| 29 | E :: test_ac29_next_daily_open | Uygun günlük AL/SAT sonraki işlem gününün açılış zamanı/fiyatını kullanır. |
| 30 | E :: test_ac30_intrabar_stop_trigger | Kapanış toparlansa da önceden etkin stop teması değerlendirilir. |
| 31 | E :: test_ac31_stop_target_collision | Bilgi hedefi satış yaratmaz; stop onaylı emir modeline göre değerlendirilir. |
| 32 | E :: test_ac32_stop_gap_fill | Stop ötesi açılış fiyatı onaylı modelle hesaplanır. |
| 33 | U :: test_ac33_exit_reason_visible | Koruyucu çıkış gerekçesi aynı paneldedir. |
| 34 | U :: test_ac34_confidence_not_probability | Puanla birlikte olasılık olmadığı görünür. |
| 35 | U :: test_ac35_assumptions_visible | Maliyet ve gerçekleşme varsayımları görülebilir. |
| 36 | B :: test_ac36_compute_budget | Ekran/geçmiş hesaplaması ayrı parametrelerle AK05 bütçesini aşmaz. |
| 37 | U :: test_ac37_unknown_no_unconditional_buy | Bilinmeyen pozisyonla AL bilgi olarak kalır. |
| 38 | U :: test_ac38_zero_open_quantity_invalid | Açık sıfır miktar geçersiz görünür, yok sayılmaz. |
| 39 | D :: test_ac39_missing_entry_no_stop | Eksik girişle kişisel stop üretilmez. |
| 40 | P :: test_ac40_signal_does_not_close_real_position | Teyitsiz SAT gerçek pozisyonu kapatmaz. |
| 41 | P :: test_ac41_duplicate_confirmation | Aynı teyit tekrarında tek işlem vardır. |
| 42 | P :: test_ac42_correction_keeps_execution | Fiyat düzeltmesi gerçekleşmiş işlemi silmez. |
| 43 | P :: test_ac43_paper_real_isolation | Sanal alış gerçek pozisyonu değiştirmez. |
| 44 | D :: test_ac44_single_exit_for_collision | Koruma+SAT aynı anda tek tam çıkış üretir. |
| 45 | D :: test_ac45_confirmed_stop_above_entry_valid | Kullanıcının teyitli stop yükseltmesi giriş üstünde geçerli kalır. |
| 46 | D :: test_ac46_disabled_target_keeps_stop | Hedef kapalıyken geçerli stop etkin kalır. |
| 47 | D :: test_ac47_invalid_initial_stop | Girişe eşit/üst ilk stop ayrı örneklerle reddedilir. |
| 48 | D :: test_ac48_loss_cooldown_before_two | Çıkış mumu hariç sıfır/bir tamamlanmış mumda yeni alım önerisi ve sanal giriş yoktur. |
| 49 | D :: test_ac49_loss_cooldown_exact_two | İkinci kapanışta bekleme engeli kalkar; uygun sinyal sonraki açılışta uygulanır. |
| 50 | D :: test_ac50_profit_no_loss_cooldown | Net kârlı çıkış zarar beklemesi başlatmaz. |
| 51 | E :: test_ac51_exact_stop_touch | Stopta eşitlik tetik koşulunu sağlar. |
| 52 | E :: test_ac52_target_touch_no_sale | Bilgi hedefinde eşitlik satış üretmez. |
| 53 | E :: test_ac53_target_gap_no_sale | Bilgi hedefinin üstünde açılış tek başına satış üretmez. |
| 54 | U :: test_ac54_ambiguous_sequence_label | Bilinmeyen sıra kesin gözlenmiş işlem olarak sunulmaz. |
| 55 | E :: test_ac55_preentry_touch_no_exit | Girişten önceki temas pozisyonu kapatmaz. |
| 56 | E :: test_ac56_explicit_zero_fee | Açık sıfır maliyet bilinmeyen sayılmaz. |
| 57 | E :: test_ac57_no_double_spread | Fiyata dahil makas tekrar eklenmez. |
| 58 | U :: test_ac58_negative_delay_rejected | Negatif tepki süresi mesajla engellenir. |
| 59 | E :: test_ac59_cash_including_fees | Ücret dahil nakit aşılırsa işlem oluşmaz. |
| 60 | V :: test_ac60_closed_session_not_stale | Son beklenen mum mevcutken kapalı seans eskilik değildir. |
| 61 | V :: test_ac61_open_tail_valid_history | Açık son mumla yeterli tamamlanmış geçmiş değerlendirilir. |
| 62 | V :: test_ac62_unready_required_fields | Satır sayısı yetse bile zorunlu alan hazır değilse alım yoktur. |
| 63 | V :: test_ac63_high_below_low | Çelişkili OHLC'den yeni karar üretilmez. |
| 64 | U :: test_ac64_open_position_data_risk | Açık pozisyonda güncel risk doğrulanamıyor görünür. |
| 65 | U :: test_ac65_old_decision_timestamp | Eski kararın zamanı ve eskiliği görünür. |
| 66 | I :: test_ac66_no_parity_success_missing_component | Zorunlu bileşen eksikken eşitlik başarısı yoktur. |
| 67 | I :: test_ac67_internal_net_precision | Ekranda aynı görünen farklı hesap sonucu eşit sayılmaz. |
| 68 | I :: test_ac68_timezone_same_instant | Aynı anın farklı saat dilimi gösterimi eşit kalır. |
| 69 | I :: test_ac69_repeat_paper_decision_once | Aynı kesit ve başlangıç tekrarında ikinci sanal işlem oluşmaz. |
| 70 | X :: test_ac70_provider_timeout_not_success | Sağlayıcı zaman aşımı boş başarıya dönüşmez. |
| 71 | U :: test_ac71_out_of_scope_label | Kapsam dışı mevcut seçenek doğrulanmadı etiketiyle kalır. |
| 72 | U :: test_ac72_unknown_without_interaction | Bilinmiyor mesajı tıklama/hover olmadan görünür. |
| 73 | B :: test_ac73_total_wait_budget | Ağ/ilk çalışma dahil toplam bekleme AK05 bütçesini aşmaz. |
| 74 | D :: test_ac74_stop_fixed_until_confirmation | Fiyat artışı ve tekrar değerlendirme teyitsiz stop değişikliği yaratmaz. |
| 75 | E :: test_ac75_stop_raise_effective_time | Stop yükseltmesi önceki temaslara geriye dönük uygulanmaz. |
| 76 | U :: test_ac76_no_invented_stop_raise | Kayıtsız yükseltme oluşturulmaz; sabit stop + SAT varsayımı görünür. |
| 77 | P :: test_ac77_ignored_buy_no_real_trade | Teyit edilmeyen AL gerçek işlem oluşturmaz. |
| 78 | P :: test_ac78_late_entry_preserves_trade_time | Geç kayıt gerçek işlem zamanını değiştirmez. |
| 79 | U :: test_ac79_real_and_simulated_labels | Gerçek ve simüle sonuçların dayanağı ayrı görünür. |
| 80 | U :: test_ac80_record_trade_unknown_fee | Bilinmeyen komisyonla gerçek teyit kabul edilir; doğrulanmış net sonuç gösterilmez. |
| 81 | P :: test_ac81_late_fee_preserves_trade | Sonradan komisyon maliyeti günceller; teyitli fiyat/miktar/zaman değişmez. |
| 82 | U :: test_ac82_no_next_bar_pending | Sonraki günlük mum yoksa bekleyen sinyal görünür; sanal gerçekleşme oluşturulmaz. |
| 83 | I :: test_ac83_initial_stop_atr_parity | Giriş 100/ATR 4 için üç yol da yuvarlama öncesi 90 stop üretir; ayrıca açılış boşluğu olan giriş örneği sınanır. |
| 84 | I :: test_ac84_decision_candle_atr_frozen | Karar ATR'si 4/giriş 100 iken sonraki ATR 6 olsa da aynı girişin stopu 90 kalır. |
| 85 | D :: test_ac85_net_loss_after_known_fee | Pozitif fiyat farkı, bilinen komisyonla negatife dönerse bekleme başlar. |
| 86 | U :: test_ac86_provisional_loss_classification | Bilinmeyen komisyonla negatif/sıfır/pozitif fark senaryolarında bekleme ve geçici etiket doğrulanır; net başarı iddiası yoktur. |
| 87 | D :: test_ac87_breakeven_no_cooldown | Net veya geçici sıfır sonuç bekleme başlatmaz. |
| 88 | P :: test_ac88_late_fee_no_new_cooldown | Geç komisyonla rapor zarara dönse de yeni bekleme/sinyal yoktur. |
| 89 | P :: test_ac89_late_fee_preserves_cooldown | Devam eden/bitmiş bekleme sınırları maliyet revizyonuyla değişmez. |
| 90 | U :: test_ac90_paper_defaults | İlk ayarlarda varlık başına 10.000 sermaye/1.000 işlem tutarı para birimiyle görünür. |
| 91 | I :: test_ac91_custom_paper_amounts | Seçilen geçerli farklı sermaye/tutar yeni değerlendirmede kullanılır. |
| 92 | I :: test_ac92_asset_balances_independent | Bir varlıkta işlem diğerinin nakdini değiştirmez. |
| 93 | U :: test_ac93_no_mixed_currency_total | TL/USD/USDT parasal sonuçları tek toplama dönüşmez. |
| 94 | E :: test_ac94_fee_added_to_notional | 10.000 nakit, 1.000 tutar ve 2 ücret sonrası 8.998 nakit kalır; alım 1.000 kalır. |
| 95 | E :: test_ac95_exact_cash_with_fee | Toplam bedelde tam eşitlik kabul edilir; bir hassasiyet birimi altında giriş reddedilir. |
| 96 | U :: test_ac96_unknown_fee_cash_unverified | Bilinmeyen komisyonda ücret dahil nakit yeterliliği doğrulanmış gösterilmez. |
| 97 | U :: test_ac97_unknown_spread_slippage_labels | Eksik makas/kayma ayrı etiketlenir; sıfır veya tam maliyet sonrası doğrulanmış sonuç sayılmaz. |
| 98 | I :: test_ac98_market_cost_assumptions | Yeni sanal değerlendirme seçilen piyasa varsayımını kullanır; diğer piyasanın ayarı korunur. |
| 99 | E :: test_ac99_real_fill_no_double_slippage | Sanal makas/kayma değişimi teyitli gerçek fiyata ve o fiyat farkının sonucuna uygulanmaz. |
| 100 | E :: test_ac100_quantity_round_down | 1.000 bütçe/300 fiyat/1 adım için miktar 3, bedel 900, kullanılmayan nakit 100 olur. |
| 101 | U :: test_ac101_below_minimum_quantity | Bütçe en küçük miktara yetmiyorsa sanal giriş yoktur ve neden görünür. |
| 102 | U :: test_ac102_unknown_quantity_step | Miktar adımı bilinmiyorsa uydurma kesirli işlem oluşturulmaz; engel görünür. |
| 103 | P :: test_ac103_real_quantity_unchanged | Sanal yuvarlama ayarı gerçek teyitli miktarı değiştirmez. |
| 104 | V :: test_ac104_usd_usdt_equivalence | Geçerli ETH/USD verisiyle ETH/USDT pozisyonu değerlendirilir; para birimi farkı engel veya ek teyit yaratmaz. |
| 105 | U :: test_ac105_gold_reference | XAU görünümünde GC=F vadeli altın referansı olduğu görünür; farklı gerçek altın ürününde gerçekleşmiş stop teyidi sayılmaz. |
| 106 | U :: test_ac106_derived_gold | Türetilmiş GRAM_TRY verisinde stop simülasyonu ve ATR işlem uygunluğu doğrulanmış olarak sunulmaz. |
| 107 | V :: test_ac107_source_fallback | Yedek kaynak kullanıldığında değişim görünür; tek karar geçmişinde iki kaynağın mumları karıştırılmaz. |
| 108 | E :: test_ac108_open_stop_priority | Bekleyen SAT varken açılış etkin stopa eşit veya altındaysa tek sanal satış oluşur; gerekçe stop, maliyet öncesi fiyat açılıştır. |
| 109 | E :: test_ac109_entry_bar_stop | Açılışta gerçekleşen sanal alışın pozitif ilk stopuna aynı mumda aşağı temas olduğunda stop seviyesinden maliyet öncesi tek çıkış oluşur. |
| 110 | E :: test_ac110_invalid_stop | Giriş eksi 2,5 karar ATR sonucu sıfır veya negatifse sanal alış oluşmaz. |
| 111 | E :: test_ac111_cost_formula | Temel fiyat 100, tam makas 100 baz puan ve kayma 25 baz puanken komisyon öncesi model alış fiyatı 100,75 ve satış fiyatı 99,25 olur. |
| 112 | E :: test_ac112_cost_boundary | Yarım makas ve kayma toplamı 10.000 baz puanken giriş reddedilir; 9.999 baz puan diğer koşullar geçerliyken bu üst sınırdan engellenmez. |
| 113 | U :: test_ac113_fee_boundary | Yüzdesel komisyon için sıfır kabul edilir; negatif, sonlu olmayan, yüzde 100 veya üstü değer mesajla reddedilir. |
| 114 | P :: test_ac114_prospective_cost | Sanal makas/kayma değiştiğinde önceki sanal kayıt korunur; sonraki değerlendirme yeni varsayımı kullanır. |
| 115 | E :: test_ac115_stop_rounding | Ham ilk stop 90,03 ve fiyat adımı 0,05 iken öneri 90,05 olur; öneri girişe eşit veya üstündeyse ilk stop geçersizdir. |
| 116 | V :: test_ac116_crypto_delay | Beklenen UTC günlük mum eksikken kapanıştan 4 dakika 59 saniye sonra yeni veri bekleniyor, tam 5 dakika sonra eski veri görünür; iki durumda da yeni AL oluşmaz. |
| 117 | V :: test_ac117_yahoo_delay | Beklenen Yahoo günlük mum eksikken seans kapanışından 29 dakika 59 saniye sonra yeni veri bekleniyor, tam 30 dakika sonra eski veri görünür; iki durumda da yeni AL oluşmaz. |
| 118 | V :: test_ac118_history_boundary | Diğer uygunluk koşulları sağlandığında 199 geçerli kapanmış günlük mum alımı engeller; 200 mum asgari geçmiş filtresini geçer. |
| 119 | V :: test_ac119_open_flag | Sağlayıcının açık işaretli mumu, seans sonu ve tolerans geçse dahi sinyal geçmişine alınmaz. |
| 120 | U :: test_ac120_component_failure | Zorunlu bileşen hesaplanamadığında yeni AL/SAT oluşmaz ve bileşen adı görünür; geçerli nötr sonuç hata sayılmaz. |
| 121 | I :: test_ac121_ml_time_boundary | Yalnız değerlendirme anından sonraki veri değiştirildiğinde geçmiş kesite ait ML kararı değişmez. |
| 122 | P :: test_ac122_legacy_record | Eski otomatik kapanış kullanıcı satış teyidi olmadan doğrulanmış gerçek performansa katılmaz; eski/teyitsiz olarak görünür. |

## AC dışında gereksinim tamamlayıcı testler

| Requirements | Test | Kanit |
|---|---|---|
| R05/R15 | U :: test_requirement_unknown_position_default | Bilgi yokken bilinmiyor görünür. |
| R01 | U :: test_requirement_flat_buy_action | Pozisyonsuz AL yeni alım gösterir. |
| R01 | U :: test_requirement_open_buy_means_hold | Açık pozisyonda AL tut gösterir. |
| R02 | D :: test_requirement_wait_preserves_position | Koruma yokken BEKLE çıkış yaratmaz. |
| R02 | D :: test_requirement_flat_wait_no_entry | Pozisyonsuz BEKLE giriş yaratmaz. |
| R03 | U :: test_requirement_sell_full_exit | SAT tam çıkış eylemini gösterir. |
| R03 | D :: test_requirement_flat_sell_no_sale | Pozisyonsuz SAT satış yaratmaz. |
| R04 | D :: test_requirement_stop_overrides_buy | Stop, AL'a rağmen tam çıkış üretir. |
| R05 | P :: test_requirement_signal_does_not_open_real_position | Teyitsiz AL gerçek pozisyon açmaz. |
| R06/R26 | I :: test_requirement_decision_parity | Aynı bağlamda üç tüketici, bağımsız beklenen eylemle eşleşir. |
| R07 | I :: test_requirement_stop_parity | Ekran/test/sanal stopu aynı onaylı seviyedir. |
| R07 | I :: test_requirement_target_parity | Ekran/test/sanal etkin hedefi aynıdır. |
| R10/R26 | I :: test_requirement_fill_parity | Simüle zaman ve fiyat, elle hazırlanmış beklenen işlemle eşleşir. |
| R24 | V :: test_requirement_history_exact_n | N geçerli mumda geçmiş filtresi engellemez. |
| R09/R24 | V :: test_requirement_freshness_exact_limit | Beklenen mum eksikken toleransın tam sınırında eski veri engeli vardır. |
| R09 | V :: test_requirement_freshness_over_limit | Sınırın bir ölçüm birimi üstünde yeni alım yoktur. |
| R09/R28 | U :: test_requirement_filter_reason_visible | Veri filtresi nedeni karar panelinde görünür. |
| R09 | V :: test_requirement_nonpositive_price | Sıfır ve negatif fiyat ayrı parametrelerle alımı engeller. |

- R15–R17: P grubunda negatif miktar, eksik teyit zamanı, açık satış teyidi, düzeltmenin yeniden açılışta korunması, gerçekleşmiş işlemi iptalle silmeme; U grubunda bilinmeyen pozisyonla SAT/TUT talimatı verilmemesi.
- R18–R20: D grubunda stop+BEKLE, bilgi hedefi+SAT, kullanıcı teyitli başabaş stop, sıfır/negatif etkin seviye, devre dışı hedefin satış yaratmaması ve çıkış mumu hariç iki mum sayımı.
- R23: E grubunda negatif makas/kayma, oran üst sınırı, giriş/çıkış ücretleri ayrı, ücret dahil nakde tam eşitlik, yuvarlama sırası.
- R24–R25: V/X grubunda eksik seans mumu, sırasız ama geçerli geçmiş, aynı/çelişkili yinelenen mum, hesaplama hatası, sanal kayıt hatası; son iki hata kullanıcıya sahte başarı vermemeli.
- R26–R28: I grubunda veri düzeltmesi ve ayar sürümü değişimi; U grubunda eksik bileşenin adı, geçersiz seçimde sessiz varsayılana dönmeme ve tüm kritik mesajların ek etkileşimsiz görünmesi.
- R17/R27: P/I gruplarında yeniden başlatma, eşzamanlı aynı teyit ve yarıda kalan yazım sonrası tekrar; gerçek kullanıcı dosyaları yerine geçici kopyalar kullanılır.

## Çalıştırma ve kabul yöntemi

Her dilimde ilgili matris grupları `python -m pytest` ile hedefli çalıştırılır; uygulama sırasında yazılan davranış testi önce başarısızlığı, sonra başarıyı göstermelidir. Birim testler sabit saat ve yerel veriyle ağdan bağımsızdır. Nihai regresyon mevcut `tests/` kümesini kapsar; performans testleri AK05 ortamında ayrı çalıştırılır. Görsel kanıt ve kullanıcı akışı birim testlerinin yerine geçmez, onları tamamlar.

Bu plan Takım Yöneticisi tarafından onaylandı ve feature/0003-islem-tutarliligi dalında uygulandı. Nihai test ve ölçüm kanıtları uygulama raporu ile `docs/evidence/0003/` altında tutulur; bağımsız QA kararı ayrıca alınır.
