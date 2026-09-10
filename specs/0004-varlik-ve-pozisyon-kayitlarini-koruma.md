# Spec: 0004 — Varlık ve Pozisyon Kayıtlarını Koruma

> Şablon: [TEMPLATE.md](TEMPLATE.md). Durum: Taslak; INTENT ve CLARIFY kararları onaylı.
> Rol: Analist — INTENT · CLARIFY · SPEC. Uygulama planı veya teknik çözüm içermez.
> **İşleme notu (rev. 1):** Bu revizyondaki G01–G17 maddeleri QA oturumunun kanıtlı bulgu
> listesinden gelir ve Takım Yöneticisi tarafından tümü onaylanmıştır. Metni, Takım
> Yöneticisinin açık talimatıyla **QA oturumu işlemiştir**; bu, `docs/roles/README.md`
> uyarınca SPEC rozetinin Analist'e ait olması kuralına bilinçli bir istisnadır.
> **rev. 2:** QA kanıtıyla açılan **P4** (kayıt yerinin başlatma klasörüne bağlı olması)
> Takım Yöneticisi tarafından onaylandı; kısıta ve AC18'e işlendi.
> **rev. 3:** Takım Yöneticisinin rev.1 metnine yönelik üç bulgusu işlendi — AC12c kapsamı
> aktif kayıtlarla sınırlandı, AC01 ikiye ayrıldı (AC01 + AC01b), yeniden yayınlama için
> ayrı kriter (AC19) eklendi.
> **rev. 4:** **P5** karara bağlandı — Streamlit yayını mevcut ve aktif kullanımda; AC05b ve
> AC19 için ölçüm ortamı vardır.
> **rev. 5:** **P6** karara bağlandı — program yalnız Streamlit yayınında kullanılıyor ve kayıp
> orada yaşanıyor. Referans ortam yayındır; AC01b'nin kaynağı yayın kayıtlarıdır. P4 bulgusu
> geçerli bir kusur olarak kalır ama bildirilen kaybın nedeni değildir.
> **rev. 6:** Kayıp mekanizması **P7**'de kanıt zinciriyle kayda geçti (karar gerektirmez).
> **rev. 7:** **P3** karara bağlandı — ilgisiz projeye ait belgeler bağlayıcı değil; bu iş için
> geçerli test düzeni pytest + `tests/` + mevcut CI olarak tespit edildi ve DoD'a işlendi.
> Açık kalanlar: P1, P2.
> **rev. 8:** Takım Yöneticisinin PLAN incelemesi kararları işlendi — **P1 onaylandı**
> (örnek "toplam 100 pozisyon", ölçüm yalnız kayıtların okunup gösterilmesini kapsar),
> **P2 daraltıldı** (yeni cihaz eşitleme özelliği yok; sunucu tarafı kalıcı saklama ihtiyacı
> bunun dışındadır), portföy sıfırlaması kapsama alındı (**AC11d**) ve belirsiz/okunamayan
> pozisyon kayıtlarının kapalı sayılamayacağı **R4.1** olarak yazıldı. AC05b ve AC19
> tamamlanma şartı olarak korunur. Açık konu kalmadı.
> **rev. 9:** Uygulama sonrası QA denetiminin bulguları işlendi. Kodda düzeltilenler
> spec metnini değiştirmez; metne giren iki madde şudur: **eşzamanlı yazma sınırı**
> (bilinen sınır olarak kısıtlara eklendi, ayrı spec konusu) ve **yayın yetkilendirmesi**
> (açık konu **P8**). AC05b ve AC19 hâlâ ölçülmedi.
> **rev. 10:** İkinci QA turu — F10 (otomatik kapanışın geri alınabilmesi) kodda
> düzeltildi; **F11** (otomatik kapatmanın yanlış anahtarları okuması) bu işin kapsamı
> dışında bir mevcut kusur olarak kayda geçti ve ayrı spec önerildi. AC05b, AC19 (F7) ve
> AC17'nin dar kapsamı (F9) açık kalmaya devam ediyor.
> **rev. 11:** Uygulama `main`'e alındı (PR #10, squash-merge). Birleştirmede spec 0003 ile
> iki uyum kararı verildi (aşağıda). SCORECARD dolduruldu. Kapanış için kalan tek şart
> yayın ölçümleridir: **AC05b ve AC19**.

## Intent
Varlık yönetimini kullanan kullanıcı, eklediği varlıkları ve aktif pozisyonlarını tekrar eklemek zorunda kalmadan güvenle takip etmek istiyor.
Mevcut ve yeni kayıtlar, pozisyon açılmamış varlıklar dahil, sayfa yenilendiğinde, uygulama kapatılıp açıldığında ve uzun süre kullanılmayıp Streamlit uyku durumundan döndüğünde bilgileriyle korunmalıdır.
Başarı, kullanıcının geri döndüğünde kayıtlarını miktar ve alış bilgileriyle bulması; kısmen kapatılmış pozisyonların aktif kalması ve tamamen kapanan pozisyonun varlığının listeden kendiliğinden kaldırılmamasıdır.
Varlık yalnızca kullanıcının açık silme işlemiyle kaldırılabilir ve aktif pozisyonu varken silinemez.
Geçmişte kaybolmuş kayıtları geri getirmek ve alım-satım ya da analiz kurallarını değiştirmek bu çalışmanın dışındadır.

## Requirements
- R1: Geliştirme öncesinde mevcut olan varlıklar ve aktif pozisyonlar bilgileriyle korunmalıdır.
- R2: Başarıyla eklenen her varlık, pozisyonu bulunmasa da kullanıcının sonraki ziyaretlerinde erişilebilir olmalıdır.
- R3: Sayfa yenileme, uygulamayı kapatıp açma, kullanım dışı geçen süre ve uykuya geçip yeniden açılma kayıt kaybına veya bilgilerin sıfırlanmasına yol açmamalıdır.
- R4: Aktif pozisyonun miktarı, alış bilgileri ve kayıtlı diğer işlem bilgileri, geçerli bir kullanıcı işlemi veya mevcut iş kuralı bunları değiştirmedikçe korunmalıdır.
  - **R4.1 (rev.8):** Miktarı okunamayan, eksik veya durumu tanınmayan bir pozisyon kaydı
    kapatılmış sayılamaz; sorunlu kayıt olarak korunur ve ilişkili varlığın silinmesini
    engeller. Aktiflik yorumu, görünürlük ile silme engellerinde aynı olmalıdır.
- R5: Kısmen kapatılmış pozisyon aktif kabul edilmeli; tamamen kapatıldığında ilgili varlık listede kalmalıdır. Yeni bir kısmi kapatma işlevi istenmemektedir.
- R6: Kullanıcının açık silme işlemi bulunmadan varlık kaldırılmamalı; aktif pozisyonla ilişkili varlığın silinmesi engellenmeli ve nedeni kullanıcıya açıklanmalıdır.
  - **R6.1 (G02):** Bekleyen emri bulunan varlık da silinemez; kullanıcıya nedeni bekleyen emir olarak açıklanır. Bekleyen emirdeki kilitli tutar sahipsiz bırakılamaz.
  - **R6.2 (G11):** Birden çok kaydı aynı anda kaldıran sıfırlama işlemleri de açık silme sayılır; ancak kullanıcıdan ayrıca onay istenmeden yürütülemez ve aktif pozisyonu ya da bekleyen emri olan varlıklar bu işlemle de kaldırılamaz.
- R7: Liste görünümünü daraltmak kayıtları değiştirmemeli; görünümden gizlenmek silinmek anlamına gelmemelidir.
- R8: Gerçekten kayıt bulunmaması ile mevcut kayıtlara erişilememesi kullanıcıya farklı durumlar olarak gösterilmelidir; erişim sorunu kayıtların boşaltılmasına yol açmamalıdır.
  - **R8.1 (G03):** Kayıtlara erişilemediği sürece uygulama kayıt yazmaz; kullanıcı kayıt değiştiren işlemleri erişim düzelene kadar yapamaz ve bunun nedeni kendisine açıklanır.
  - **R8.2 (G14):** Seçilen varlığın karşılığı bulunamadığında başka bir varlığın bilgileri gösterilmez; kullanıcıya varlığın bulunamadığı bildirilir.
  - **R8.3 (G15):** Bilgileri eksik veya okunamayan bir kayıt silinmez ve boş kayıtla değiştirilmez; kullanıcıya sorunlu kayıt olarak gösterilir.
- R9: Mevcut giriş kurallarına göre geçersiz bir ekleme girişimi, mevcut varlıkları ve pozisyonları değiştirmemeli; kullanıcıya anlaşılır bir açıklama verilmelidir.
  - **R9.1 (G12):** Var olan bir varlıkla aynı adı taşıyan ekleme girişimi, mevcut kaydı sessizce değiştiremez; kullanıcıya aynı adın kayıtlı olduğu bildirilir.

## Constraints
- Hedef kullanım ortamı, kullanıcının belirttiği Streamlit yayınıdır; uygulamanın uykuya geçmesi engellenmesi gereken bir davranış olarak tanımlanmamıştır.
- Kullanılmama süresi kayıt silme gerekçesi olamaz; kayıtların korunması düzenli ziyaret veya yeniden ekleme gerektiremez.
- **Saklama üst sınırı yoktur (G06):** Kayıtların saklanmasında zamana bağlı üst sınır yoktur; süre geçmesi tek başına silme gerekçesi olamaz.
- **Saklama yeri bağımsızlığı (G17 + P4, onaylı):** Kayıtlar, uygulamanın yeniden yayınlanmasından ve barındırma ortamının yeniden başlatılmasından etkilenmeyecek biçimde saklanmalıdır. **Kayıtlar ayrıca, uygulamanın hangi klasörden başlatıldığından bağımsız olarak aynı yerde bulunmalıdır.** Teknik çözüm bu spec'in konusu değildir, PLAN aşamasına aittir.
- **Mevcut kural — listede en az bir varlık (G10):** Listede en az bir varlık bulunması mevcut bir kuraldır ve korunur; son varlığın silinmesi engellenir ve nedeni açıklanır.
- Onaylı işlemler dışında varlık ve aktif pozisyon kaybı için tolerans sıfırdır.
- Koruma, başarıyla eklenmiş kayıtları kapsar; henüz tamamlanmamış form girişleri kayıt sayılmaz.
- Kayıtlı miktar ve alış bilgileri korunur; canlı piyasa fiyatının veya hesaplanan kâr/zararın aynı kalması beklenmez.
- **Yasaklar:** Otomatik varlık silme; aktif pozisyonlu veya bekleyen emirli varlığı silme; erişilemeyen kayıtları boş kayıtlarla değiştirme; bilinmeyen varlık yerine başka bir varlığın bilgilerini gösterme; mevcut alım-satım ve analiz kurallarını değiştirme.
- Geliştirme ve onay disiplini için [AGENTS.md](../AGENTS.md) geçerlidir. Para/yüzde kuralının bu iş için kapsamı: bkz. Definition of Done (G16).
- **Performans hedefi (P1, onaylandı — rev.8):** Uygulama kullanıma hazır olduktan sonra, 100 varlık ve toplam 100 aktif pozisyon içeren örnekte kayıtların görünmesi 5 saniyeyi aşmasın; aynı koşullardaki 10 ölçümün her biri bu sınırı sağlasın. Streamlit'in uyanma süresi bu ölçüme dahil edilmesin. Ölçüm sınırları için bkz. AC17 (G09). Bu hedef yalnız kayıtların okunup gösterilmesini ölçer; uygulamanın toplam açılış hızını garanti etmez, canlı fiyat bekleme süresi ayrı değerlendirilir.
- **KAPSAM DIŞI (V1):**
  - Geçmişte kaybolmuş kayıtların geri getirilmesi.
  - Alım-satım, otomatik pozisyon kapatma ve analiz kurallarının değiştirilmesi.
  - Yeni kısmi pozisyon kapatma işlevi.
  - Yeni filtre veya filtre tercihini sonraki ziyaret için hatırlama işlevi.
  - Uygulamanın sürekli uyanık tutulması.
  - Sahipsiz kalmış pozisyonları temizleyen yeni bir işlev (G13; koruma silme engeliyle sağlanır).
  - Para/yüzde sayı türü dönüşümü (G16; ayrı spec konusu).
  - Eşzamanlı yazma çakışmasının çözülmesi (bkz. bilinen sınır, rev.9).
- **Spec 0003 ile uyum kararı (rev.11):** 0003'ün satış akışı, **kısmi satışta da**
  pozisyona `CLOSED_CONFIRMED` durumunu yazıyor. Bu, AC09 ile çelişirdi (kalan miktarı olan
  pozisyon aktif kalmalı ve varlığı silinememeli). Karar: **kalan miktar durumun önündedir**
  (G01'in tanımı esas alınır) — kapanış durumu yazılmış olsa bile kalan miktarı sıfırdan
  büyük olan kayıt aktiftir ve silmeyi engeller. Alternatif (0003'ün satış akışını
  değiştirmek) birleştirme sırasında sessizce yapılamayacağı için seçilmedi; 0003 tarafında
  ele alınması Takım Yöneticisinin kararına bırakılmıştır.
- **Mevcut kusur — otomatik kapatma çalışmıyor (QA F11, rev.10; KAPSAM DIŞI):**
  Otomatik kapatma kayıttan `Giris` ve `Miktar` anahtarlarını okuyor, arayüz ise
  pozisyonu `Giriş` ve `Adet` anahtarlarıyla yazıyor; ayrıca arayüz pozisyona TP/SL hiç
  yazmıyor. Sonuç: **arayüzün ürettiği hiçbir pozisyonda otomatik kapatma çalışmıyor**;
  TP/SL taşıyan bir kayıt geldiğinde ise hesap hatalı sonuç veriyor. Bu, bu işten önce
  de var olan bir kusurdur ve "otomatik pozisyon kapatma kurallarının değiştirilmesi"
  KAPSAM DIŞI olduğu için burada düzeltilmemiştir. Bu spec'in kriterleri otomatik
  kapanmanın tetiklenmemesi üzerine kurulu olduğundan (AC önsözü, G08) bu iş
  etkilenmez. **Öneri:** ayrı bir spec açılsın; gerekçe, kullanıcı TP/SL koyduğunu
  sanıp korunmuyor olabilir. Karar Takım Yöneticisinindir.
  **Güncelleme (rev.11):** Spec 0003 otomatik kapatmayı tümüyle kaldırdı; fiyat teması artık
  yalnız uyarı üretiyor, pozisyonu ve nakdi değiştirmiyor. Bu bulgunun konusu böylece
  kendiliğinden ortadan kalkmıştır.
- **Bilinen sınır — eşzamanlı yazma (QA F6, rev.9):** Kayıtlar belge olarak, tek
  parça hâlinde yazılır ve sürüm denetimi yoktur. Aynı kayıtlar iki oturumda (iki sekme
  ya da iki cihaz) aynı anda değiştirilirse, sonra yazan öncekinin değişikliğini iz
  bırakmadan siler. Tek kullanıcılı kullanımda olasılığı düşüktür; V1'de kapsam dışıdır
  ve **ayrı bir spec konusudur**. Hiçbir kabul kriteri bu durumu kapsamaz.
- **V1 sınırı (P2, onaylandı — rev.8):** Yeni bir cihaz eşitleme özelliği eklenmez; mevcut erişim korunur. Sunucu tarafındaki kalıcı saklama ihtiyacı bu sınırın dışındadır ve geçerlidir. Mevcut cihazlar arası davranış kaldırılmaz.

## Context
- İlgili alan: Varlık Yönetimi ve Aktif Pozisyonlar.
- Kullanıcı bildirimi: Eklenen varlıklar bir süre sonra kayboluyor ve tekrar eklenmeleri gerekiyor.
- Uykuya geçme ile kayıp arasındaki nedensellik henüz doğrulanmadı; uyku sonrası koruma beklentisi açıkça onaylandı.
- **İş terimleri (G01):** *Varlık*, kullanıcının takip etmek üzere eklediği kayıttır. *Aktif pozisyon*, kalan miktarı sıfırdan büyük olan pozisyondur. Kalan miktarı sıfıra inen pozisyon *tamamen kapatılmış* sayılır; kapanış hangi yolla (kullanıcı satışı ya da mevcut otomatik kural) olursa olsun aynı kabul edilir. *Bekleyen emir*, henüz başlamamış ancak tutarı kilitlenmiş kayıttır ve aktif pozisyondan ayrı bir durumdur.
- Kullanıcı, alakasız olan `docs/domain.md` dosyasını kaldırdığını bildirdi; bu spec o dosyaya dayanmaz.
- İlgili mevcut spec'ler: 0001 ve 0002 mobil grafik konusundadır; bu iş için davranış bağımlılığı tanımlanmadı.
- **P3 — doküman uyumu (KARAR: ilgisiz belgeler bağlayıcı değil):** Takım Yöneticisi, söz konusu belgelerin alakasız bir projeye ait olarak oluşturulduğunu bildirdi. `docs/architecture.md` bu arada depodan kaldırılmıştır. [testing.md](../docs/testing.md)'nin teknolojiye özgü bölümleri (xUnit çatısı, `Metot_Durum_BeklenenSonuc` adlandırması, .NET test projesi ayrımı, sipariş/sepet akışı örnekleri) bu iş için **bağlayıcı değildir**; teknolojiden bağımsız "Genel" bölümü (yeşil pipeline olmadan merge yok, önce hatayı gösteren test, kaçan hatanın SCORECARD'a işlenmesi) geçerliliğini korur.
  **Bu iş için geçerli test düzeni (kod üzerinde doğrulandı):** çatı **pytest** (`requirements-dev.txt`: pytest 9.0.3); testler `tests/` altında (16 dosya); sürekli tümleştirme `.github/workflows/tests.yml` ile `python -m pytest tests/ -v` çalıştırır. Kural değişmez: **her kabul kriteri en az bir testle** karşılanır; arayüz kriterleri için ekran görüntüsü kanıtı verilir.
  **Kapsam notu:** `AGENTS.md`'nin "Neyi Nerede Bulursun" tablosu hâlâ kaldırılmış `docs/architecture.md`'yi ve testing.md'yi "backend xUnit" olarak işaret etmektedir. Bunun düzeltilmesi bu spec'in kapsamı dışındadır ve ayrıca ele alınmalıdır (QA bulgusu; karar Takım Yöneticisinindir).
- **P4 — kayıt yeri başlatma biçimine bağlı (QA kanıtı; KARAR: onaylandı):** Kullanıcı, kayıt dosyalarının programı kullanırken masaüstünde oluştuğunu bildirdi ve bu doğrulandı: `portfolio.json` masaüstünde bulunuyor (2025-12-01 tarihli, iki bekleyen emir ve kilitli tutar içeriyor), proje klasöründe ise yok. `varliklar.json` ise **hiçbir yerde bulunamadı** — varlık listesi her açılışta varsayılana düşüyor. Ayrıca masaüstünde, güncel sürümün hiçbir yerinden okunmayan eski bir `trade_history.json` duruyor.
  **Karar:** Saklama yeri bağımsızlığı kısıtı genişletildi (bkz. Constraints) ve **AC18** koşulsuz kriter olarak yürürlüğe girdi.
  **Kapsam düzeltmesi (rev.5, P6 sonrası):** Takım Yöneticisi programı yerelde kullanmadığını bildirdiği için bu bulgu, **bildirilen kaybın nedeni değildir**; masaüstündeki dosyalar geçmiş bir yerel çalıştırmadan kalmadır ve AC01b'nin referansı olarak kullanılmaz. Bulgu yine de kodda gerçek bir kusurdur: kayıt yolları göreli olduğu için kayıt yeri, süreci başlatan çalışma dizinine bağlıdır. Kısıt ve AC18 bu nedenle korunur.
  **Gerekçe:** Her iki kayıp mekanizmasının (yayında yeniden başlatma, yerelde çalışma dizini) çözümü aynıdır: kayıtların sabit ve ortamdan bağımsız bir yerde tutulması. AC18 bu özelliği ucuz ve tekrarlanabilir biçimde kanıtlar.
- **P5 — yayın ortamı (KARAR: yayın mevcut ve aktif kullanımda):** Takım Yöneticisi, Streamlit yayınının var olduğunu ve hâlen kullanıldığını bildirdi. AC05b ve AC19 bu ortamda yürütülür; ölçüm koşulu mevcuttur. Yayın adresi ve erişimi, ölçümlerin yapılabilmesi için QA oturumuna iletilecektir (işletme adımı; spec'i bloke etmez).
- **P6 — referans ortam (KARAR: yayın):** Takım Yöneticisi, programı yerelde kullanmadığını, yalnız Streamlit yayını üzerinden kullandığını ve **kaybın orada yaşandığını** bildirdi. Buna göre bu işin tek gerçek kullanım ve doğrulama ortamı yayındır; AC01b'nin referansı yayındaki kayıtlardır.
  **Sonuç — kayıp mekanizması:** Bildirilen kayıp, kayıtların yayın ortamının kendi diskinde tutulmasından kaynaklanıyor olabilir; yayın yeniden başlatıldığında veya yeniden yayınlandığında bu kayıtlar başlangıç durumuna döner. Doğrulaması **AC19**'dur.
  **Zamana duyarlı uyarı:** AC01b'nin pozisyon tarafı referans kopyası yayındaki kayıtlardan **geliştirme başlamadan önce** alınmalıdır; her kayıp olayı bu referansı yok eder.
- **P7 — kayıp mekanizması, QA kanıtı (bilgi; karar gerektirmez):** Takım Yöneticisi, varlık listesinin kodda bulunduğunu bildirdi; doğrulandı ve kayıp zinciri şöyledir:
  1. Kayıt dosyaları göreli adlarla tanımlı (`portfolio.json`, `varliklar.json`) ve **ikisi de sürüm kontrolünde izlenmiyor** (`git ls-files` ile doğrulandı).
  2. Yayın, sürüm kontrolündeki içerikten kurulduğu için her kurulumda bu iki dosya **yoktur**.
  3. Uygulama bu durumda koddaki yerleşik değerlere döner: 12 girişlik yerleşik varlık listesi ve boş pozisyon listesi + 1000 birimlik başlangıç bakiyesi.
  4. Kullanım sırasında eklenen varlık ve pozisyonlar yalnız yayın ortamının kendi diskine yazılır; ortam yeniden başladığında veya yeniden yayınlandığında 3. adıma dönülür.
  **Durum:** 1. ve 3. adımlar kod üzerinde doğrulanmıştır; 2. ve 4. adımlar yayında ölçülerek kesinleşecektir — doğrulaması **AC19**'dur. Bu madde bir karar değil, PLAN'a girdi olarak bırakılan kanıttır.
  **Sonuç:** Kullanıcının gözlemi ("eklediğim varlıklar kayboluyor, liste eski hâline dönüyor") bu zincirle birebir örtüşmektedir.
- **P8 — yayın erişimi (QA F3; KARAR: yayın herkese açık, kısıtlanacak):** Kayıtlar artık kalıcı olduğu için risk
  profili değişti: eskiden yayının diskindeki kayıtlar yeniden başlatmada siliniyordu,
  şimdi paylaşılan bir veritabanında kalıcı. Uygulamada kimlik doğrulama yoktur; yayın
  herkese açıksa adresi bilen herkes portföyü görebilir, pozisyon açabilir, emir iptal
  edebilir ve sıfırlayabilir — bu değişiklikler artık kalıcıdır.
  **Öneri:** Yayın herkese açıksa Streamlit'in kendi erişim kısıtlaması (özel uygulama /
  izleyici listesi) açılsın. **Gerekçe:** Uygulamaya yetki katmanı eklemek bu spec'in
  kapsamı dışındadır; kalıcı depo koruma sorununu çözerken yetkisiz kalıcı değişiklik
  yüzeyi açar ve bu yüzey yayın ayarından kapatılabilir.
  **Karar (rev.9):** Takım Yöneticisi yayının herkese açık olduğunu bildirdi ve erişimi
  Streamlit tarafından kısıtlayacağını belirtti. Kodda değişiklik yapılmaz; kısıtlama
  yapılana kadar yayına gerçek portföy verisi girilmemelidir. Uygulama içi giriş/parola
  ayrı bir spec konusudur.
  **Durum:** Kısıtlama Takım Yöneticisi tarafından uygulandı; P8 kapandı.
- P1–P8 karara bağlanmıştır (rev.9). Uygulama önünde açık CLARIFY konusu kalmamıştır.

## Acceptance Criteria
> Her satır bağımsız başlangıç koşuluyla doğrulanır; her kriter için ayrı test/kanıt sağlanır.
> **Başlangıç koşulu kuralı (G08):** Koruma senaryolarında başlangıç durumu, mevcut otomatik kapanma kurallarının tetiklenemeyeceği şekilde kurulur (hedef ve zarar kes seviyeleri güncel fiyattan uzak seçilir). Otomatik kapanmanın kendisi kayıp sayılmaz.

- [ ] **AC01 — Hazırlanmış kayıtların korunması:** Geliştirme öncesinde hazırlanmış bir varlık ve ona bağlı bir aktif pozisyon içeren başlangıç durumundan uygulama geliştirme sonrası açıldığında, her iki kayıt ve kayıtlı bilgileri aynen bulunur. Bu kriter kullanıcı ortamında gerçek kayıt bulunup bulunmamasından bağımsız olarak yürütülür.
- [ ] **AC01b — Gerçek kullanıcı kayıtlarının karşılaştırılması (referans: yayın, P6):** Geliştirme başlamadan önce **yayındaki** varlık ve aktif pozisyon kayıtlarının bir kopyası kanıt olarak alınır; geliştirme sonrası aynı yayında bu kopyadaki her kayıt ve bilgisi birebir bulunur. Kopya, kayıtların hâlâ mevcut olduğu bir anda alınır. *(QA notu: varlık tarafının başlangıç referansı ayrıca kopya gerektirmez — yayın her açılışta koddaki yerleşik listeye döndüğü için referans o listedir, bkz. P7. Kopya yalnız pozisyon tarafı için gereklidir. Masaüstündeki yerel dosyalar bu kriterin referansı değildir — bkz. P4 kapsam düzeltmesi.)*
- [ ] **AC02 — Pozisyonsuz varlık:** Hiç pozisyonu olmayan bir varlık başarıyla eklendikten sonra uygulama kapatılıp açıldığında aynı varlık listede bulunur.
- [ ] **AC03 — Yenileme:** Bir varlık ve aktif pozisyon bulunan sayfa yenilendiğinde iki kaydın bilgileri yenileme öncesiyle aynıdır.
- [ ] **AC04 — Yeniden açma:** Bir varlık ve aktif pozisyon bulunan uygulama kapatılıp yeniden açıldığında iki kaydın bilgileri kapanış öncesiyle aynıdır.
- [ ] **AC05a — Süreç yeniden başlatma (G07):** Bir varlık ve aktif pozisyon bulunan uygulamanın süreci tamamen sonlandırılıp yeniden başlatıldığında iki kayıt, miktar ve alış bilgileri dahil, öncekiyle aynıdır.
- [ ] **AC05b — Uyku sonrası, tek seferlik (G07):** Yayındaki uygulama uyku sonrası açıldığında AC05a ile aynı sonuç ekran görüntüsüyle kanıtlanır; ölçüm bir defa yapılır ve tarihiyle kaydedilir.
- [ ] **AC06 — Uzun süre kullanmama (G06):** Kayıtların yaşı 30 gün öncesine ayarlanmış bir başlangıç durumundan uygulama açıldığında iki kayıt bilgileriyle bulunur.
- [ ] **AC07 — Boş durum (G14):** Hiç varlık veya pozisyon eklenmemiş uygulama yeniden açıldığında boş durum açıklaması gösterilir; listede hiçbir varlık adı görünmez ve kendiliğinden kayıt oluşmaz.
- [ ] **AC08 — Alt sınır:** Tek bir pozisyonsuz varlık içeren uygulama uyku sonrası açıldığında varlık sayısı bir olarak kalır.
- [ ] **AC09 — Kısmi kapanış sınırı:** Kalan miktarı sıfırdan büyük olacak şekilde kısmen kapatılmış bir pozisyonun varlığı silinmek istendiğinde silme engellenir ve kayıtlar korunur.
- [ ] **AC10 — Tam kapanış sınırı:** Son aktif pozisyonunun kalan miktarı sıfıra inmiş bir varlık için silme işlemi yapılmadan uygulama yeniden açıldığında varlık listede kalır.
- [ ] **AC11 — Açık silme (G10):** Listede birden fazla varlık varken, aktif pozisyonu ve bekleyen emri olmayan bir varlık kullanıcı tarafından açıkça silindikten sonra uygulama yeniden açıldığında o varlık listede bulunmaz.
- [ ] **AC11b — Son varlık sınırı (G10):** Listedeki son varlık silinmek istendiğinde işlem engellenir, varlık korunur ve nedeni kullanıcıya açıklanır.
- [ ] **AC11c — Toplu sıfırlama sınırı (G11):** Aktif pozisyonu olan bir varlık varken liste sıfırlama denendiğinde işlem yapılmaz ve neden açıklanır; sıfırlama her durumda kullanıcının ayrıca onayını ister.
- [ ] **AC11d — Portföy sıfırlama sınırı (rev.8):** Aktif pozisyonu, bekleyen emri veya sorunlu kaydı bulunan bir portföyde sıfırlama denendiğinde işlem yapılmaz ve neden açıklanır; sıfırlama her durumda kullanıcının ayrıca onayını ister.
- [ ] **AC12 — Silme engeli:** Aktif pozisyonlu bir varlık için silme girişiminde bulunulduğunda varlık ve pozisyon korunur, kullanıcıya aktif pozisyon nedeniyle silinemediği açıklanır.
- [ ] **AC12b — Bekleyen emir engeli (G02):** Yalnız bekleyen emri olan bir varlık için silme girişiminde bulunulduğunda işlem engellenir; emir ve kilitli tutar korunur, neden kullanıcıya açıklanır.
- [ ] **AC12c — Sahipsiz aktif kayıt yokluğu (G13):** Başlangıçta varlığıyla bağlantısı geçerli olan aktif pozisyonlar ve bekleyen emirler, silme engelleri yürürlükteyken karşılığı olmayan bir varlığa işaret eder duruma gelmez. Kriter yalnız aktif pozisyon ve bekleyen emirleri kapsar; tamamen kapatılmış pozisyonların geçmiş kayıtları ile bu iş öncesinden gelen sahipsiz kayıtlar kapsam dışıdır (bkz. KAPSAM DIŞI).
- [ ] **AC13 — Görünüm değişimi (G04):** İki varlık kayıtlıyken Enstrüman seçimi bir varlıktan diğerine değiştirilip geri dönüldüğünde iki varlık da önceki bilgileriyle listede kalır.
- [ ] **AC14 — Görünürlük:** Biri tamamen kapanmış, diğeri aktif iki pozisyon bulunan durumda Aktif Pozisyonlar görünümünde yalnızca aktif olan yer alır; kapalı pozisyonun varlığı Varlık Yönetimi'nde görünmeye devam eder.
- [ ] **AC15 — Geçersiz girdi:** Mevcut giriş kurallarınca reddedilen bir varlık ekleme girişimi sonrasında mevcut varlık ve pozisyon bilgileri değişmez, yeni varlık oluşmaz ve kullanıcıya anlaşılır açıklama gösterilir.
- [ ] **AC15b — Aynı ad (G12):** Kayıtlı bir adla yeniden ekleme denendiğinde mevcut varlığın bilgileri değişmez ve kullanıcıya aynı adın kayıtlı olduğu açıklanır.
- [ ] **AC16 — Erişim sorunu:** Kayıtlı bir varlık ve aktif pozisyona geçici olarak erişilemediğinde kullanıcıya erişim sorunu gösterilir; erişim düzeldiğinde iki kayıt önceki bilgileriyle bulunur.
- [ ] **AC16b — Erişim sorununda yazma engeli (G03):** Erişim sorunu sırasında bir kayıt değiştirme girişimi yapıldığında işlem gerçekleşmez ve erişim düzeldiğinde önceki kayıtlar bilgileriyle bulunur.
- [ ] **AC16c — Eksik bilgili kayıt (G15, R4.1):** Bilgileri eksik bir kayıt bulunduğunda kayıt korunur, boş kayıtla değiştirilmez, kullanıcıya sorunlu olduğu bildirilir ve ilişkili varlığın silinmesi engellenir.
- [ ] **AC17 — Performans (G09, P1 onaylı):** Uygulama hazırken 100 varlık ve toplam 100 aktif pozisyonun görünmesi, aynı koşullarda yapılan 10 ölçümün her birinde en fazla 5 saniye sürer. Ölçüm, kayıtların okunmasından listenin görünmesine kadar geçen süredir; canlı piyasa fiyatı çekme süresi ölçüme dahil edilmez. Ölçüm yerel geliştirme ortamında alınır.
- [ ] **AC18 — Başlatma biçiminden bağımsızlık (P4, onaylı):** Bir varlık ve aktif pozisyon kaydedildikten sonra uygulama farklı bir klasörden başlatıldığında aynı kayıtlar bilgileriyle bulunur. Yayında tek bir başlatma biçimi olduğundan bu kriter geliştirici ortamında yürütülür; ölçüm sınırı AC17 ile aynı mantıktadır.
- [ ] **AC19 — Yeniden yayınlama sonrası koruma:** Bir varlık ve aktif pozisyon kayıtlıyken uygulama yeniden yayınlandıktan sonra, yayın öncesindeki iki kayıt bilgileriyle bulunur. Kriter süreç yeniden başlatma (AC05a) ve farklı klasörden başlatma (AC18) kriterlerinden ayrı yürütülür; bu ikisi yeniden yayınlamayı tek başına kanıtlamaz.

## Definition of Done
- [x] P1–P7 için Takım Yöneticisi kararları spec'e işlendi; spec uygulama için onaylandı (rev.8).
- [x] **P8:** Yayın erişimi Streamlit ayarlarından kısıtlandı (Takım Yöneticisi tarafından uygulandı).
- [ ] Tüm kabul kriterleri ayrı test/kanıtla karşılandı; uygulanamaz kriter bırakılmadı.
- [ ] İş davranışı testleri P3 kararına uygun yürütüldü: her kabul kriteri için `tests/` altında en az bir pytest testi yazıldı ve `.github/workflows/tests.yml` ile yeşil geçti.
- [ ] Her arayüz kriteri için ekran görüntüsü ve kritik ekleme → uyku → yeniden açma akışı için smoke test kanıtı üretildi.
- [ ] Streamlit yayınında uyku öncesi/sonrası kayıt karşılaştırmasıyla koruma doğrulandı (AC05b).
- [ ] Geçerli pipeline kontrolleri geçti ve kullanıcıya teknik hata sızmadı.
- [ ] **Para/yüzde kapsamı (G16):** Bu iş kayıtları koruma işidir; mevcut para/yüzde gösterimi olduğu gibi korunur, sayı türü dönüşümü yapılmaz. Genel para kuralının bu uygulamaya uyarlanması ayrı bir spec konusudur.
- [ ] Ayrı QA oturumunda bağımsız doğrulama yapıldı; Takım Yöneticisi sonucu değerlendirdi.
- [ ] PR ve squash-merge işlemleri [git.md](../docs/git.md) kurallarına göre tamamlandı.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 11 — rev.1 QA bulguları; rev.2 P4; rev.3 TY bulguları; rev.4 P5; rev.5 P6; rev.6 P7 kanıtı; rev.7 P3; rev.8 P1/P2 + AC11d/R4.1; rev.9 QA denetimi (F6 sınırı, P8); rev.10 ikinci QA turu (F10/F11); rev.11 merge + SCORECARD |
| Düzeltme turu sayısı | 3 — 1: spec metni; 2 ve 3: QA kod denetimleri sonrası düzeltmeler |
| Bulgu gerçek/gürültü oranı | 17/0 spec incelemesi + 9/0 ve 2/0 kod denetimleri (triyajdan geçen) |
| Regresyon sayısı | 7 — hepsi merge öncesinde yakalandı ve düzeltildi: 5 spec-0001 testi (boş kayıtta uygulamanın durması, AC07 davranış değişikliği); 1 test izolasyon kusuru (kırık depo taklidinin diğer test dosyalarına sızması); 1 spec-0003 uyum kusuru (`CLOSED_CONFIRMED`) |
| Test sayısı | 366 yeşil (bu iş 62 test ekledi); CI `pytest` 2 dk 1 sn |
| Ölçüm | AC17: 10 ölçümün en uzunu **0,001 sn** (sınır 5 sn) |
| Kaçan hata | 3 (spec metni, rev.1) + **5 (kod, Developer)** — QA denetiminde bulundu: F1 yazma koparsa bellekteki değişiklik geri alınmıyor; F2 oturum ortasında kopan erişimde yazan kontroller açık kalıyor; F4 bilgisi eksik bekleyen emir sayfayı düşürüyor; F5 yerel arka uçta yanlış "korunuyor" rozeti; F8 dosya adlarının ikinci gerçek kaynağı. **+1 (2. tur):** F10 — F1 düzeltmesinin kendi açtığı kusur: otomatik kapanış sonrası anlık görüntü tazelenmediği için meşru kapanış geri alınabiliyordu. **Toplam kaçan hata: 3 metin + 6 kod.** Hepsi için, düzeltme geri alındığında başarısız olan regresyon testi yazıldı |
