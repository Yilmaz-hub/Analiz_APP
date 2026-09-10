# Spec: 0005 — Performans, Strateji, Risk ve Sanal Doğrulama

> Şablon: [TEMPLATE.md](TEMPLATE.md). Revizyon 1 — bağımsız QA denetimi sonrası boşluklar kapatıldı: referans ortam, zaman otoritesi, rejim sınıfları, miktar formülü, hata durumları ve sınır değerleri yazıldı. Q01–Q08 kararları hala onay bekliyor. Uygulama onayı değildir.
> Rol: Analist. Bu belge önceki yol haritasının 4–7. adımlarını kapsar; V1 uygulama planındaki aynı numaralı teknik adımları ifade etmez.

## Intent
Kullanıcı, dalgalı piyasalarda gördüğü AL sinyaline ne ölçüde güvenebileceğini geçmiş ve ileri dönem sonuçlarıyla değerlendirmek istiyor.
Önceki CLARIFY kararlarına uygun olarak günlük mumlarla kripto spot, BIST, Amerikan hisseleri, altın ve kullanıcının bu gruplarda eklediği varlıklar izlenecek; mevcut Yahoo Finance ve XAU kullanımı korunacaktır.
Kullanıcı uygulamayı değişken saatlerde kontrol eder, her AL sinyalini işleme dönüştürmez, başlangıç stopunu esas alır ve trend sürdükçe pozisyonu tutup SAT geldiğinde piyasa fiyatından çıkmayı amaçlar; gerçekleşmiş işlemleri sonradan kaydedebilmesi korunacaktır.
Başarı, mevcut yaklaşımı alternatiflerle maliyetleri ve kayıp riski görünür biçimde karşılaştırmak, kâr koruma seçeneklerini ölçmek ve seçilen yaklaşımı ileri dönemde izlemektir.
Bu çalışma en dipte alış, en tepede satış veya her AL sinyalinde kazanç vaat etmez; yeni stratejinin kullanılmasına ancak önceden belirlenmiş değerlendirme koşulları ve kullanıcı kararıyla geçilir.

## Requirements
### Adım 4 — Güvenilir performans raporu
- **R01:** Mevcut AL→SAT yaklaşımı, V1'in stop ve bekleme kurallarıyla ayrı bir referans olarak korunmalı; aday yaklaşımlar ve al-tut aynı varlık, dönem, başlangıç sermayesi ve açıklanmış maliyet koşullarında karşılaştırılmalıdır.
- **R02:** Rapor net getiri, sermayenin zirveden en büyük düşüşü, kapanmış işlem sayısı ve kapanmış işlem başına ortalama net sonucu göstermelidir. Açık pozisyonlar dönem sonu sermaye değerine dahil edilmeli, gerçekleşmiş sonuçtan ayrılmalıdır.
- **R03:** Getiri dışarıdan para giriş/çıkışıyla şişirilmemeli; doğrulanamayan maliyet veya değerleme eksikleri sonucun yanında görünmelidir. Eksik maliyet sıfır kabul edilmemelidir: maliyeti bilinmeyen işlem içeren metrik **üst sınır** olarak etiketlenir, eksik maliyetli işlem sayısı yanında görünür ve sonuç temiz net sonuç gibi sunulmaz. İlgili işlemler örneklemden sessizce dışlanmaz.
- **R04:** Sonuçlar piyasa, varlık, para birimi, dönem, strateji sürümü ve yükselen/düşen/yatay piyasa koşuluna göre süzülebilmelidir. Filtrelenmiş örnek sayısı görünmelidir.
- **R05:** Ayar seçiminde kullanılan dönem ile seçimde kullanılmamış değerlendirme dönemi ayrı gösterilmeli; sonraki bilgiler geçmiş kararları etkilememelidir.
### Adım 5 — Çalkantıya uygun strateji denemeleri
- **R06:** Mevcut filtrelerin etkisi, aynı koşullarda yalnız incelenen filtrenin değiştiği karşılaştırmalarla ölçülmelidir. Kırılım ve yükseliş içi geri çekilme girişleri ayrı adaylar olarak değerlendirilebilmelidir.
- **R07:** Her adayın giriş, çıkış, piyasa koşulu, ayarları, karşılaştırma ölçütleri ve değerlendirme dönemi sonuç görülmeden önce belirlenmelidir; denenen başarısız adaylar da değerlendirme geçmişinde kalmalıdır.
- **R08:** Adaylar getiri, düşüş ve işlem beklentisi üzerinden karşılaştırılmalı; yeterli veri bulunmaması, ölçütü karşılamama ve ölçütü karşılama farklı sonuçlar olarak sunulmalıdır. Bir piyasadaki başarı diğer piyasalara aktarılmamalıdır.
- **R09:** Mevcut strateji varsayılan kalmalıdır. Ölçütleri karşılayan aday dahi kullanıcının açık seçimi olmadan aktif sinyal davranışını değiştirmemelidir; geçmiş sonuçlar gelecekteki kazanma olasılığı olarak sunulmamalıdır.
### Adım 6 — Pozisyon riski ve kâr koruma
- **R10:** Kullanıcı işlem başına risk bütçesi ile toplam açık pozisyon risk sınırını belirleyebilmeli ve bu ayar kalıcı olmalıdır. Önerilen miktar giriş-stop mesafesi, bilinen maliyetler, kullanılabilir nakit ve ürünün miktar koşullarıyla sınırlandırılır: miktar = **aşağı yuvarla**( risk bütçesi / (giriş − stop + birim başına bilinen maliyet), ürün miktar adımı ). Sonuç miktar adımının veya ürünün asgari işlem tutarının altında kalırsa miktar üretilmez, nedeni gösterilir. Yukarı yuvarlama yasaktır.
- **R11:** Kullanıcı belirlenen dönem için kayıp sınırı koyabilmeli; bu sınır dolduğunda yeni giriş önerileri durmalı, mevcut pozisyonların koruma ve çıkış bilgileri görünmeye devam etmelidir. Sınır yalnız kullanıcının açık ve kayda geçen eylemiyle sıfırlanır; otomatik sıfırlama yoktur, kapanan dönem ile yeni dönem ayrı görünür.
- **R12:** Eksik sermaye, geçersiz stop (stop ≥ giriş veya stop ≤ 0), bilinmeyen ürün adımı veya karşılaştırılamayan para birimleri için doğrulanmış risk büyüklüğü üretilmemeli; miktar hiç önerilmez ve engelin nedeni gösterilir. Toplam risk ve kayıp sınırı yalnız aynı para biriminde hesaplanır; stopu girişin üstüne taşınmış pozisyonun riski negatif sayılmaz, sıfıra yuvarlanır. Stop bütçesi, fiyat boşluklarında azami gerçekleşmiş kayıp garantisi olarak sunulmamalıdır.
- **R13:** Sabit stop + SAT referansı; sabit hedef, iz süren stop ve kademeli çıkış adaylarıyla ayrı ayrı karşılaştırılabilmelidir. Aday koruma kuralları mevcut gerçek pozisyonun stopunu veya miktarını değiştirmemelidir.
- **R14:** Gerçek stop yükseltmesi ve satış, kullanıcının teyitli geçerlilik zamanına göre işlenmelidir; geçerlilik anının kendisi dahildir (≥). Sanal kademeli çıkışta kapanan miktar ve kalan pozisyon ayrı izlenmelidir.
### Adım 7 — Kesintisiz sanal doğrulama
- **R15:** Seçilen varlık ve sabitlenmiş strateji sürümü, kullanıcının ekranı açık tutmasına bağlı olmadan her uygun günlük kapanış sonrasında ileri dönem sanal takibe alınmalıdır. Takip, arayüz sürecinden bağımsız çalışan bir koşucu ile yürütülür (bkz. Constraints — çalışma ortamı).
- **R16:** Her sanal kararın veri kaynağı, değerlendirme zamanı, dayanak mum zamanı, strateji sürümü ve gerçekleşme varsayımları izlenebilmelidir. Sonradan tamamlanan dönemler zamanında izlenen dönemlerden ayrı işaretlenmelidir.
- **R17:** Aynı karar yeniden işlendiğinde ikinci sanal işlem oluşmamalıdır; tekillik anahtarı (varlık, strateji sürümü, dayanak mum zamanı) depoda zorlanır ve eşzamanlı ikinci yazım hata değil yok sayma üretir. Sağlayıcı sonradan bir mumu revize ederse ilgili karar yeniden hesaplanmaz, dondurulmuş kalır ve revizyon ayrı kayıtla işaretlenir. Kesinti ve veri hataları görünmeli; izlenmeyen dönemler kesintisiz başarı kanıtına katılmamalıdır.
- **R18:** İleri dönem sonucu, onaylanan asgari süre, işlem sayısı ve piyasa koşulu kapsaması tamamlanmadan yeterli kanıt olarak sunulmamalıdır. Strateji değişikliği yeni bir değerlendirme dönemi başlatmalı; önceki sürümün sonuçları korunmalıdır.
- **R19:** Gerçek teyitli işlemler, geçmiş simülasyon ve ileri dönem sanal işlemler ayrı raporlanmalıdır. Kullanıcının AL sinyalini uygulamaması gerçek işlem oluşturmaz; sonradan kayıt, işlem zamanı ile kayıt zamanını ayırır.

## Constraints
- Kapsam günlük, uzun yönlü ve kaldıraçsız işlemlerdir; kripto için OKX/Binance spot bağlamı ve mevcut Yahoo Finance/XAU kaynak tercihleri korunur. Diğer mevcut zaman aralıkları kaldırılmaz, bu spec'in doğrulama kapsamı sayılmaz.
- ETH/USD ile ETH/USDT sinyal/stop değerlendirmesinde önceki onaya göre eşdeğer kabul edilebilir; bu karar TL/USD/USDT parasal sonuçlarını kur dönüşümü olmadan toplama yetkisi vermez.
- Para ve yüzde hassasiyeti [conventions.md](../docs/conventions.md) kurallarına uyar. Ölçülemeyen metrik sıfır değil **yok değeri** taşır ve arayüzde “hesaplanamıyor” görünür.
- **Zaman otoritesi:** Tüm karar, teyit ve kayıt zamanları **UTC** saklanır; kullanıcıya yerel saat yalnız gösterimde çevrilir. “Günlük kapanış” varlığın piyasa seansına göre tanımlanır (BIST, ABD hisse ve XAU için ilgili seans kapanışı; kripto için sabitlenmiş bir UTC kesim saati). Karşılaştırmalarda zaman sınırları kapsayıcıdır (≥).
- **Sermaye ölçümü:** Getiri ve maksimum düşüş **günlük kapanış** sermayesi dizisi üzerinden hesaplanır; gün içi uç değerler ve işlem anı anlık değerleri dizide yer almaz.
- **Çalışma ortamı (Q06a):** Ekran kapalıyken süren ileri dönem takibi, arayüz sürecinden **bağımsız, başsız (headless) bir koşucu** tarafından işletim sistemi zamanlayıcısı ile tetiklenerek yürütülür; arayüzün açık olması koşul değildir. Bu ortamın kurulum ve işletim maliyeti plan aşamasında ayrıca onaylanır.
- **Referans ölçüm ortamı:** Süre bütçeleri; makine sınıfı (CPU/RAM) yazılı tek bir ortamda, canlı sağlayıcı yerine **sabit veri fixture’ı** ile ölçülür. İlk çalışma (soğuk önbellek) ile sonraki çalışma ayrı raporlanır; 10 tekrarın **en kötüsü** sınırı sağlamak zorundadır. Sağlayıcı bekleme süresi ölçüme karıştırılmaz, ayrı raporlanır.
- Yeni adaylar V1'in varsayılan kurallarından ayrı değerlendirilir; kullanıcı seçimi olmadan mevcut gerçek pozisyonlara yeni hedef, iz süren stop veya kademeli çıkış uygulanamaz.
- Gelecek verisini geçmiş değerlendirmeye katmak, maliyeti bilinmeyen sonucu tam net sonuç gibi göstermek, başarısız adayları karşılaştırmadan saklamak ve puanı kanıtlanmış kazanma olasılığı gibi sunmak yasaktır. Saat/tarih ileri alınarak üretilen hızlandırılmış gözlem ileri dönem yeterlilik sayacına katılmaz.
- **Performans hedefi — Q06b önerisi, onay bekliyor:** Yukarıdaki referans ölçüm ortamında: hazır 10.000 işlem kaydında rapor/filtre sonucu **2 saniye**; 20 varlık × 1.000 günlük mum × 3 aday değerlendirmesi **120 saniye**; kullanılabilir günlük veri geldikten sonra ileri dönem kararı **5 dakika** içinde kayda alınır. Kesinti başarı kabul edilmez.
- **KAPSAM DIŞI:** Gerçek emir gönderimi; kaldıraç ve açığa satış; gün içi yeni stratejiler; garantili getiri; kullanıcı onayı olmadan stratejiye geçiş; geçmiş kayıp kayıtların kurtarılması; bağımsız bir veri sağlayıcı değiştirme projesi.
- Aşama sırası 4 → 5 → 6 → 7'dir; raporlama ve V1 tutarlılığı kanıtlanmadan aday üstünlüğü ilan edilmez. Her aşama ayrı kabul kanıtı ve uygulama dilimi gerektirir.

## Context
- İlgili alanlar: performans karşılaştırması, sinyal değerlendirmesi, pozisyon riski, sanal takip ve gerçek işlem teyidi.
- Kaynak: [0003 V1 spec'inin mevcut çalışma kopyası](../.worktrees/feature-0003-islem-tutarliligi/specs/0003-islem-kararlari-ve-performans-tutarliligi.md), özellikle “KAPSAM DIŞI (V1)” adım 4–7. Kullanıcının eski 0002 bağlantısı şu anda diskte yoktur; 0002 numarası mobil grafik spec'inde kullanılmıştır. Bu nedenle boş olan 0005 numarası seçilmiştir.
- Bağımlılık: 0003'ün karar/veri/gerçekleşme tutarlılığı bağımsız kanıtla doğrulanmalıdır; bu belge onun tamamlandığı iddiası değildir. [0004 kayıt koruma](0004-varlik-ve-pozisyon-kayitlarini-koruma.md) kalıcı kayıt davranışının ilgili bağımlılığıdır.
- Terimler: “net sonuç” bilinen tüm işlem maliyetleri sonrası sonuç; “maksimum düşüş” dönem içindeki toplam sermaye değerinin önceki zirvesine göre en büyük yüzdesel düşüşü; “işlem beklentisi” kapanmış pozisyonların ortalama net sonucudur. Kademeli satışlar tek pozisyonun parçalarıdır; pozisyon ancak son parça kapanınca kapanmış işlem sayılır. “Aday parmak izi” bir adayın tüm ayarlarından türeyen kimliktir; parmak izi değişirse strateji sürümü değişmiş sayılır. “İzlenen gün” ileri dönem takibinin fiilen çalıştığı gündür; kesintili günler izlenen gün sayılmaz.
- `docs/domain.md` ve `docs/architecture.md`, alakasız bir projeye ait oldukları için Takım Yöneticisi tarafından kaldırılmıştır; bu belge onlara dayanmaz. Alan terimleri yukarıdaki **Terimler** satırındadır. `docs/testing.md` 2026-09-10'da bu projenin yığınına göre yeniden yazılmıştır (pytest + görsel kanıt).

### CLARIFY — uygulama öncesi zorunlu kararlar
Önceki onaylar Intent'te korunmuştur. Aşağıdaki yeni öneriler ve bunlara bağlı kriterler karar verilene kadar taslaktır; sayısal örnekler gerçek hesap için yatırım önerisi değildir.

| Karar | Cevabı gereken soru | Öneri | Gerekçe |
|---|---|---|---|
| Q01 | Karşılaştırma dönemi, al-tut miktarı ve sermaye hareketleri nasıl ele alınsın? | Mevcut ortak tarihlerde son 3 yıl; ilk %60 ayar seçimi, son %40 dokunulmamış değerlendirme. Her iki tarafta aynı başlangıç sermayesi, al-tutta ilk uygun açılışta maliyet sonrası alınabilen miktar; dış nakit hareketi olan dönemler bu sürümde karşılaştırılamaz. Bölme tam olmadığında ayar dilimi **aşağı yuvarlanır**, kalan tarihler değerlendirmeye gider. Toplam uygun tarih **250'nin altındaysa** dönem “yetersiz geçmiş” sayılır ve karşılaştırma yeterli kanıt olarak sunulmaz. Al-tutta bölünmeyen sermaye kalıntısı nakit olarak tutulur ve dönem sonu sermayesine eklenir. | Başlangıç koşullarını eşitler; para yatırmayı strateji getirisi saymaz. Yuvarlama yönü yazılmazsa değerlendirme dilimi ayar seçimine sızar; kalıntı nakit yok sayılırsa al-tut referansı haksız düşük çıkar. |
| Q02 | Yükselen/düşen/yatay sınıfları ve giriş adaylarının kesin kuralları ne olsun? | Kodda üç sınıflı bir tanım **yoktur**: [config.py](../config.py) yalnız 0–100 rejim skoru (`RegimeConfig`) ve MA üstü/altı ikili filtresi (`REGIME_MA_PERIOD`) içerir. Bu nedenle üç sınıf, mevcut skor ve MA konumundan türeyen bir **rejim sınıflandırıcısı** olarak tanımlanır; eşikleri ve sınıflandırıcı sürümü sonuçlar görülmeden sabitlenir ve her gözleme sürümüyle yazılır. Kırılım ve geri çekilmenin giriş/çıkış eşikleri de aday bazında, sonuç görülmeden ayrı karar kaydında onaylanır. | Var olmayan bir tanıma atıf yapmayı önler; eşikler sonuçtan sonra seçilirse Constraints'in “sonuca göre tanım değiştirme” yasağı ihlal edilir ve AC41 ölçülemez. |
| Q03 | Bir aday hangi koşulda tercih edilmeye uygun sayılsın? | Aynı değerlendirme kesitinde referanstan daha yüksek maliyet sonrası getiri, daha büyük olmayan maksimum düşüş ve pozitif işlem beklentisi birlikte aransın; en az 30 kapanmış işlem altı yetersiz sayılsın. Al-tut karşılaştırması ayrıca gösterilsin. Sınırlar: getiri **kesin büyük** olmalı (eşitlik yetmez), maksimum düşüş **büyük olmamalı** (eşitlik engel değil), beklenti **kesin pozitif** olmalı (tam 0 yetmez). | Tek başına yüksek getiriyle riski veya küçük örneklemi gizlemeyi önler; 30 işlem istatistiksel kesinlik garantisi değildir. Karşılaştırma yönü yazılmazsa her ölçütte iki farklı doğru cevap oluşur. |
| Q04 | İşlem/portföy risk yüzdeleri, kayıp dönemi ve karma para birimi davranışı ne olsun? | Varsayılan risk profili tanımsız kalsın; kullanıcı açıkça oran seçsin. Oranlar 0'dan büyük, %100'den küçük olsun; toplam risk aynı para biriminde hesaplanabilsin. Oran, [conventions.md](../docs/conventions.md) yüzde ölçeğinde temsil edilebilir olmalı; ölçek altında kalıp sıfırlanan girdi reddedilir. Kayıp sınırı kullanıcının belirlediği başlangıçtan itibaren sürsün, otomatik sıfırlanmasın; sıfırlama yalnız kullanıcının açık ve kayda geçen eylemiyle olsun. Toplam risk ve kayıp sınırı farklı para birimlerini karıştırmak zorunda kalırsa doğrulanmış sonuç üretilmesin. | Kullanıcının sermayesi ve kayıp toleransı bilinmeden oran atamaz; zaman ve kur belirsizliğini görünür tutar. Sıfırlama yolu yazılmazsa sınır dolduktan sonra uygulamadan çıkış yolu kalmaz. |
| Q05 | Hangi kâr koruma adayları, hangi parametrelerle denensin? | Sabit hedef, yalnız yukarı taşınan iz süren stop ve kademeli çıkış ayrı adaylar olsun; hedef katsayısı, takip mesafesi, kademe oranları ve geçerlilik anları denemeden önce kullanıcı onayıyla sabitlensin. İlk denemede birlikte etkinleştirilmesin. | Sonucu hangi kuralın değiştirdiğini ayırt etmeyi sağlar; gerçek stop alışkanlığını kendiliğinden değiştirmez. |
| Q06a | Ekran kapalıyken takip **hangi ortamda** koşacak? | Arayüzden bağımsız, işletim sistemi zamanlayıcısıyla tetiklenen başsız bir koşucu (bkz. Constraints — çalışma ortamı); kurulum ve işletim maliyeti plan aşamasında ayrıca onaylanır. | Mevcut uygulama tek süreçli bir arayüzdür ([app.py](../app.py)) ve zamanlayıcı içermez; ortam kararı verilmeden AC35 tanımı gereği kırmızı kalır ve Adım 7 dilimi ortasında **AP-10**'a düşer. |
| Q06b | Sayısal süre bütçeleri ve referans ölçüm ortamı kabul ediliyor mu? | Constraints'teki referans ölçüm ortamı (sabit fixture, soğuk/sıcak ayrımı, 10 tekrarın en kötüsü) ve 2 sn / 120 sn / 5 dk bütçeleri onaylansın; sağlayıcı bekleyişi ayrı raporlansın. | “Tanımlı referans ortam” yazılmazsa AC45/AC46 tekrarlanamaz ve ölçüm kodu değil ağı ölçer. |
| Q07 | İleri dönem için yeterlilik eşiği ve test standardı ne olsun? | Her değerlendirilen piyasa ve strateji için en az 90 takvim günü, 30 kapanmış işlem ve üç piyasa koşulunda gözlem birlikte aransın; eksik koşulda süre uzasın. **Test çatısı bölümü UYGULANDI (2026-09-10, yönetici onayı):** [testing.md](../docs/testing.md) bu projenin gerçek yığınına (Python 3 / Streamlit / pytest) göre yeniden yazıldı ve `AGENTS.md`'nin yığın cümlesi düzeltildi; bu spec ayrı bir çatı istisnası taşımaz. **Karar bekleyen kısım:** yeterlilik eşiği — her piyasa ve strateji için en az 90 izlenen gün, 30 kapanmış işlem ve üç piyasa koşulunda gözlem. | Kanıt: kökte `.csproj`/`.sln`/`package.json` yok, `requirements.txt` + `tests/` (pytest) var — yani `docs/testing.md`'in xUnit zorunluluğu ve `AGENTS.md`'in .NET + React yığın cümlesi kodla çelişiyor (**AP-10**). Aynı istisna her spec'te tekrar yazılırsa Altın Kural 7 (tek gerçek kaynağı) bozulur; DoD'daki test satırı bu karara bağlıdır. |
| Q08 | Gerçek işlemin **geriye dönük** kaydı ve strateji sürüm değişimi açık pozisyonu nasıl etkilesin? | Geriye kayıt süre sınırı olmadan serbest kalsın; etkilediği kapanmış dönem raporu “geç kayıtla güncellendi” işaretiyle görünsün ve ileri dönem yeterlilik sayacına katılmasın. Açık gerçek pozisyon **giriş anındaki sürümle** yönetilmeye devam etsin; yeni sürüm yalnız yeni girişlere uygulansın. | Geç kayıt sessizce geçmiş raporu değiştirirse AC12/R05'in “geçmişi dondurma” ilkesi çiğnenir; sürüm devri yazılmazsa aday seçimi mevcut açık pozisyonun kurallarını fiilen değiştirir (R09 ihlali). |

## Acceptance Criteria
> Her kriter kendi verisiyle bağımsız test edilir. Q etiketli kriter önerilen karara bağlıdır; karar değişirse kriter revize edilir. Sayısal örnekler hesap davranışını sınar.

- [ ] **AC01 — Varsayılan:** Mevcut stratejisi seçili kullanıcı ilk açılışta aynı stratejiyi aktif görür; yeni aday seçilmiş olmaz.
- [ ] **AC02 — Boş durum:** Seçili dönemde hiç işlem yoksa kapanmış işlem sayısı 0 ve işlem beklentisi “hesaplanamıyor” görünür.
- [ ] **AC03 — Net getiri:** Dış nakit hareketi olmayan, tüm maliyetleri dahil başlangıç sermayesi 10.000 ve son sermayesi 10.500 olan örnekte net getiri %5 gösterilir.
- [ ] **AC04 — Düşüş:** Dış nakit hareketi olmayan 10.000 → 12.000 → 9.000 → 11.000 sermaye dizisinde maksimum düşüş %25 gösterilir.
- [ ] **AC05 — Beklenti:** Kapanmış iki pozisyonun net sonuçları +100 ve −40 olduğunda işlem başına beklenti +30, ilgili para biriminde gösterilir.
- [ ] **AC06 — Açık pozisyon:** Yalnız bir açık pozisyon bulunan dönemde kapanmış işlem sayısı 0 kalır ve pozisyonun dönem sonu değeri sermayeye katılır.
- [ ] **AC07 — Eksik maliyet görünürlüğü:** Komisyonu bilinmeyen işlem içeren raporda “maliyet eksik” açıklaması sonuç yanında ek tıklama olmadan görünür.
- [ ] **AC08 — Filtre:** ETH ve BIST işlemleri içeren raporda ETH seçildiğinde metrikler yalnız ETH işlemleriyle hesaplanır.
- [ ] **AC09 — Geçersiz tarih:** Başlangıcı bitişinden sonra olan dönem reddedilir ve açıklama gösterilir.
- [ ] **AC10 — Para birimi:** 100 TL ve 100 USD sonuç içeren rapor bunları 200 tutarında tek parasal toplam olarak göstermez.
- [ ] **AC11 — Eşit karşılaştırma:** Başlangıç sermayesi veya dönemi **tam eşit olmayan** iki değerlendirme karşılaştırıldığında (ör. 10.000 ile 10.000,01) eşit koşullarda karşılaştırılamadığı görünür ve tek üstünlük sonucu yayımlanmaz.
- [ ] **AC12 — Gelecek verisi:** Bir karar tarihinden sonraki fiyatlar değiştirildiğinde o tarihin kararı değişmez.
- [ ] **AC13 — Ayrılmış dönem, Q01:** 100 uygun tarih içeren örnekte ilk 60 tarih ayar seçimine, kalan 40 tarih ayrı değerlendirmeye atanır.
- [ ] **AC14 — Nakit hareketi, Q01:** Değerlendirme döneminde dışarıdan para yatırılmışsa önerilen ilk kapsamda strateji getiri karşılaştırması uygun sayılmaz.
- [ ] **AC15 — Tek filtre denemesi:** Bir filtre etkisi karşılaştırmasında diğer ayarları farklı olan adaylar tek filtre etkisi sonucu olarak sunulmaz.
- [ ] **AC16 — Başarısız aday:** Ölçütleri karşılamayan kaydedilmiş aday, değerlendirme geçmişinde başarısız sonucu ile görünür.
- [ ] **AC17 — Otomatik geçiş yok:** Tüm aday başarı ölçütleri sağlandığında kullanıcı seçimi yapılmadan aktif strateji değişmez.
- [ ] **AC18 — Yetersiz örnek, Q03:** 29 kapanmış işlemi olan aday, getiri üstün olsa bile yeterli değerlendirme sayılmaz.
- [ ] **AC19 — Tam örnek sınırı, Q03:** 30 kapanmış işlemi olan aday yalnız işlem sayısı koşulu nedeniyle engellenmez.
- [ ] **AC20 — Üstünlük sınırı, Q03:** Aday getirisi referansa eşitse “daha yüksek getiri” koşulu sağlanmış sayılmaz.
- [ ] **AC21 — Risk ölçütü, Q03:** Adayın getirisi yüksek fakat maksimum düşüşü referanstan büyükse aday tercih ölçütlerini karşılamaz.
- [ ] **AC22 — Piyasa ayrımı:** Kripto değerlendirmesi yeterli, BIST değerlendirmesi yetersiz olan aday için BIST sonucu yeterli gösterilmez.
- [ ] **AC23 — Tanımsız risk, Q04:** Risk profili belirlememiş kullanıcı için risk bazlı miktar **hiç** önerilmez (işaretli tahmin de gösterilmez) ve eksik bilgi açıklanır.
- [ ] **AC24 — Risk miktarı:** Risk bütçesi 100, giriş 100, stop 90, bilinen maliyetler 0, miktar adımı 1 ve yeterli nakit durumunda önerilen miktar 10 olur.
- [ ] **AC25 — Geçersiz stop:** Girişe eşit stop ile risk bazlı miktar istenirse hesap reddedilir ve nedeni gösterilir.
- [ ] **AC26 — Geçersiz oran, Q04:** Risk oranına 0, negatif, %100 veya %100 üstü değer girildiğinde her örnek ayrı sınanarak reddedilir.
- [ ] **AC27 — Toplam risk sınırı:** Toplam risk limiti 300, mevcut risk 200 ve yeni işlem riski 100 iken diğer koşullar geçerliyse toplam risk engeli oluşmaz.
- [ ] **AC28 — Toplam risk aşımı:** Toplam risk limiti 300, mevcut risk 200 ve yeni işlem riski 100,01 iken yeni giriş engellenir.
- [ ] **AC29 — Kayıp sınırı:** Tanımlı dönem kaybı sınıra tam eşit olduğunda yeni giriş engellenir.
- [ ] **AC30 — Çıkış görünürlüğü:** Kayıp sınırı nedeniyle girişler engelliyken açık pozisyonun SAT ve stop bilgileri görünür kalır.
- [ ] **AC31 — Bilinmeyen ürün adımı:** Miktar adımı bilinmeyen varlıkta risk miktarı **hiç** önerilmez ve engelin nedeni gösterilir.
- [ ] **AC32 — Gerçek stop korunur:** Sanal iz süren stop adayı ilerletildiğinde kullanıcının gerçek pozisyonunda kayıtlı stop değişmez.
- [ ] **AC33 — Stop zamanı:** Gerçek stop yükseltmesinin geçerlilik anından **kesin önce** olan fiyat teması yeni stop ile satış teyidi oluşturmaz.
- [ ] **AC34 — Kademeli çıkış:** 10 birimlik sanal pozisyonun 4 birimi satıldığında 6 birim açık kalır ve kapanmış pozisyon sayısı artmaz.
- [ ] **AC35 — Ekrandan bağımsız takip, Q06a:** Arayüz süreci kapalıyken kullanılabilir yeni günlük veri geldiğinde sanal karar bağımsız koşucu tarafından **en çok 5 dakika** içinde kayda alınır.
- [ ] **AC36 — İzlenebilirlik:** Bir ileri dönem kararı açıldığında kaynak, karar zamanı, mum zamanı ve strateji sürümü görünür.
- [ ] **AC37 — Tekrar güvenliği:** Aynı varlık, sürüm ve günlük karar yeniden işlendiğinde sanal işlem sayısı artmaz.
- [ ] **AC38 — Kesinti görünürlüğü:** İzlemenin çalışmadığı dönem sonradan tamamlandığında kayıt “sonradan oluşturuldu” olarak görünür ve zamanında gözlem sayısına katılmaz.
- [ ] **AC39 — Süre alt sınırı, Q07:** Diğer koşullar sağlanmış olsa bile **89 izlenen günlük** ileri takip yeterli sayılmaz.
- [ ] **AC40 — Tam süre sınırı, Q07:** İşlem sayısı ve koşul kapsaması tamamlanmış **90 izlenen günlük** takip süre koşulunu sağlar.
- [ ] **AC41 — Eksik piyasa koşulu, Q07:** Süre ve işlem sayısı tamamlanmış fakat yatay piyasa gözlemi olmayan takip yeterli sayılmaz.
- [ ] **AC42 — Sürüm değişimi:** Yeni strateji sürümüne geçildiğinde yeni sürümün ileri gözlem sayısı önceki sürümden devralınmaz.
- [ ] **AC43 — Gerçek/sanal ayrımı:** Kullanıcının uygulamadığı bir AL sinyalinin sanal alış üretmesi gerçek teyitli işlem sayısını artırmaz.
- [ ] **AC44 — Geç kayıt:** Dünkü gerçek işlem bugün kaydedildiğinde işlem zamanı dün, kayıt zamanı bugün olarak korunur.
- [ ] **AC45 — Rapor süresi, Q06b:** Hazır 10.000 kayıt için rapor/filtre işlemi tanımlı referans ortamda 10 tekrarın her birinde en fazla 2 saniyede görünür.
- [ ] **AC46 — Değerlendirme süresi, Q06b:** Hazır 20 varlık × 1.000 günlük mum × 3 aday değerlendirmesi tanımlı ortamda 10 tekrarın her birinde en fazla 120 saniyede tamamlanır.
- [ ] **AC47 — Al-tut, Q01:** Sıfır maliyet, 1.000 başlangıç sermayesi, ilk uygun açılış 100, miktar adımı 1 ve dönem sonu fiyatı 110 olan al-tut örneğinde son sermaye 1.100 olur.
- [ ] **AC48 — Deneme tanımı, Q02/Q05:** Giriş veya çıkış kuralı henüz onaylanmamış aday için sonuçlar onaylı strateji değerlendirmesi olarak yayımlanmaz.

### Ek kriterler — Revizyon 1 (QA denetimi sonrası)
> Parantez içindeki B/S/H/K kodları Revizyon 1 QA bulgu raporuna atıftır (izlenebilirlik); her kriter yine tek başına test edilir.

- [ ] **AC49 — Sıfır taban, S01:** Başlangıç sermayesi 0 veya negatif olan dönemde net getiri yüzdesi üretilmez, “hesaplanamıyor” görünür.
- [ ] **AC50 — Ölçüm frekansı, S02:** Gün içi 8.000'e inip 11.000 kapanan günün bulunduğu dizide maksimum düşüş yalnız günlük kapanış değerlerinden hesaplanır.
- [ ] **AC51 — Düşüş eşitliği, Q03:** Adayın maksimum düşüşü referansa tam eşitse bu ölçüt nedeniyle engellenmez.
- [ ] **AC52 — Sıfır beklenti, Q03:** İşlem beklentisi tam 0 olan aday pozitif beklenti koşulunu sağlamaz.
- [ ] **AC53 — Bölünmeyen ayrım, Q01:** 7 uygun tarihte ayar seçimine 4, ayrı değerlendirmeye 3 tarih atanır.
- [ ] **AC54 — Yetersiz geçmiş eşiği, Q01:** 249 uygun tarih içeren dönem “yetersiz geçmiş” olarak işaretlenir ve yeterli kanıt sayılmaz.
- [ ] **AC55 — Tek günlük dönem, S05:** Başlangıcı bitişine eşit dönem reddedilmez; sonuç o günün verisiyle üretilir.
- [ ] **AC56 — Ortak tarih yok, S05:** Karşılaştırılan iki varlığın ortak işlem tarihi yoksa karşılaştırma yapılmaz ve nedeni gösterilir.
- [ ] **AC57 — Veri öncesi dönem, S05:** Tümü varlığın ilk verisinden önce olan dönem için metrik üretilmez, “yetersiz geçmiş” görünür.
- [ ] **AC58 — Ölçek altı oran, Q04:** Yüzde ölçeğinde sıfıra yuvarlanan risk oranı girdisi (ör. %0,001) reddedilir.
- [ ] **AC59 — Geçerlilik anı, S07:** Fiyat teması stop yükseltmesinin geçerlilik anıyla tam aynı zamanda gerçekleştiğinde temas yeni stopa dahil sayılır.
- [ ] **AC60 — Küsuratlı kademe, S08:** Miktar adımı 0,0001 olan varlıkta kısmi satış sonrası kalan miktar ürün adımına uygun tek değer olarak gösterilir.
- [ ] **AC61 — Son parça, S08:** Kademeli çıkışta son parça da kapandığında kapanmış işlem sayısı tam 1 artar.
- [ ] **AC62 — Nakit kalıntısı, Q01:** Sıfır maliyet, 1.000 sermaye, ilk açılış 300 ve miktar adımı 1 olan al-tut örneğinde 3 birim alınır ve 100 nakit dönem sonu sermayesine dahil edilir.
- [ ] **AC63 — Kesintili gün, S10:** İzlemenin çalışmadığı gün izlenen gün sayacına katılmaz.
- [ ] **AC64 — Bilinmeyen filtre değeri, H01:** Tanımlı olmayan piyasa/kategori değeri ile rapor istendiğinde istek doğrulama hatasıyla (400) reddedilir ve teknik hata metni sızmaz.
- [ ] **AC65 — Veri bulunmayan filtre, H01:** Tanımlı ama o dönemde verisi olmayan filtre için hata değil boş sonuç ve örnek sayısı 0 gösterilir.
- [ ] **AC66 — Olmayan kayıt, H01:** Var olmayan bir değerlendirme kaydı istendiğinde bulunamadı (404) sonucu döner.
- [ ] **AC67 — Ters stop, H02:** Girişten yüksek stop ile risk bazlı miktar istenirse hesap reddedilir ve nedeni gösterilir.
- [ ] **AC68 — Sıfır/negatif stop, H02:** Stop 0 veya negatif verildiğinde hesap reddedilir ve nedeni gösterilir.
- [ ] **AC69 — Ön kayıtsız aday, H03:** Ölçütleri ve değerlendirme dönemi önceden kaydedilmemiş aday için koşum başlatılmaz.
- [ ] **AC70 — Eşzamanlı işleme, H04:** Aynı varlık, sürüm ve mum zamanı için iki koşum eşzamanlı çalıştığında tek sanal işlem kaydı oluşur.
- [ ] **AC71 — Mum revizyonu, H05:** Karara dayanak mum sağlayıcı tarafından sonradan revize edildiğinde karar değişmez ve revizyon ayrı kayıtla işaretlenir.
- [ ] **AC72 — Sınır sıfırlama, H06:** Kayıp sınırı dolmuş kullanıcıda sınır yalnız açık kullanıcı eylemiyle sıfırlanır; eylem kaydı ve yeni dönem başlangıcı görünür.
- [ ] **AC73 — Karma para birimi riski, H07:** Açık pozisyonların riski TL ve USD karışık olduğunda toplam risk doğrulanmış gösterilmez ve nedeni açıklanır.
- [ ] **AC74 — Karma para birimi kaybı, H07:** Dönem kaybı farklı para birimlerinden oluştuğunda kayıp sınırı doğrulanmış hesaplanmaz ve nedeni açıklanır.
- [ ] **AC75 — Risksiz pozisyon, H08:** Stopu girişin üstüne taşınmış pozisyonun riski sıfır sayılır; toplam risk bütçesini artırmaz.
- [ ] **AC76 — Geç kayıt işareti, Q08:** Kapanmış bir döneme geriye dönük gerçek işlem kaydedildiğinde o dönem raporu “geç kayıtla güncellendi” işaretiyle görünür ve ileri dönem yeterlilik sayacı değişmez.
- [ ] **AC77 — Sürüm devri, Q08:** Aktif strateji sürümü değiştiğinde açık gerçek pozisyon giriş anındaki sürümün kurallarıyla yönetilmeye devam eder.
- [ ] **AC78 — Yok değeri, H12:** Ölçülemeyen metrik 0 değil yok değeri taşır; boş dönemde getiri ve düşüş de “hesaplanamıyor” görünür.
- [ ] **AC79 — Parmak izi, H13:** Aday ayarlarından biri değiştiğinde strateji sürümü değişmiş sayılır ve önceki sürümün gözlem sayacı devralınmaz.
- [ ] **AC80 — Ayrı giriş adayları, R06:** Kırılım ve yükseliş içi geri çekilme adayları ayrı kayıt ve ayrı sonuçla listelenir.
- [ ] **AC81 — Sınır ayarının kalıcılığı, R10:** Kullanıcının belirlediği işlem başına ve toplam risk sınırı kaydedilir ve sonraki miktar hesabında kullanılır.
- [ ] **AC82 — Maliyetli miktar, R10:** Risk bütçesi 100, giriş 100, stop 90, birim başına bilinen maliyet 1 ve miktar adımı 1 iken önerilen miktar 9 olur.
- [ ] **AC83 — Adım altı miktar, R10:** Hesaplanan miktar ürün adımının veya asgari işlem tutarının altında kalırsa miktar üretilmez ve nedeni gösterilir.
- [ ] **AC84 — Üst sınır etiketi, R03:** Maliyeti bilinmeyen işlem içeren raporda metrik “üst sınır” olarak etiketlenir ve eksik maliyetli işlem sayısı görünür.
- [ ] **AC85 — Rejim sınıflandırıcı sürümü, Q02:** Her piyasa koşulu gözlemi, kendisini üreten rejim sınıflandırıcısının sürümüyle birlikte görünür.
- [ ] **AC86 — Sağlayıcı bekleyişi, Q06b:** Performans raporunda sağlayıcı bekleme süresi toplam süreden ayrı gösterilir.
- [ ] **AC87 — Hızlandırılmış test, R18:** Saat/tarih ileri alınarak üretilen gözlem ileri dönem yeterlilik sayacına katılmaz.

## Definition of Done
- [ ] Q01–Q08 kararları ve aday kuralları kesinleştirildi; bağımlı kriterler güncellendi ve spec yönetici tarafından onaylandı.
- [ ] Her aşamanın kabulü ayrı kanıtlandı; tüm kabul kriterleri bağımsız test/kanıtla karşılandı.
- [ ] İş davranışı testleri, Q07 kararı ve [testing.md](../docs/testing.md) ile uyumlu çalıştırıldı; test çatısı istisnası spec'e değil `docs/testing.md`'ye yazıldı ve `AGENTS.md`'deki yığın cümlesi gerçek kodla uyumlu hale getirildi.
- [ ] Arayüz kriterlerinin ekran görüntüleri ve rapor → aday karşılaştırma → risk engeli → ileri takip kritik akışının smoke kanıtı üretildi.
- [ ] Geçerli lint, derleme/açılış ve test kontrolleri yeşil; para/yüzde kuralları sağlandı, teknik hatalar kullanıcıya sızmadı.
- [ ] Performans ölçümleri Constraints'teki referans ölçüm ortamında yapıldı; ortam, veri büyüklüğü, ilk/sonraki çalışma süreleri ve sağlayıcı bekleyişi ayrı raporlandı.
- [ ] İleri takip özelliğinin çalışması ile seçilen stratejinin yeterli gerçek ileri dönem kanıtı toplaması ayrı raporlandı; hızlandırılmış test 90 günlük gerçek kanıt sayılmadı.
- [ ] Ayrı QA oturumu doğrulaması tamamlandı; yönetici sonucu değerlendirdi.
- [ ] PR ve squash-merge [git.md](../docs/git.md) kurallarıyla tamamlandı.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 — bağımsız QA denetimi sonrası boşluk kapatma |
| Düzeltme turu sayısı | Uygulama başlamadı |
| Bulgu gerçek/gürültü oranı | Revizyon 1: 35 bulgu getirildi, 35'i gerçek kabul edildi, 0 gürültü (spec denetimi; kod denetimi yapılmadı) |
| Regresyon sayısı | Ölçülmedi |
| Kaçan hata | Ölçülmedi |
