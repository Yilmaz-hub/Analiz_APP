# Spec: 0006 — Kısmi Satış Teyidi ve Pozisyon Fiyatının Güvenilirliği

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist — INTENT · CLARIFY · SPEC. Revizyon 1 (2026-10-04).
> Durum: **ONAYLI (Revizyon 3, 2026-10-04).** Q01–Q08 Takım Yöneticisi tarafından önerildiği gibi onaylandı; uygulama TDD ile başlayabilir.
> Kaynak: kullanıcının canlı uygulamada bildirdiği iki sorun (2026-10-04). Revizyon 2 (2026-10-04): B bölümü, kullanıcının "fiyat alınamadı demek çözüm değil, ekranın amacı fiyatı göstermek" itirazı ve `LINKUSD`/`HBARUSD` örnekleri üzerine kök nedenle yeniden yazıldı.

## Intent
Kullanıcı elindeki pozisyonun bir kısmını (ör. %40'ını ya da istediği bir miktarı) satıp kalanını tutabilmek istiyor. Bugün "Kar Al / Satış Yap" ekranı miktar kutusu gösteriyor ama yalnızca pozisyonun tamamını kabul ediyor; bu, kâr almak isteyen kullanıcıyı ya hepsini satmaya ya da hiç satmamaya zorluyor.
Kullanıcı ayrıca uygulamayı açtığında "Pozisyonlar" listesinde, kendi sonradan eklediği varlıkların (örnek: `LINKUSD`, `HBARUSD`) kârının 0 göründüğünü, grafiği açınca gerçek değere döndüğünü bildiriyor. Bu ekranın amacı güncel fiyatı ve kârı göstermektir; fiyat alınamadığını bildirmek çözüm değildir. Sorun, sembolün uygulamada nasıl yorumlandığı ve fiyatın grafikten farklı bir yoldan alınmasıdır.
Başarı: kullanıcı pozisyonunun herhangi bir bölümünü kayda geçirebilir, kalan pozisyon doğru görünür ve hesaplanır; grafiği açılabilen her varlığın pozisyon satırı, grafik açılmadan, güncel fiyat ve gerçek kârla görünür.

## Requirements
### A — Kısmi satış teyidi
- **R01:** Kullanıcı, elindeki miktarın sıfırdan büyük ve eldeki miktara eşit ya da daha küçük herhangi bir bölümünü satış olarak teyit edebilmelidir. Tam miktar, bugünkü tam çıkış davranışını aynen korur.
- **R02:** Kısmi satıştan sonra pozisyon **aktif** kalır. Kalan miktar = eski miktar − satılan miktar; ürünün miktar adımına uygun tek değerdir (kayan nokta artığı yoktur). Giriş fiyatı ve kayıtlı stop değişmez.
- **R03:** Gerçekleşen kâr/zarar satılan miktarın giriş maliyetine göre, nakit satış tutarı kadar artarak hesaplanır; yatırım tutarı kalan miktarın maliyetine iner. Tüm tutarlar `Decimal`dir.
- **R04:** Kısmi satışlar tek pozisyonun parçalarıdır: kapanmış işlem sayısı ve performans raporu, pozisyon ancak **son parça** kapandığında tam 1 kapanmış işlem sayar ([0005](0005-performans-strateji-risk-ve-sanal-dogrulama.md) AC61 ile tutarlı).
- **R05:** Satış girişi iki yoldan yapılabilir: miktar ya da yüzde. Yüzde, eldeki miktara uygulanıp ürün miktar adımına **aşağı** yuvarlanır; yuvarlanınca sıfıra düşen yüzde reddedilir ve nedeni gösterilir. Hızlı seçim düğmeleri sunulur (Q02).
- **R06:** Her kısmi satış ayrı, kimlikli bir olaydır; aynı satış tekrar teyit edilirse ikinci kayıt oluşmaz. Günlük (`position_journal`) kalan pozisyonu doğru yeniden kurar; işlem zamanı ile kayıt zamanı ayrımı korunur.
- **R07:** Eldeki miktardan fazla, sıfır/negatif miktar, gelecek zaman ve girişten önceki zaman nedeniyle reddedilen satış, nedeniyle gösterilir; kayıt değişmez.
- **R08:** V1 SAT uyarısı ve stop çıkışı "tam çıkış" yönlendirmesi olarak kalır; uygulama kısmi satışı kendiliğinden yapmaz ya da önermez.

### B — Pozisyon fiyatının güvenilirliği
- **R09:** Pozisyonlar listesindeki güncel fiyat, o varlığın grafiğini gösteren kaynaklarla ve aynı sembol yorumuyla alınmalıdır. Grafiği açılabilen her varlık için liste, grafik açılması gerekmeden, güncel fiyat ve gerçek kârı göstermelidir; sonradan eklenen varlık ile varsayılan listedeki varlık arasında fark olmamalıdır.
- **R10:** Kullanıcı kripto sembolünü tire olmadan yazdığında (`LINKUSD`, `HBARUSD`) uygulama onu varsayılan listedeki biçimle (`LINK-USD`) aynı varlık türü olarak tanımalıdır: aynı fiyat kaynakları, aynı günlük kapanış kuralı ([0003](done/0003-islem-kararlari-ve-performans-tutarliligi.md) kripto politikası) ve [0005](0005-performans-strateji-risk-ve-sanal-dogrulama.md) raporlarında, aday değerlendirmesinde ve ileri takipte aynı kripto piyasası ve para birimi. Kayıtlı sembol sessizce değiştirilmez; yorum okuma anında yapılır (spec 0004: sessiz dönüşüm yok).
- **R11:** Bir fiyat kaynağı yanıt vermezse sıradaki kaynağa geçilir; kullanıcı bunu fark etmez, fiyat yine görünür.
- **R12 (savunma, çözüm değil):** Beklenmedik biçimde hiçbir kaynaktan fiyat alınamazsa kâr/zarar sessizce 0 yapılmaz; satır bunu belirtir ve toplam değer de belirtir.

## Constraints
- Para ve yüzde `Decimal` ([conventions.md](../docs/conventions.md)); sunum yuvarlaması hesap girdisi olamaz.
- Mevcut kayıtlar (tek çıkış alanlı `JournalExitEventId`, `Çıkış Adedi`, `Gerçekleşen Çıkış`) okunmaya devam eder; veri kaybı ve sessiz dönüşüm yoktur (spec 0004).
- Karar mantığı (V1 AL/BEKLE/SAT) değişmez; bu spec yalnız gerçekleşen işlemin **kaydını** ve fiyat gösterimini kapsar.
- **Kapsam dışı:** ek alım (pozisyona ekleme), otomatik satış, gerçek emir gönderimi, kaldıraç/açığa satış, kâr hedefi/iz süren stop adaylarının gerçek pozisyona uygulanması ([0005](0005-performans-strateji-risk-ve-sanal-dogrulama.md) R13 gereği bunlar yalnız sanal kalır).

## Context
- İlgili modüller: `trade_confirmation.py` (`validate_sell`, `confirm_sell`, `reconcile`), `position_journal.py` (`_rebuild_positions`), `positions.py` (`build_active_rows`), `data_fetchers.py` (`get_live_price_for_portfolio`), `app.py` ("Kar Al / Satış Yap").
- **Mevcut durum (kod okumasıyla doğrulandı):** `validate_sell` miktar pozisyona eşit değilse `MIKTAR_POZISYONLA_ESIT_DEGIL` döner (spec 0003 R03: "SAT tam çıkış"; 0003 kapsam dışı: "kısmi satış"). `confirm_sell` pozisyonu `Adet = 0` ve `CLOSED_CONFIRMED` yapar. `PositionJournal._rebuild_positions` her SATIŞ'ı pozisyonu tamamen kapatan olay sayar. `reconcile` tek çıkış kimliği varsayar. Yani kısmi satış yalnız bir kutuyu açmak değil, bu üç yerin birlikte değişmesidir.
- **Fiyat sorununun kök nedeni (kodla ve taklit edilmiş ağ yanıtlarıyla yeniden üretildi):** (1) `get_live_price_for_portfolio` Binance'e yalnız `api.binance.com` adresinden sorar; grafik ise `data-api.binance.vision`, `api.binance.us` ve `api.binance.com`'u sırayla dener. İlk adres ABD çıkışlı sunucularda engelli olabilir (HTTP 451); grafik çalışır, canlı fiyat çalışmaz. (2) Binance'ten yanıt gelmeyince Yahoo'ya kullanıcının yazdığı sembolle sorulur; Yahoo kriptoyu yalnız `LINK-USD` biçiminde tanır, `LINKUSD` için hata verir ve fiyat 0 döner. Tireli sembol (`LINK-USD`) Yahoo yedeğiyle fiyat bulduğu için varsayılan listedeki varlıklar sorunsuz görünür. (3) `build_active_rows` fiyat 0 iken giriş fiyatını kullanır; kâr tam 0 görünür, uyarı yoktur.
- **Aynı kökten ikinci etki:** tire olmadan yazılan kripto sembolleri uygulamanın başka yerlerinde de kripto sayılmıyor. 0005'in `market_map.market_of("LINKUSD")` boş döner (rapor "piyasası tanınmıyor" der, aday değerlendirmesi ve ileri takip koşucusu varlığı atlar); [0003](done/0003-islem-kararlari-ve-performans-tutarliligi.md)'ün `policy_for_symbol("LINKUSD")` bunu ABD hissesi politikasına (16:00 New York kapanışı) bağlar. R10 bunu da kapsar.
- **Doğrulanamayan kısım:** Streamlit Cloud'da `api.binance.com`'un gerçekten engelli olup olmadığı bu ortamdan görülemiyor (burada Yahoo ve Binance erişimi sınırlı). Çözüm bu varsayıma bağlı değildir: R09–R11 her kaynak sırasında ve her sembol biçiminde geçerlidir.
- Terimler: **kısmi satış** — pozisyon miktarının bir bölümünün satışı, pozisyon aktif kalır; **gecikmeli fiyat** — canlı fiyat yerine kullanılan son günlük kapanış; **tam çıkış** — kalan miktarın tamamının satışı.
- Bağımlılıklar: [0003](done/0003-islem-kararlari-ve-performans-tutarliligi.md) (satış akışı), [0004](done/0004-varlik-ve-pozisyon-kayitlarini-koruma.md) (kalan miktar durumun önündedir, G01).

### CLARIFY — kararlar (hepsi 2026-10-04'te önerildiği gibi onaylandı)
| Karar | Soru | Öneri | Gerekçe |
|---|---|---|---|
| Q01 | Kısmi satış serbest mi? | Evet: 0 < miktar ≤ eldeki miktar. Tam miktar bugünkü tam çıkıştır. | Kullanıcının isteği; "ya hepsi ya hiç" kâr almayı engelliyor. Sınır dahil: tam miktar geçerlidir. |
| Q02 | Yüzde girişi ve hızlı seçimler? | Miktar kutusuna ek olarak yüzde kutusu; hazır düğmeler %25, %50, %75, %100. Serbest yüzde (ör. %40) ve serbest miktar her zaman girilebilir. | Her değeri düğmeyle sunmak ekranı kalabalıklaştırır; serbest giriş "farklı bir değer" isteğini karşılar. |
| Q03 | Kalan pozisyonun stopu? | Değişmez; yalnız kullanıcının teyitli yükseltmesiyle değişir. | Spec 0003 ilkesi: stop kullanıcı teyidiyle değişir. Satış stopu kendiliğinden oynatırsa korumayı sessizce değiştirir. |
| Q04 | Kâr/zarar hangi maliyete göre? | Pozisyonun giriş fiyatına göre, satılan miktar oranında. Ek alım kapsam dışı olduğundan tek giriş fiyatı vardır. | Basit ve denetlenebilir; ortalama maliyet ek alım gelirse ayrı spec ister. |
| Q05 | Kapanmış işlem sayısı? | Pozisyon son parçada kapanınca tam 1. | 0005 AC61 ve 0003 "kademeli satışlar tek pozisyonun parçalarıdır" terimiyle tutarlı; aksi halde performans raporu şişer. |
| Q06 | Eski kayıtlar ve veri biçimi? | Yeni `Çıkışlar` listesi eklenir (her satış: kimlik, miktar, fiyat, zaman); tam kapanışta eski alanlar da doldurulmaya devam eder, eski kayıtlar olduğu gibi okunur. | Geri uyumlu; mevcut portföy ve günlük kayıtları bozulmaz (spec 0004). Tek çıkış alanı kısmi satışı taşıyamaz. |
| Q07 | Pozisyon fiyatı hangi kaynaktan ve hangi sembol yorumuyla alınsın? | Grafikle aynı kaynak sırası (önce `data-api.binance.vision`, sonra `api.binance.us`, sonra `api.binance.com`, ardından Yahoo). Tire olmadan yazılan kripto sembolleri (`LINKUSD`, `HBARUSD`) okuma anında tireli biçime (`LINK-USD`) çevrilerek yorumlanır; her kaynak kendi biçimini alır. Depodaki kayıt değişmez. | Kök neden iki parçalı: dar kaynak listesi ve Yahoo'nun tireli sembol beklemesi. İkisi birlikte çözülmezse biri düzelirken öteki 0 göstermeyi sürdürür. Grafik zaten çalışan yolu kullandığı için yeni bir bağımlılık gerekmez. |
| Q08 | Sembol yorumu nerede uygulansın, depodaki kayıt dönüştürülsün mü? | Dönüştürülmez; yorum okuma anında, uygulamanın her yerinde tek bir kuralla yapılır (fiyat, piyasa/para birimi, günlük kapanış kuralı). Varlık eklenirken, sembolün nasıl yorumlanacağı kullanıcıya gösterilir (ör. `LINKUSD → LINK-USD, kripto`). | Sessiz veri dönüşümü yasaktır (spec 0004). Kullanıcının canlı Neon verisine dokunmayan tek yol budur; ayrıca aynı hata başka yerlerde de çözülür. Alternatif: ekleme anında canonical biçime çevirmek — eski kayıtları düzeltmez. |

**Önceki taslaktaki Q07 ("fiyat alınamayınca son kapanışı etiketle") kullanıcı tarafından reddedildi:** ekranın amacı fiyatı göstermek; etiketlemek kök nedeni çözmez. Savunma davranışı yalnız R12 olarak kaldı.

## Acceptance Criteria
> Her kriter tek başına test edilir; her kriter bir pytest testi, docstring ilk satırı `ACnn`. Q etiketli kriter önerilen karara bağlıdır.
### A — Kısmi satış
- [ ] **AC01 — Kısmi miktar, Q01:** 10 adetlik pozisyonun 4 adedi satış olarak teyit edildiğinde satış kabul edilir.
- [ ] **AC02 — Kalan miktar, R02:** Aynı satıştan sonra pozisyon aktif kalır ve kalan miktar tam 6'dır.
- [ ] **AC03 — Küsuratlı kalan, R02:** Miktar adımı 0,0001 olan 0,3333 adetlik pozisyonda 0,1666 satıldığında kalan tam 0,1667'dir (kayan nokta artığı yok).
- [ ] **AC04 — Tam miktar, Q01:** Eldeki miktarın tamamı satıldığında pozisyon bugünkü gibi kapanır ve kalan miktar 0'dır.
- [ ] **AC05 — Fazla miktar, R07:** Eldeki miktardan büyük satış reddedilir, nedeni gösterilir ve kayıt değişmez.
- [ ] **AC06 — Geçersiz miktar, R07:** 0 veya negatif miktarlı satış reddedilir ve kayıt değişmez.
- [ ] **AC07 — Stop değişmez, Q03:** Kısmi satıştan sonra kayıtlı stop aynen kalır.
- [ ] **AC08 — Kâr hesabı, Q04:** Giriş 100, satış 110, 4 adet satıldığında gerçekleşen kâr 40; nakit 440 artar; yatırım tutarı kalan 6 adedin maliyeti olan 600'e iner.
- [ ] **AC09 — Kapanmış işlem sayısı, Q05:** İki parça halinde satılan pozisyon, ilk parçadan sonra kapanmış işlem sayısını artırmaz; son parçadan sonra tam 1 artırır.
- [ ] **AC10 — Yüzde girişi, R05:** %40 girildiğinde 10 adetlik pozisyon için satılacak miktar 4 olur.
- [ ] **AC11 — Yüzde yuvarlama, R05:** Miktar adımı 1 olan 10 adetlik pozisyonda %45 girildiğinde satılacak miktar 4'tür (aşağı yuvarlanır, yukarı yuvarlama yok).
- [ ] **AC12 — Sıfıra düşen yüzde, R05:** Miktar adımı 1 olan 10 adetlik pozisyonda %5 girildiğinde (0,5 adet → 0) satış reddedilir ve nedeni gösterilir.
- [ ] **AC13 — Tekrar teyit, R06:** Aynı kısmi satış ikinci kez teyit edildiğinde ikinci kayıt oluşmaz, kalan miktar değişmez.
- [ ] **AC14 — Günlük, R06:** Kısmi satıştan sonra günlük, pozisyonu kalan miktarla yeniden kurar (pozisyon silinmez).
- [ ] **AC15 — Zaman ayrımı, R06:** Dünkü kısmi satış bugün kaydedildiğinde işlem zamanı dün, kayıt zamanı bugün olarak korunur.
- [ ] **AC16 — Eski kayıt, Q06:** Tek çıkış alanlı eski bir kapanmış kayıt hatasız okunur ve aynı sonucu gösterir.
- [ ] **AC17 — Ekran, R05:** "Kar Al / Satış Yap" ekranında yüzde kutusu ve hazır düğmeler görünür; %40 girilip onaylanınca pozisyon kalan miktarla listede kalır.
- [ ] **AC18 — SAT uyarısı, R08:** SAT uyarısı verildiğinde ekran "tamamını sat" yönlendirmesini korur; kısmi satış kendiliğinden yapılmaz.
### B — Fiyat güvenilirliği
- [ ] **AC19 — Tiresiz sembol, Q07:** İlk Binance adresi yanıt vermediğinde ve ikincisi fiyat döndürdüğünde `LINKUSD` sembollü varlığın güncel fiyatı bulunur (0 değil).
- [ ] **AC20 — İkinci örnek, Q07:** Aynı koşulda `HBARUSD` sembollü varlığın güncel fiyatı bulunur.
- [ ] **AC21 — Kaynak sırası, R11:** İlk kaynak hata verdiğinde fiyat sıradaki kaynaktan alınır ve kullanıcıya hata görünmez.
- [ ] **AC22 — Yahoo biçimi, Q07:** Binance kaynaklarının hiçbiri yanıt vermediğinde `LINKUSD` için Yahoo'ya `LINK-USD` biçimiyle sorulur ve fiyat bulunur.
- [ ] **AC23 — Tireli sembol, R09:** `LINK-USD` sembollü varlığın fiyatı bugünkü gibi bulunur (gerileme yok).
- [ ] **AC24 — Kripto tanıma, R10:** `LINKUSD` ve `HBARUSD` kripto piyasası ve USD para birimi olarak tanınır; günlük kapanış kuralı tireli kripto ile aynıdır (00:00 UTC), ABD hissesi kuralı uygulanmaz.
- [ ] **AC25 — Hisse ayrımı, R10:** `AAPL`, `THYAO.IS`, `EURUSD=X`, `XAU_GOLD` ve `GRAM_TRY` yorumları değişmez (yanlış kripto tanıma yok).
- [ ] **AC26 — Kayıt değişmez, Q08:** Depoda `LINKUSD` olarak kayıtlı sembol, fiyat ve rapor akışlarından sonra da `LINKUSD` olarak kalır.
- [ ] **AC27 — Ekran, R09:** Sonradan eklenmiş `LINKUSD` varlığının pozisyon satırı, grafik açılmadan, kaynaktan gelen güncel fiyat ve ona göre kâr/zararla görünür; kâr 0 görünmez.
- [ ] **AC28 — Yorum gösterimi, Q08:** Varlık eklerken `LINKUSD` için "LINK-USD, kripto olarak okunacak" bilgisi görünür.
- [ ] **AC29 — 0005 uyumu, R10:** `LINKUSD` sembollü varlık için performans raporu, aday değerlendirmesi ve ileri takip varlığı atlamaz.
- [ ] **AC30 — Savunma, R12:** Tüm kaynaklar yanıt vermezse satırda kâr/zarar 0 yazmaz, nedenini belirtir ve toplam değer bunu belirtir.
- [ ] **AC31 — Canlı fiyat varken:** Fiyat bulunduğunda satır etiket taşımaz ve bugünkü gibi hesaplanır.

## Definition of Done
- [ ] Q01–Q08 kararları kesinleşti; kriterler güncellendi; spec Takım Yöneticisi tarafından onaylandı.
- [ ] Tüm kabul kriterleri bağımsız pytest testiyle karşılandı; arayüz kriterlerinin ekran görüntüleri eklendi.
- [ ] Geçerli lint ve test kontrolleri yeşil; para/yüzde `Decimal`; teknik hata kullanıcıya sızmadı.
- [ ] Kök neden yeniden üretimi (taklit ağ yanıtlarıyla) testlere dönüştü; canlıda `LINKUSD`/`HBARUSD` ile doğrulandı ve sonucu kaydedildi (AP-05).
- [ ] Ayrı QA oturumu doğrulaması tamamlandı.
- [ ] PR ve squash-merge [git.md](../docs/git.md) kurallarıyla tamamlandı.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 3 — Rev 3: Q01–Q08 onaylandı; onaydan önce Rev 2'de B bölümü kullanıcı itirazıyla kök nedenle yeniden yazıldı |
| Düzeltme turu sayısı | Uygulama başlamadı |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | Ölçülmedi |
| Kaçan hata | Ölçülmedi |
