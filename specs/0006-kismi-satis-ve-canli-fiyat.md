# Spec: 0006 — Kısmi Satış Teyidi ve Pozisyon Fiyatının Güvenilirliği

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist — INTENT · CLARIFY · SPEC. Revizyon 1 (2026-10-04).
> Durum: **TASLAK — karar bekliyor.** Q01–Q07 onaylanmadan kod yazılmaz (AGENTS.md Altın Kural 1 ve 4).
> Kaynak: kullanıcının canlı uygulamada bildirdiği iki sorun (2026-10-04).

## Intent
Kullanıcı elindeki pozisyonun bir kısmını (ör. %40'ını ya da istediği bir miktarı) satıp kalanını tutabilmek istiyor. Bugün "Kar Al / Satış Yap" ekranı miktar kutusu gösteriyor ama yalnızca pozisyonun tamamını kabul ediyor; bu, kâr almak isteyen kullanıcıyı ya hepsini satmaya ya da hiç satmamaya zorluyor.
Kullanıcı ayrıca uygulamayı açtığında "Pozisyonlar" listesinde, kendi sonradan eklediği varlıkların kârının 0 göründüğünü, grafiği açınca gerçek değere döndüğünü bildiriyor. Gerçek kâr bilinmezken 0 göstermek kullanıcıyı yanıltır; bilinmeyen değer sıfır diye sunulmaz.
Başarı: kullanıcı pozisyonunun herhangi bir bölümünü kayda geçirebilir, kalan pozisyon doğru görünür ve hesaplanır; fiyatı alınamayan varlık hiçbir zaman "kâr 0" gibi görünmez.

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
- **R09:** Aktif pozisyon listesinde canlı fiyatı alınamayan varlık için kâr/zarar sessizce 0 yapılmaz. Sırasıyla: (1) son bilinen günlük kapanış fiyatı kullanılır ve satır "gecikmeli fiyat (tarih)" etiketi taşır; (2) hiçbir fiyat yoksa satırda "fiyat alınamadı" ve kâr/zarar "hesaplanamıyor" yazar.
- **R10:** Bu davranış varsayılan listedeki ve kullanıcının sonradan eklediği varlıklar için aynıdır; grafiğin açılması gerekmez.
- **R11:** Toplam pozisyon değeri, fiyatı alınamayan ya da gecikmeli fiyatla hesaplanan pozisyonları ayrıca belirtir.

## Constraints
- Para ve yüzde `Decimal` ([conventions.md](../docs/conventions.md)); sunum yuvarlaması hesap girdisi olamaz.
- Mevcut kayıtlar (tek çıkış alanlı `JournalExitEventId`, `Çıkış Adedi`, `Gerçekleşen Çıkış`) okunmaya devam eder; veri kaybı ve sessiz dönüşüm yoktur (spec 0004).
- Karar mantığı (V1 AL/BEKLE/SAT) değişmez; bu spec yalnız gerçekleşen işlemin **kaydını** ve fiyat gösterimini kapsar.
- **Kapsam dışı:** ek alım (pozisyona ekleme), otomatik satış, gerçek emir gönderimi, kaldıraç/açığa satış, kâr hedefi/iz süren stop adaylarının gerçek pozisyona uygulanması ([0005](0005-performans-strateji-risk-ve-sanal-dogrulama.md) R13 gereği bunlar yalnız sanal kalır).

## Context
- İlgili modüller: `trade_confirmation.py` (`validate_sell`, `confirm_sell`, `reconcile`), `position_journal.py` (`_rebuild_positions`), `positions.py` (`build_active_rows`), `data_fetchers.py` (`get_live_price_for_portfolio`), `app.py` ("Kar Al / Satış Yap").
- **Mevcut durum (kod okumasıyla doğrulandı):** `validate_sell` miktar pozisyona eşit değilse `MIKTAR_POZISYONLA_ESIT_DEGIL` döner (spec 0003 R03: "SAT tam çıkış"; 0003 kapsam dışı: "kısmi satış"). `confirm_sell` pozisyonu `Adet = 0` ve `CLOSED_CONFIRMED` yapar. `PositionJournal._rebuild_positions` her SATIŞ'ı pozisyonu tamamen kapatan olay sayar. `reconcile` tek çıkış kimliği varsayar. Yani kısmi satış yalnız bir kutuyu açmak değil, bu üç yerin birlikte değişmesidir.
- **Fiyat sorununun mekanizması (doğrulandı):** `build_active_rows` canlı fiyat 0 dönünce giriş fiyatını kullanır; değer = yatırım olur ve kâr tam 0 görünür, hiçbir uyarı yoktur. Aynı pozisyonun grafiği açıkken fiyat grafikteki son kapanıştan geldiği için doğru görünür.
- **Doğrulanamayan kısım:** canlı fiyatın neden 0 döndüğü. Bu geliştirme ortamından Yahoo'ya erişim engelli (HTTP 403) olduğu için yeniden üretilemedi. Kullanıcıdan etkilenen varlığın sembolü istenmiştir (açık bilgi, aşağıda). Çözüm kök nedene bağlı değildir: R09 her nedenle fiyat alınamadığında geçerlidir.
- Terimler: **kısmi satış** — pozisyon miktarının bir bölümünün satışı, pozisyon aktif kalır; **gecikmeli fiyat** — canlı fiyat yerine kullanılan son günlük kapanış; **tam çıkış** — kalan miktarın tamamının satışı.
- Bağımlılıklar: [0003](done/0003-islem-kararlari-ve-performans-tutarliligi.md) (satış akışı), [0004](done/0004-varlik-ve-pozisyon-kayitlarini-koruma.md) (kalan miktar durumun önündedir, G01).

### CLARIFY — onay bekleyen kararlar
| Karar | Soru | Öneri | Gerekçe |
|---|---|---|---|
| Q01 | Kısmi satış serbest mi? | Evet: 0 < miktar ≤ eldeki miktar. Tam miktar bugünkü tam çıkıştır. | Kullanıcının isteği; "ya hepsi ya hiç" kâr almayı engelliyor. Sınır dahil: tam miktar geçerlidir. |
| Q02 | Yüzde girişi ve hızlı seçimler? | Miktar kutusuna ek olarak yüzde kutusu; hazır düğmeler %25, %50, %75, %100. Serbest yüzde (ör. %40) ve serbest miktar her zaman girilebilir. | Her değeri düğmeyle sunmak ekranı kalabalıklaştırır; serbest giriş "farklı bir değer" isteğini karşılar. |
| Q03 | Kalan pozisyonun stopu? | Değişmez; yalnız kullanıcının teyitli yükseltmesiyle değişir. | Spec 0003 ilkesi: stop kullanıcı teyidiyle değişir. Satış stopu kendiliğinden oynatırsa korumayı sessizce değiştirir. |
| Q04 | Kâr/zarar hangi maliyete göre? | Pozisyonun giriş fiyatına göre, satılan miktar oranında. Ek alım kapsam dışı olduğundan tek giriş fiyatı vardır. | Basit ve denetlenebilir; ortalama maliyet ek alım gelirse ayrı spec ister. |
| Q05 | Kapanmış işlem sayısı? | Pozisyon son parçada kapanınca tam 1. | 0005 AC61 ve 0003 "kademeli satışlar tek pozisyonun parçalarıdır" terimiyle tutarlı; aksi halde performans raporu şişer. |
| Q06 | Eski kayıtlar ve veri biçimi? | Yeni `Çıkışlar` listesi eklenir (her satış: kimlik, miktar, fiyat, zaman); tam kapanışta eski alanlar da doldurulmaya devam eder, eski kayıtlar olduğu gibi okunur. | Geri uyumlu; mevcut portföy ve günlük kayıtları bozulmaz (spec 0004). Tek çıkış alanı kısmi satışı taşıyamaz. |
| Q07 | Fiyat alınamayınca ne gösterilsin? | Önce son günlük kapanış ("gecikmeli fiyat (tarih)" etiketiyle), o da yoksa "fiyat alınamadı" + "hesaplanamıyor". Kâr hiçbir durumda sessizce 0 olmaz. | Grafiğin kullandığı yol zaten çalışıyor (kullanıcı bildirdi); bilinmeyen değer sıfır diye sunulmaz ([conventions.md](../docs/conventions.md)). |

**Açık bilgi (karar değil):** Kârı 0 görünen varlığın sembolü nedir (ör. `AAPL`, `XYZ-USD`)? Kök nedeni (Yahoo/Binance erişimi, sembol biçimi) bulmak için AP-05 gereği en küçük yeniden üretim denenecek; sonuç bu spec'in kararlarını değiştirmez.

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
- [ ] **AC19 — Gecikmeli fiyat, Q07:** Canlı fiyatı alınamayan varlığın satırı son günlük kapanışla hesaplanır, "gecikmeli fiyat" ve tarih etiketi taşır ve kâr giriş fiyatından farklıysa 0 görünmez.
- [ ] **AC20 — Fiyat yok, Q07:** Hiçbir fiyatı olmayan varlığın satırında "fiyat alınamadı" ve kâr/zarar "hesaplanamıyor" yazar; 0 yazmaz.
- [ ] **AC21 — Sonradan eklenen varlık, R10:** Sonradan eklenmiş bir varlığın canlı fiyatı alınamadığında, grafik açılmadan, pozisyon satırı AC19'daki gibi görünür.
- [ ] **AC22 — Toplam, R11:** Fiyatı alınamayan ya da gecikmeli fiyatlı pozisyon varsa toplam değer bunu belirtir.
- [ ] **AC23 — Canlı fiyat varken, R09:** Canlı fiyat alındığında satır bugünkü gibi canlı fiyatla hesaplanır ve etiket taşımaz.

## Definition of Done
- [ ] Q01–Q07 kararları kesinleşti; kriterler güncellendi; spec Takım Yöneticisi tarafından onaylandı.
- [ ] Tüm kabul kriterleri bağımsız pytest testiyle karşılandı; arayüz kriterlerinin ekran görüntüleri eklendi.
- [ ] Geçerli lint ve test kontrolleri yeşil; para/yüzde `Decimal`; teknik hata kullanıcıya sızmadı.
- [ ] Kök neden için en küçük yeniden üretim denendi ve sonucu kaydedildi (AP-05).
- [ ] Ayrı QA oturumu doğrulaması tamamlandı.
- [ ] PR ve squash-merge [git.md](../docs/git.md) kurallarıyla tamamlandı.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 — taslak |
| Düzeltme turu sayısı | Uygulama başlamadı |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | Ölçülmedi |
| Kaçan hata | Ölçülmedi |
