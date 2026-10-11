# Spec: 0019 — Trend Filtreli Aday (C) ve Ağırlıklı Portföy Testi (P1)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 2 (2026-10-11; altın %20 ve para birimi kuralı).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi onayı (2026-10-11, "Onaylıyorum"; ağırlıklar "Evet kalsın").

## Kanıt (deney, ML'siz V1; Binance günlük; komisyon %0,1; kurallar sonuç görülmeden belirlendi)
- **Tek varlık, 2018–2026, 8 kripto:** C (V1 AL + kapanış > 200 günlük basit ortalama) V1'den 7/8 varlıkta iyi; ortalama yıllık %18,7 → %24,3. Kâr hedefi (2R) ve 1R/2R kademeli satış getiriyi düşürdü.
- **Portföy (tek kasa), ağırlıklar BTC/ETH %20, SOL/XRP/ADA/AVAX/LINK %10, DOGE %5 (girişteki toplam varlığın yüzdesi):**

| Dönem | Al-tut (aynı ağırlıklar) | C ayrı kasalar (aynı ağırlıklar) | **C + P1 (ortak kasa)** |
|---|---|---|---|
| 2018-11 – 2026 | yıllık %39,8, düşüş −%86 | %41,3, −%56 | **%55,7, −%66** |
| 2021-05 – 2026 | −%0,8, −%83 | %7,0, −%46 | **%13,1, −%47** |

- **Destekte / 1 ATR altında ekleme** (sinyal hâlâ AL iken) tutarlı iyileşme vermedi (2021–2026'da yıllık %13 → %5–6); eklenen işlemlerin ~2/3–3/4'ü stopla kapandı. Kapsam dışı.
- **Sınırlar:** geçmiş veri; bugün var olan 8 bilinen coin (batanlar yok); kayma yok; sonuçlar gelecek getirisi değildir.

## Ek kanıt: altın (GC=F, 2016–2026)
- Tek başına: al-tut yıllık %12,9, C %8,3 (12 işlem); 2018→ al-tut %16,7 / −%25, C %12,2 / −%20.
- Kripto P1'e %20 altın eklenince yıllık getiri 2018→ %54,8 → %61,8; 2021→ %12,3 → %19,0; 2024→ %10,3 → %20,2 (düşüş −%35 → −%27).

## Requirements
- **R01 — Aday kural (C):** Yeni giriş kuralı `TREND_SMA`: V1 kararı AL **ve** kapanış > son N günün basit ortalaması (varsayılan N=200, ayar `sma`). Çıkış V1 (SAT ve 2,5 ATR stop). Bu kural piyasa koşulu (yükselen) kısıtına tabi değildir; filtre ortalamanın kendisidir. Onaylı giriş kuralları listesine eklenir ve varsayılan adaylar arasında yer alır.
- **R02 — Sanal takipte V1'in yanında:** İleri takip koşucusu, aktif stratejiyle birlikte `TREND_SMA` adayının kararlarını da kendi sürüm etiketiyle (ADAY-…) kaydeder; aday ön kaydı koşucu tarafından yapılır (yoksa). Aktif strateji değişmez (kullanıcının açık seçimi gerekir).
- **R03 — Portföy backtest'i:** Seçilen varlıklar tek kasayı paylaşır. Her girişte, o günkü toplam varlığın varlığa ait ağırlığı kadar (nakit yetmezse mevcut nakitle; ağırlığın yarısından azı kalıyorsa giriş yapılmaz) alım yapılır. Kurallar tek varlık motoruyla aynı: karar ertesi açılışta, sabit 2,5 ATR stop, SAT'ta tamamı satılır, zararlı çıkıştan sonra 2 gün bekleme, kullanıcı komisyonu. Para hesabı Decimal.
- **R04 — Varsayılan ağırlıklar:** BTC, ETH ve ons altın (XAU) %20; diğer kriptolar %10; DOGE %5 (kullanıcı değiştirebilir; altın eklenmesi 2026-10-11 onayı). Varsayılan seçim: kriptolar + USD altın. Ağırlık toplamı %100'ü geçebilir (hepsi aynı anda açık olmaz); nakit yetmezse R03 kuralı uygulanır.
- **R05 — Rapor:** Toplam ve yıllık getiri, en büyük düşüş (günlük kapanış), işlem / kazanan sayısı, aynı ağırlıklarla al-tut karşılaştırması, varlık başına katkı, sermaye eğrisi. Al-tut: her varlık ilk günde ağırlığı kadar alınır ve tutulur.
- **R07 — Para birimi:** Tek kasada farklı para birimli varlıklar (ör. USD ve TL) seçilirse test çalışmaz, uyarı verilir.
- **R06:** Veri uygulamanın günlük verisidir (son 1000 gün; 200 günlük ısınma sonrası ~2,2 yıl); bu ekranda yazılır. Sonuç geçmiş veridir, gelecek getirisi değildir.

## Acceptance Criteria
- [ ] **AC01 — Aday karar, R01:** `TREND_SMA` AL'i yalnız V1 AL ve kapanış > SMA iken üretir; SAT V1'den geçer; ayar `sma` kararı değiştirir; yükselen koşulu aranmaz.
- [ ] **AC02 — Varsayılan aday, R01:** `default_candidates` `TREND_SMA` adayını içerir ve onaylıdır.
- [ ] **AC03 — Paralel takip, R02:** Koşucu bir varlık için hem aktif (V1) hem `TREND_SMA` sürümünün kararını kaydeder; sürümler ayrıdır.
- [ ] **AC04 — Tek kasa, R03:** Bir varlıkta açık pozisyon varken diğer varlıkta AL kalan nakitle girer; ağırlık toplam varlığın yüzdesidir; nakit yetmezse giriş sınırlanır/atlanır.
- [ ] **AC05 — Motor eşliği, R03:** Tek varlık %100 ağırlıkla portföy testi, aynı kurallarla tek varlık motorunun işlem giriş/çıkış günleriyle aynıdır.
- [ ] **AC06 — Ağırlıklar, R04/R07:** Varsayılan ağırlıklar BTC/ETH/XAU 20, DOGE 5, diğer 10; karışık para biriminde test başlatılamaz.
- [ ] **AC07 — Rapor, R05/R06:** Getiri, yıllık, düşüş, işlem sayısı, al-tut ve varlık katkısı üretilir; ekran veri sınırını yazar.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü; yayında portföy backtest'i çalışır

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Geçmiş test yalnız tek varlıklıydı; portföy düzeyinde nakit paylaşımı hiç ölçülmemişti |
