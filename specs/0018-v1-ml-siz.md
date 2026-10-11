# Spec: 0018 — AL/SAT Kararlarından ML'nin Çıkarılması

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-11).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi onayı (2026-10-09, "onayladı"), deney sonuçları üzerine.

## Kanıt (deney, 2026-10-09; BTC, ETH, SOL, XRP, AVAX, DOGE, LINK, ADA; Binance günlük, 1000 gün)
- **Belirlenimcilik:** Aynı veride ML'li iki koşu birebir aynı (BTC, 801 gün, 0 fark). ML'nin kendisi rastgele değil.
- **Kayan pencere:** Uygulama her gün son 1000 günü alır; pencere bir gün kayınca ortak 800 günde değişen karar sayısı: **ML'siz 8 varlıkta da 0**, ML'li 0–7 (BTC 7, LINK 6, SOL 6, XRP 4, AVAX 3, ADA 3, ETH 2, DOGE 0). Aynı geçmiş günün kararı gün gün değişebiliyor.
- **Sonuç:** ML 801 kararın %1–2,5'ini değiştiriyor; toplam K/Z ML'siz +764 $, ML'li −108 $ (8 varlığın 7'sinde kötüleşme). ML'li geçmiş test varlık başına ~10 dk, ML'siz ~14 sn.
- **Sınır:** ML'siz sonuç da kârlı değildir (+764 $'ın kaynağı tek XRP işlemi; XRP hariç ≈ −1.280 $). Bu spec kararları kararlı ve ölçülebilir yapar; stratejiyi kârlı yapmaz.

## Requirements
- **R01:** `DecisionEngineConfig.V1_INCLUDE_ML = False`. AL/SAT üreten tüm yollar (ekran sinyali her zaman dilimi, geçmiş test kararları, ileri takip koşucusu, kağıt ticaret, tarayıcı, portföy) varsayılan olarak ML'siz çalışır; ML ağırlığı diğer boyutlara dağıtılır (mevcut `_compute_bar_score` davranışı).
- **R02:** Karar kuralı değiştiği için yeni strateji sürümü oluşur (sürüm parmak izi `DecisionEngineConfig`'i içerir; koşucu etiketi `/ML-YOK` taşır). Eski sürümün sanal takip kayıtları ayrı kalır, karışmaz.
- **R03:** Ekran metinleri "ML dahil" demez; "ML kararlara girmez" der.
- **R04:** Açıkça `include_ml=True` ile çağıran araştırma yolları çalışmaya devam eder.

## Acceptance Criteria
- [ ] **AC01 — Ayar, R01:** Varsayılan `V1_INCLUDE_ML` False.
- [ ] **AC02 — ML çalışmaz, R01:** Varsayılan ekran sinyali, ham sinyal ve geçmiş test kararları ML modelini çağırmaz.
- [ ] **AC03 — Kararlılık:** Pencere bir gün kayınca ortak günlerin kararları değişmez.
- [ ] **AC04 — Ekran = geçmiş test:** Varsayılan ekran kararı ile geçmiş test kararı her barda aynı.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü; yayında backtest'in saniyeler içinde bittiği görülür

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | ML'nin kararlılığa ve sonuca etkisi hiç ölçülmemişti |
