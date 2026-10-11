# Spec: 0021 — Stopsuz Al-Tut Yöntemi, TL Varlıkların Dolara Çevrilmesi

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-11).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi onayı (2026-10-11, "1 evet dolar bazında"; "Onayliyorum. Altinda al-tut kapsamına alınabilir").

## Kanıt (deney; yöntemler ve ölçüt sonuç görülmeden yazıldı; dolar bazında; komisyon %0,1)
- **Hisseler (2014–2026; QQQ + 15 Nasdaq, XU100 + 15 BIST; BIST günlük USD/TRY ile dolara çevrildi):** C, kanal 50/20 ve RSI(2) dip, iki dönemde de (≤2022-06 / 2022-07+) al-tut'un getiri/düşüş oranını geçemedi. Nasdaq medyan yıllık: al-tut %23 / %26, C %8 / %10. BIST: 2014–2022 her yöntem dolar bazında zararda (al-tut −%9), 2022+ al-tut %18. Ön kayıtlı ölçüte göre iki borsada da zamanlama yöntemi seçilmedi → al-tut.
- **Altın:** tek başına al-tut zamanlamadan iyi; ancak ortak kasada C ile tutulan altın, al-tut'tan iyi ya da eşit: yıllık 2018→ %62,3 vs %57,3; 2021→ %24,1 vs %20,8; 2022→ %21,0 vs %21,1; 2024→ %19,2 vs %18,3 (C'de düşen altından boşalan nakit kripto girişlerine gider). Bu yüzden altın varsayılan C kalır, al-tut seçenektir.
- **Sınırlar:** bugünkü listelerden seçilmiş hisseler (hayatta kalma yanlılığı); temettü ve nakit faizi yok; geçmiş veri, gelecek getirisi değildir.

## Requirements
- **R01 — Tut yöntemi:** Portföy motorunda `hold` varlıkları karar ve stop kullanmaz: ilk işlem gününde, o günkü toplam varlığın ağırlığı kadar (nakit yetmezse mevcut nakitle; ağırlığın yarısından azı kalıyorsa ertesi gün yeniden denenir) alınır ve dönem sonuna kadar tutulur. Tut alımı o günün zamanlamalı girişlerinden önce yapılır. Net K/Z dönem sonu değeri − toplam maliyettir.
- **R02 — Varlık sınıfına göre yöntem:** `asset_method`: altcoin → kanal (0020); hisse (ABD, BIST) → tut; altın → C (panel seçeneği "Altını al-tut ile tut" açıkken tut); diğerleri → C. Altcoin sayılmayanlar BTC/ETH'ye ek olarak sarılı/stake türevleri (WBTC, BTCB, WETH, stETH…) ve sabit coinlerdir (USDT, USDC, DAI…).
- **R03 — Sanal takip:** Tut yöntemli varlıkta (hisse) izlenecek karar olmadığından aday sürümü açılmaz; aktif strateji (V1) takibi sürer. Altın takibi C ile sürer.
- **R04 — Dolar kasası:** Tek kasa dolardır. USDT dolar sayılır; TL varlıklar (BIST, gram altın) günlük USD/TRY kuruyla (takvim gününe göre, eksik günde son bilinen kur) dolara çevrilir; ATR de aynı kurla ölçeklenir. Kur alınamazsa varlık teste alınmaz ve nedeni yazılır. Bu, 0019 R07'nin (karışık para biriminde test yok) yerini alır.
- **R05 — Varsayılan ağırlık:** BIST hisseleri %5 (dolar bazında güvenilir değil); diğer varsayılanlar 0019 R04.
- **R06 — 0020 QA düşük bulguları:** kural notları yalnız kullanılan yöntemler için yazılır; ileri takip tablosunda, aynı varlığın başka sürümü 3 günden fazla ilerideyse sürüm "Takip durdu" diye işaretlenir; kanal kararında gereksiz hesap yapılmaz; 0020 R01 pencere metni netleştirildi.

## Acceptance Criteria
- [ ] **AC01 — Tut, R01:** Tut varlığı ilk gün ağırlığı kadar alınır, stop seviyesinin altına düşse de satılmaz; Net K/Z dönem sonu değeridir; nakit zamanlamalı varlıkla paylaşılır.
- [ ] **AC02 — Yöntem, R02/R05:** AAPL ve THYAO.IS tut; altın varsayılan C, seçenekle tut; altcoin kanal; BTC/ETH, türevleri ve sabit coinler C; hissede aday yok; BIST varsayılan ağırlığı %5.
- [ ] **AC03 — Kur, R04:** TL fiyatlar takvim gününün kuruna bölünür; eksik günde son bilinen kur; ilk kurdan önceki günler atılır.
- [ ] **AC04 — Ekran, R02/R04:** BTC + BIST hissesi + altın birlikte test edilir; yöntem sütunu hisseyi al-tut, altını seçeneğe göre gösterir; TL varlık seçilince kur notu çıkar, test engellenmez.
- [ ] **AC05 — Sanal takip, R03:** Koşucu hissede yalnız V1 sürümünü kaydeder.
- [ ] **AC06 — Durmuş sürüm, R06:** Güncellenmeyen sürüm "Takip durdu" olarak işaretlenir.
- [ ] **AC07 — Notlar, R06:** Kural notları kullanılan yöntemlere göredir.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü; yayında portföy testi hisse ve TL varlıklarla çalışır

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | — |
