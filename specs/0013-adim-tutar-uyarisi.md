# Spec: 0013 — Adet/Lot Adımı ve İşlem Tutarı Tutarsızlığı Uyarısı

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-05).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-05): ETH için "5 adet alım, işlem tutarı 1000 USD, sermaye 10000 USD" girmiş; "5 alım 1000 USD etmez, mantıksızlık var."

## Sorun
"Adet/lot adımı" alanı **en küçük alınabilir miktardır** (ör. ETH için 0.001), "kaç adet alınacağı" değil. Kullanıcı 5 yazınca bir adım 5 ETH (~20.000 USD) olur; işlem tutarı 1000 USD olduğundan miktar sıfıra yuvarlanır ve **hiç alım yapılamaz**. Hiçbir yerde uyarı yoktur; ayrıca etiket yanlış anlaşılmaya açıktır.

## Intent
Tutarsız ayar kaydedilmeden önce ve sanal takipte açıkça uyarılsın; alanın anlamı etikette yazılsın.

## Requirements
- **R01:** Etiket alanın anlamını söyler: "en küçük alınabilir miktar (ör. ETH 0.001)".
- **R02:** Adım × güncel fiyat > işlem tutarı ise panelde uyarı: hiç alım yapılamaz; adımın kaç adet alınacağı olmadığı belirtilir.
- **R03:** Sanal takip koşucusu aynı durumda not düşer ("adet adımı × fiyat işlem tutarını aşıyor; hiç alım yapılamaz").
- **R04:** Kayıt engellenmez (kullanıcı bilerek girebilir); yalnız uyarılır. Hesap mantığı değişmez.

## Acceptance Criteria
- [ ] **AC01 — Hesap, R02:** `step_exceeds_notional` adım × fiyat işlem tutarını aşıyorsa tutarı, aşmıyorsa ya da adım/fiyat bilinmiyorsa `None` döner.
- [ ] **AC02 — Panel, R01/R02:** ETH 5 adım, 1000 tutar, ~fiyat 4000 → uyarı görünür; 0.001 adımda uyarı yok; etiket "en küçük alınabilir miktar" içerir.
- [ ] **AC03 — Koşucu, R03:** Aynı durumda koşucu raporunda not vardır.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Alanın anlamı etikette yoktu; tutarsız ayar sessizce "hiç işlem yok" üretiyordu |
