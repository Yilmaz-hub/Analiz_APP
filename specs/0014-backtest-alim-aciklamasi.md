# Spec: 0014 — Geçmiş Testte Yapılmayan Alımların Açıklanması

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-07).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-07): "ETH için 1 günlükte backtest yapmak istedim. Sonuç 231 alım yapılamadı, hiçbir getiri vermedi. Birçok veri istedi, ne istediği belli değil. Neden alım yapmadı, bu sistem düzgün çalışmıyor."

## Kök neden (kodla doğrulandı)
`enter_position` adım 5, tutar 1000, fiyat ~4500 için `ASGARI_MIKTAR` döndürür: 1000 / 4500 = 0,22 ETH, 5'lik adıma aşağı yuvarlanınca 0. Her AL sinyali (231) bu yüzden işleme dönüşmedi. Adım boş olsaydı neden `MIKTAR_ADIMI_BILINMIYOR` olurdu. Motor doğru çalışıyor; ekran nedeni "Miktar asgari işlem miktarının altında" gibi kullanıcının hangi ayarı değiştireceğini söylemeyen bir metinle yazıyor ve panelde alanların ne istediği açıklanmıyor.

## Requirements
- **R01:** Geçmiş testte yapılmayan alımların nedeni, kullanıcının değiştirebileceği ayar adıyla ve somut sayılarla yazılır (işlem tutarı, adım) ve ne yapılacağı söylenir.
- **R02:** İşlem varsayımları panelinde her alanın açıklaması (help) vardır; zorunlu tek alanın adet/lot adımı olduğu yazılır.
- **R03:** Hesap mantığı değişmez.

## Acceptance Criteria
- [ ] **AC01 — Büyük adım, R01:** `ASGARI_MIKTAR` metni tutarı, adımı ve örnek değeri (ETH 0.001) içerir; ham kod yoktur.
- [ ] **AC02 — Adım yok, R01:** `MIKTAR_ADIMI_BILINMIYOR` metni panele ve örnek değere yönlendirir.
- [ ] **AC03 — Diğer nedenler, R01:** Diğer kodlar Türkçe metinle yazılır.
- [ ] **AC04 — Ekran, R01:** Büyük adımla geçmiş test, ekranda nedeni ayar adıyla açıklar.
- [ ] **AC05 — Alan açıklamaları, R02:** Altı alanın hepsinde help vardır; zorunlu alan yazılıdır.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Engel kodu doğruydu ama metni eyleme dönük değildi; panel alanları açıklamasızdı |
