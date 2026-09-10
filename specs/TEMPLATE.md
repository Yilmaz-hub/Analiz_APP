# Spec: <NNNN> — <Kısa Ad>

> Bu şablondan üret: kopyala → `specs/NNNN-<ad>.md`. Bitince → `specs/done/`.
> Kural: **spec'i olmayan feature'a başlanmaz** (bkz. `AGENTS.md`).

## Intent
<Bu feature'ı KİM, NEDEN istiyor; başarı neye benziyor? 3-5 cümle, İŞ dilinde.
Teknik çözümü değil, iş sonucunu anlat.>

## Requirements
- <Ne yapılmalı — madde madde, iş gereksinimi.>
- <...>

## Constraints
- <Teknik/iş kısıtları. Örn: para/yüzde `Decimal`; arayüz katmanı iş mantığı barındırmaz.>
- **Kapsam dışı:** <bu spec'in KAPSAMADIĞI şeyler — açıkça yaz.>

## Context
- İlgili modül(ler): <kök dizindeki Python modülleri, ör. signal_engine / portfolio / scanner / data_fetchers>
- İlgili alan terimleri: <bu spec'te tanımla — Terimler>
- Bağımlılıklar / ilgili spec'ler: <NNNN, ...>

## Acceptance Criteria
> Her satır **tek başına test edilebilir** olmalı (bkz. docs/testing.md).
- [ ] <Kriter 1 — gözlemlenebilir davranış>
- [ ] <Kriter 2>
- [ ] <Kriter 3>

## Definition of Done
- [ ] Tüm kabul kriterleri test/kanıtla karşılandı.
- [ ] Her kriter bir pytest testi (docs/testing.md).
- [ ] Arayüz: her kritere ekran görüntüsü + kritik akış smoke test.
- [ ] Yeşil test pipeline (pytest).
- [ ] Para/yüzde alanları `Decimal`; kullanıcıya teknik hata / traceback sızmıyor.
- [ ] PR şablonu dolduruldu, squash-merge (docs/git.md).

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | <n> |
| Düzeltme turu sayısı | <n> |
| Bulgu gerçek/gürültü oranı | <gerçek>/<gürültü> |
| Regresyon sayısı | <n> |
| Kaçan hata | <n> |
