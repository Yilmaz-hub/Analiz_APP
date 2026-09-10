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
- <Teknik/iş kısıtları. Örn: para/yüzde decimal; endpoint minimal API.>
- **Kapsam dışı:** <bu spec'in KAPSAMADIĞI şeyler — açıkça yaz.>

## Context
- İlgili modül(ler): <catalog / ordering / pricing / returns — bkz. docs/architecture.md>
- İlgili alan terimleri: <bkz. docs/domain.md>
- Bağımlılıklar / ilgili spec'ler: <NNNN, ...>

## Acceptance Criteria
> Her satır **tek başına test edilebilir** olmalı (bkz. docs/testing.md).
- [ ] <Kriter 1 — gözlemlenebilir davranış>
- [ ] <Kriter 2>
- [ ] <Kriter 3>

## Definition of Done
- [ ] Tüm kabul kriterleri test/kanıtla karşılandı.
- [ ] Backend: her kriter bir xUnit testi (docs/testing.md).
- [ ] Arayüz: her kritere ekran görüntüsü + kritik akış smoke test.
- [ ] Yeşil pipeline (lint + build + test).
- [ ] Para/yüzde alanları decimal; kullanıcıya teknik hata sızmıyor.
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
