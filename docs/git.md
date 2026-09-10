# Git İş Akışı

## Branch Stratejisi

- Biçim: **`feature/<spec-no>-<ad>`** — ör. `feature/0001-catalog-pagination`.
- **SPEC'SİZ BRANCH AÇILMAZ.** Branch'in adındaki `<spec-no>` `specs/` altındaki bir
  mini-spec'e karşılık gelmelidir.
- `main` her zaman yeşil ve dağıtılabilir kalır.

## Commit Formatı

**Conventional Commits + plan atıfı:**

```
feat(catalog): sayfalama endpoint'i [plan 0001/3]
```

- Tipler: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`.
- Kapsam (`scope`) modül adıdır (`catalog`, `ordering`, `pricing`, `returns`).
- `[plan <spec-no>/<adım>]` atıfı zorunlu.
- **AI kuralı:** Ekip üyesi (ajan) commit'leri de aynı standarda uyar. Mesajı **üye yazar,
  Takım Yöneticisi onaylar.**

## Yasaklar

- `main`'e **doğrudan commit yok.**
- **force push yok.**
- **History silme yok.** Bir değişikliği geri almak = **`revert`** (yeni commit).

## PR Şartları

- **PR şablonu** doldurulur (spec atıfı, kabul kriterleri, kanıt/ekran görüntüsü).
- **Yeşil pipeline zorunlu** (lint + build + testler).
- En az bir onay: mesajı/PR'ı üye hazırlar, **Takım Yöneticisi onaylar**.

## Merge

- **Squash-merge** kuralı: özellik dalı `main`'e tek, temiz commit olarak alınır.
- Squash commit başlığı Conventional Commits + plan atıfına uyar.

## PR Şablonu (öneri)

```
## Spec
- İlgili spec: specs/NNNN-<ad>.md

## Ne değişti
- ...

## Kabul kriterleri
- [ ] Kriter 1 → test/kanıt
- [ ] Kriter 2 → test/kanıt

## Kanıt
- (arayüz için ekran görüntüsü / backend için test çıktısı)

## Kontroller
- [ ] Yeşil pipeline
- [ ] Para/yüzde alanları decimal
- [ ] Kullanıcıya teknik hata sızmıyor
```
