# Git İş Akışı

## Branch Stratejisi

- Biçim: **`feature/<spec-no>-<ad>`** — ör. `feature/0003-islem-tutarliligi`.
- **SPEC'SİZ BRANCH AÇILMAZ.** Branch'in adındaki `<spec-no>` `specs/` altındaki bir
  mini-spec'e karşılık gelmelidir.
- `main` her zaman yeşil ve çalıştırılabilir kalır.

## Commit Formatı

**Conventional Commits + spec/plan atıfı:**

```
feat(kayitlar): varlik ve pozisyon kayitlarini koru [spec 0004]
fix(chart): mobilde legend'i yatay/uste al [spec 0002 rev1]
chore(spec): 0001 ve 0002'yi done/ altina tasi [plan 0002/3]
```

- Tipler: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`.
- Kapsam (`scope`) değişen alanın adıdır: `chart`, `trading`, `kayitlar`, `signal`,
  `scanner`, `spec`, `docs`, `ci` …
- Atıf zorunludur: `[spec <NNNN>]` (gerekirse `revN`) ya da plan adımı için
  `[plan <NNNN>/<adım>]`.
- **AI kuralı:** Ekip üyesi (ajan) commit'leri de aynı standarda uyar. Mesajı **üye yazar,
  Takım Yöneticisi onaylar.**

## Yasaklar

- `main`'e **doğrudan commit yok.**
- **force push yok.**
- **History silme yok.** Bir değişikliği geri almak = **`revert`** (yeni commit).

## PR Şartları

- **PR şablonu** doldurulur (spec atıfı, kabul kriterleri, kanıt/ekran görüntüsü).
- **Yeşil pipeline zorunlu:** `.github/workflows/tests.yml` → `python -m pytest tests/ -v`.
- En az bir onay: mesajı/PR'ı üye hazırlar, **Takım Yöneticisi onaylar**.

## Merge

- **Squash-merge** kuralı: özellik dalı `main`'e tek, temiz commit olarak alınır.
- Squash commit başlığı Conventional Commits + spec/plan atıfına uyar.

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
- (arayüz için ekran görüntüsü / pytest çıktısı)

## Kontroller
- [ ] Yeşil pipeline (pytest)
- [ ] Para/yüzde alanları Decimal
- [ ] Kullanıcıya teknik hata / traceback sızmıyor
```
