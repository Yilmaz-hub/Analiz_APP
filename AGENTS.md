# AGENTS.md — İşaret Levhası

> Bu dosya **işaret eder, kural kopyalamaz.** Ayrıntı her zaman `docs/` altındadır.
> Tek gerçek kaynağı ilkesi: bir kural yalnızca tek bir yerde yazılır; başka yerde ona
> **link** verilir.

## Proje Özeti

Spec-güdümlü (spec-driven) geliştirilen bir üründür. Backend **.NET minimal API**,
frontend **React + Vite + TypeScript**. Her iş parçası önce `specs/` altında bir
mini-spec olarak yazılır, sonra kodlanır. Ekip; öneri getiren üyeler (ajanlar) ve
kararı veren **Takım Yöneticisi**nden oluşur.

## Oturumlar / Roller

| Oturum | Rol | Ne yapar | Ne yapamaz |
|---|---|---|---|
| 💻 **DEV (A)** — Analist + Developer | spec → plan → build → düzeltme | Üretir; feature'ın bağlamını taşır | Kendi işini denetleyemez |
| 🔍 **QA (B)** — Denetçi | review + verify + repro | Okur; test/ölçüm **çalıştırır**; kanıtlı bulgu üretir | **Kod yazamaz.** A'nın sohbet geçmişini göremez |
| 👥 **EKİP (C)** (Faz 12) | Planner / Developer / Reviewer / Security / Test | A ve B disiplininin rollere dağıtılmış, orkestrasyonlu hali | İnsan checkpoint'lerini atlayamaz |

**İlke:** Üreten denetlemez; denetleyen üretmez. B, A'nın bağlamından izole çalışır ve
bulgusunu kanıtla (test/ölçüm/repro) getirir. Kararı her zaman **Takım Yöneticisi** verir
(bkz. Altın Kural 4, öneri kuralı). C, aynı disiplinin orkestrasyonlu halidir; insan
checkpoint'leri atlanamaz. Rol ayrıntıları: `docs/roles/`.

## Altın Kurallar (en fazla 7)

1. **Spec yoksa kod yok.** `specs/` içinde spec'i olmayan feature'a başlanmaz. → `specs/TEMPLATE.md`
2. **Para ve yüzde her zaman `decimal`.** `double`/`float` yasak. → `docs/conventions.md`
3. **Endpoint'ler minimal API.** Controller sınıfı açılmaz. → `docs/architecture.md`
4. **Öneri kuralı:** Soru/seçenek/bulgu/açık konu getiren, kendi **önerisini ve
   gerekçesini** de getirir. Kararı **Takım Yöneticisi** verir; öneri **onaysız uygulanmaz.**
5. **`main` korunur.** SPEC'siz branch açılmaz (`feature/<spec-no>-<ad>`); `main`'e
   doğrudan commit / force push / history silme yasak; geri alma = **revert**. → `docs/git.md`
6. **Her kabul kriteri tek başına test edilebilir; her kriter bir test.** → `docs/testing.md`
7. **İşaret et, kopyalama.** Kuralı tek yerde tut, başka yerden link ver.

## Neyi Nerede Bulursun

| Konu | Dosya |
|------|-------|
| Modül haritası, yasaklar, kapsam dışı | `docs/architecture.md` |
| Ubiquitous language + iş kuralları (sipariş/kupon/iade) | `docs/domain.md` |
| Adlandırma, katmanlar, hata yönetimi, para=decimal | `docs/conventions.md` |
| Branch/commit/PR/merge kuralları | `docs/git.md` |
| Test yaklaşımı (backend xUnit + arayüz görsel kanıt) | `docs/testing.md` |
| Frontend konvansiyonları (React+Vite+TS iskeleti) | `docs/frontend.md` |
| Spec şablonu (SCORECARD dahil) | `specs/TEMPLATE.md` |
| Bitmiş spec'ler | `specs/done/` |
| **Roller** — oturum açılışında rolünü üstlen | `docs/roles/` |
| **Sorun protokolleri** — AP kodlarının çözümü | `docs/ap.md` |

## Çalışma Disiplini

1. **Spec önce.** `specs/TEMPLATE.md`'den `specs/NNNN-<ad>.md` üret; Intent iş dilinde
   yazılır, her kabul kriteri tek başına test edilebilir olur.
2. **Onay al.** Öneri/soru/bulgu → gerekçeli öneri → Takım Yöneticisi kararı.
3. **Branch aç.** `feature/<spec-no>-<ad>`. Spec numarası olmadan branch açılmaz.
4. **Küçük dilimler.** Her dilim (mini-spec) yeşil pipeline ile ilerler.
5. **Kanıt üret.** Backend: her kriter bir xUnit testi. Arayüz: her mini-spec kriterine
   ekran görüntüsü + kritik akışa smoke test.
6. **Commit + PR.** Conventional Commits + plan atıfı; PR şablonu + yeşil pipeline;
   squash-merge. Mesajı üye yazar, Takım Yöneticisi onaylar.
7. **Kapat.** Bitmiş spec `specs/done/` altına taşınır; SCORECARD doldurulur.
