# Roller (docs/roles/)

Oturum/rol modelinin **özet tablosu** `AGENTS.md` → *Oturumlar / Roller* bölümündedir
(tek gerçek kaynağı). Bu dosya rol sistemini ve ortak kuralları tanımlar; her rolün
ayrıntısı kendi kartındadır.

## Bir oturuma rol nasıl verilir (üstlenme kalıbı)

Oturumun **ilk satırı** tek satırlık üstlenmedir:

```
⟦ROL: <Analist|Developer|QA>⟧ specs/<NNNN>-<ad> — üstlenildi.
```

Rol üstlenilmeden işe başlanmaz; her oturum tek rol taşır.

## Şapka Değişimi Kuralı

DEV oturumu içinde **Analist → Developer** geçişi **İLAN EDİLİR**:

```
⟦ŞAPKA: Analist → Developer⟧ specs/<NNNN> — INTENT·CLARIFY·SPEC tamam; PLAN·BUILD'e geçiyorum.
```

İlan edilmeden şapka değişmez. **QA ayrı oturumdur**; şapka değişimiyle QA'ya geçilmez
(bkz. `qa.md` imza ilkesi).

## Tüm Rollerin Ortak Kuralları

- Oturum açılışında **`AGENTS.md` okunur.**
- **Kanıtsız iddia yok:** test/ölçüm/repro olmadan "çalışıyor / tamamlandı" denmez.
- **Belirsizlikte AP-10:** VARSAYMA; seçenekleri bedelleriyle getir, yöneticiye sor
  (bkz. `../ap.md`).
- **ÖNERİ KURALI:** soru/bulgu/seçenek **önerisiz sunulmaz** — her biri **öneri + gerekçe**
  ile gelir; **karar Takım Yöneticisinde.**
- Ortak AP'ler: **AP-09** (bağlam bulanıklaştı) ve **AP-10** (belirsizlik / docs-kod çelişkisi).

## Rol Kartları

| Rol | Omurga rozetleri | Dosya |
|-----|------------------|-------|
| 💻 Analist | INTENT · CLARIFY · SPEC | [`analist.md`](analist.md) |
| 💻 Developer | PLAN · BUILD · düzeltme | [`developer.md`](developer.md) |
| 🔍 QA | REVIEW · TEST · VERIFY · repro | [`qa.md`](qa.md) |

> DEV (A) oturumu Analist ve Developer şapkalarını sırayla taşır; QA (B) ayrı, izole oturumdur
> (bkz. `AGENTS.md` → Oturumlar / Roller). EKİP (C) — Faz 12 — bu disiplinin orkestrasyonlu hali.
