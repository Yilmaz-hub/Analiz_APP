# Mimari (architecture)

> Modül haritası + yasaklar + kapsam dışı. Bu bir **iskelet**tir; dilimler ilerledikçe
> genişletilir. Ayrıntılı kod konvansiyonları → `conventions.md`.

## Genel Yapı

```
backend/        .NET minimal API (tek çözüm)
  Api/          minimal API endpoint'leri (MapGroup ile modül başına gruplama)
  Application/  use-case'ler, servisler, DTO'lar, doğrulama
  Domain/       varlıklar (entity), değer nesneleri, iş kuralları — bağımsız katman
  Infrastructure/ EF Core, repository, dış servis erişimi
frontend/       React + Vite + TypeScript  (bkz. frontend.md)
specs/          mini-spec'ler (spec yoksa kod yok)
docs/           kurallar (tek gerçek kaynağı)
```

## Katman Kuralı (bağımlılık yönü)

`Api → Application → Domain` ; `Infrastructure → Domain`.
Ok tersine çevrilmez. `Domain` hiçbir üst katmana ve altyapıya bağımlı olamaz.

## Modül Haritası (iş alanına göre)

| Modül | Sorumluluk | Ana kavramlar (→ `domain.md`) |
|-------|-----------|-------------------------------|
| Catalog | Ürün listeleme, sayfalama, arama | Ürün, Kategori |
| Ordering | Sepet, sipariş oluşturma/durum | Sipariş, Sipariş Kalemi |
| Pricing  | Kupon, indirim, tutar hesaplama | Kupon, İndirim |
| Returns  | İade talebi ve süreci | İade |

Her endpoint minimal API'dir ve ilgili modülün `MapGroup`'una bağlanır.

## Yasaklar

- **Controller sınıfı yok.** Tüm endpoint'ler minimal API (`app.MapGet/MapPost/...`).
- **Para/yüzde için `double`/`float` yok** — her zaman `decimal` (→ `conventions.md`).
- **`Domain` katmanında EF/HTTP/framework bağımlılığı yok.**
- **Spec'siz feature yok** (→ `AGENTS.md`, `specs/TEMPLATE.md`).
- İş mantığı endpoint gövdesine gömülmez; `Application` katmanına iner.

## Kapsam Dışı (bu aşamada)

- Mikroservis / dağıtık mimari (tek çözüm, modüler monolit).
- Çok kiracılı (multi-tenant) yapı.
- Kimlik/oturum sağlayıcı seçimi (ayrı spec konusudur).
- Bu iskelet, ilk dilimlerle birlikte netleşecek somut proje yapısının yerine geçmez.
