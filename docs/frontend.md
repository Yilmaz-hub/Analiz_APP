# Frontend Konvansiyonları (İSKELET)

> Yığın: **React + Vite + TypeScript.** Bu dosya iskelettir. İlk dilimde (**1.8**)
> profesyonel standarda genişletilecek: tasarım token'ları, bileşen envanteri, kalite kapıları.

## Temel Kurallar

- **Tek API client.** Tüm API çağrıları tek bir client dosyasından geçer (ör.
  `src/api/client.ts`); bileşenler doğrudan `fetch`/`axios` çağırmaz.
- **Her ekranda üç durum zorunlu:** **yükleme**, **boş**, **hata**. Üçü de tasarlanır ve
  test edilir.
- **Kullanıcıya teknik hata metni sızmaz.** Stack trace / ham hata mesajı gösterilmez;
  kullanıcıya anlaşılır mesaj verilir (bkz. `docs/conventions.md`).
- **Dilimler mini-spec'le koşar.** Her ekran/özellik `specs/` altındaki bir spec'e bağlıdır.

## Yapı (öneri)

```
frontend/
  src/
    api/        tek client + tip tanımları
    components/ paylaşılan bileşenler
    features/   modül bazlı ekranlar (catalog, ordering, ...)
    lib/        yardımcılar
```

## Para / Yüzde

- Para ve yüzde değerleri backend'den geldiği gibi (string/decimal-uyumlu) taşınır; kayan
  noktada hesap yapılmaz. Biçimleme tek bir yardımcı üzerinden yapılır.

## Kalite Kapıları (CI)

- **lint + build** yeşil olmadan PR merge edilmez (bkz. `docs/git.md`, `docs/testing.md`).
- Kritik akışa smoke test + her kritere ekran görüntüsü kanıtı (bkz. `docs/testing.md`).

## 1.8'de Genişletilecek (yer tutucu)

- [ ] Tasarım token'ları (renk, tipografi, boşluk, yarıçap).
- [ ] Bileşen envanteri (buton, input, tablo, boş/hata/yükleme durumları).
- [ ] Erişilebilirlik ve kalite kapıları eşiği.
