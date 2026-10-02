# Konvansiyonlar (conventions)

> Adlandırma, katmanlar, hata yönetimi, para. Alan terimleri → ilgili spec'in **Context → Terimler** bölümü.

## Para = decimal (mutlak kural)

- Tüm para ve yüzde alanları **`decimal`**. `double`/`float` **yasak**.
- DB tarafında para için sabit ölçek kullanılır (ör. `decimal(18,2)`); yüzde için yeterli
  ölçek (ör. `decimal(5,2)`).
- Para değeri para birimi bilgisiyle taşınır; yuvarlama tek noktada, açıkça yapılır.

## V1 analitik ara hesap istisnası

İşlem tutarlılığı feature'ı için onaylı istisna: mevcut gösterge/ML kütüphanelerinin yalnız analitik ara hesaplarında kayan nokta kullanılabilir. Analitik fiyat/ATR çıktısı finansal hesaba geçerken ondalık gösterimi üzerinden decimal'e dönüştürülür; fiyat adımı yuvarlaması bundan sonra uygulanır. Dönüşüm hassasiyet kazandırmış sayılmaz. İşlem fiyatı, stop, miktar, tutar, komisyon, bakiye ve getiri/yüzde hesapları decimal kalır; finansal defter bu istisnaya dahil değildir. Sunum yuvarlaması hesap girdisi olamaz.

## Adlandırma

- **C#**: tip/metot/`public` üye → `PascalCase`; yerel/parametre → `camelCase`;
  `private` alan → `_camelCase`; sabit → `PascalCase`; arayüz → `IAd`.
- Dosya adı = içindeki ana tip adı.
- **Alan terimleri EN kod adıyla** yazılır; terim ve karşılığı ilgili spec'in **Context → Terimler** bölümündedir.
- Boolean adları soru gibi yazılır.
- DTO'lar `...Request` / `...Response` son ekiyle.

## Katmanlar

- Ayrı mimari belgesi yoktur; modül haritası kök dizindeki Python modülleridir
  (bkz. `../AGENTS.md` Altın Kural 3). Yeni modül veya katman kararı ilgili spec'te alınır.
- Doğrulama girişte yapılır; geçersiz girdi hesap katmanına ulaşmaz.

## Hata Yönetimi

- Beklenen iş hataları için anlamlı sonuç (ör. `Result`/`ProblemDetails`), exception'la akış kontrol edilmez.
- Beklenmeyen hatalar merkezi bir hata middleware'inde yakalanır ve loglanır.
- **Kullanıcıya teknik hata metni / stack trace sızmaz** (frontend tarafı için de geçerli, bkz. `frontend.md`).
- HTTP durum kodları anlamlı: doğrulama `400`, yetki `401/403`, bulunamadı `404`, çakışma `409`.
- Loglar yapılandırılmış (structured) olur; log'a sır/PII yazılmaz.

## Genel

- Bir kural yalnızca burada; başka dosyalar link verir (tek gerçek kaynağı).
- Ölü kod, yorum satırına alınmış kod bırakılmaz; geçmiş `git`tedir.
