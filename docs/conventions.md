# Konvansiyonlar (conventions)

> Adlandırma, katmanlar, hata yönetimi, para. Alan terimleri → ilgili spec'in **Context → Terimler** bölümü.

## Para = decimal (mutlak kural)

- Tüm para ve yüzde alanları **`decimal`**. `double`/`float` **yasak**.
- DB tarafında para için sabit ölçek kullanılır (ör. `decimal(18,2)`); yüzde için yeterli
  ölçek (ör. `decimal(5,2)`).
- Para değeri para birimi bilgisiyle taşınır; yuvarlama tek noktada, açıkça yapılır.

## V1 analitik ara hesap istisnası

İşlem tutarlılığı feature'ı için onaylı istisna: mevcut gösterge/ML kütüphanelerinin yalnız analitik ara hesaplarında kayan nokta kullanılabilir. Analitik fiyat/ATR çıktısı finansal hesaba geçerken ondalık gösterimi üzerinden decimal'e dönüştürülür; fiyat adımı yuvarlaması bundan sonra uygulanır. Dönüşüm hassasiyet kazandırmış sayılmaz. İşlem fiyatı, stop, miktar, tutar, komisyon, bakiye ve getiri/yüzde hesapları decimal kalır; finansal defter bu istisnaya dahil değildir. Sunum yuvarlaması hesap girdisi olamaz.

## Adlandırma (Python)

- Modül ve dosya adları `snake_case` (`trade_execution.py`); fonksiyon/değişken `snake_case`;
  sınıf `PascalCase`; sabit `UPPER_SNAKE_CASE`. Test dosyası `tests/test_<modül>.py` (bkz. `testing.md`).
- Değer nesneleri `@dataclass(frozen=True)` ile tanımlanır.
- Alan terimleri kodda İngilizce adla yazılır; terim ve karşılığı ilgili spec'in
  **Context → Terimler** bölümündedir. Kullanıcıya görünen metin Türkçedir.
- Boolean adları soru gibi yazılır (`net_verified`).
- Ayarlanabilir sayılar ve eşikler (`250`, `0.60`, …) `config.py` içindeki bir sınıfta
  durur; hesap kodunda sihirli sayı bırakılmaz.

## Katmanlar

- Ayrı mimari belgesi yoktur; modül haritası kök dizindeki Python modülleridir
  (bkz. `../AGENTS.md` Altın Kural 3). Yeni modül veya katman kararı ilgili spec'te alınır.
- Doğrulama girişte yapılır; geçersiz girdi hesap katmanına ulaşmaz.

## Hata Yönetimi

- Beklenen iş hataları (geçersiz girdi, kayıt yok, yetersiz nakit) istisna değil **sonuç nesnesi**
  ile döner: durum + neden kodu (`Purchase(False, "YETERSIZ_NAKIT")`,
  `Entry(False, "GECERSIZ_MALIYET")`). Exception'la akış kontrol edilmez.
- Bu uygulamada HTTP API yoktur: "400/404" gibi kodlar spec'lerde kullanılmaz; karşılığı
  "geçersiz istek" ve "bulunamadı" durumlarıdır.
- Beklenmeyen hatalar `logger.py` üzerinden loglanır; log'a sır/PII yazılmaz.
- **Kullanıcıya teknik hata metni / stack trace / makine kodu sızmaz.** Neden kodları
  `trading_ui.describe_code` gibi tek bir yerden Türkçe metne çevrilir (bkz. `frontend.md`).

## Genel

- Bir kural yalnızca burada; başka dosyalar link verir (tek gerçek kaynağı).
- Ölü kod, yorum satırına alınmış kod bırakılmaz; geçmiş `git`tedir.
