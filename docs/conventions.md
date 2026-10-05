# Konvansiyonlar (conventions)

> Adlandırma, katmanlar, hata yönetimi, para. Alan terimleri → ilgili spec'in **Context → Terimler** bölümü.

## Para = decimal (mutlak kural)

- Tüm para ve yüzde alanları **`decimal.Decimal`**. `float` **yasak** (aşağıdaki V1 istisnası hariç).
- `Decimal` değerler `str` olarak saklanır (belge deposunda JSON'a `float` olarak yazılmaz) ve
  `Decimal(str(değer))` ile okunur; sabit ölçek gerektiğinde `quantize` kullanılır
  (para için iki ondalık, yüzde için iki ondalık).
- Para değeri para birimi bilgisiyle taşınır; yuvarlama tek noktada, açıkça yapılır.

## V1 analitik ara hesap istisnası

İşlem tutarlılığı feature'ı için onaylı istisna: mevcut gösterge/ML kütüphanelerinin yalnız analitik ara hesaplarında kayan nokta kullanılabilir. Analitik fiyat/ATR çıktısı finansal hesaba geçerken ondalık gösterimi üzerinden decimal'e dönüştürülür; fiyat adımı yuvarlaması bundan sonra uygulanır. Dönüşüm hassasiyet kazandırmış sayılmaz. İşlem fiyatı, stop, miktar, tutar, komisyon, bakiye ve getiri/yüzde hesapları decimal kalır; finansal defter bu istisnaya dahil değildir (defterin kayıt biçimi için aşağıdaki ayrı istisnaya bakın). Sunum yuvarlaması hesap girdisi olamaz.

## Finansal defter kayıt biçimi istisnası

Karar (Takım Yöneticisi, 2026-10-05): portföy defterinde (`portfolio` belgesi) `Adet`, `Realized`, `Yatırım`,
`Çıkış Adedi` ve `balance` alanları spec 0003'ten beri JSON `float` olarak saklanır; yayındaki gerçek kayıtlar bu
biçimdedir. **Hesap `Decimal` ile yapılır**: kayıttan okunan değer `Decimal(str(değer))` ile alınır, sonuç yazılırken
`float`'a çevrilir. Yeni eklenen alanlar (ör. `Çıkışlar` listesi) `str`/`Decimal` metni olarak yazılır. Defter
alanlarının `Decimal`/metne taşınması, yayındaki kayıtlara dokunan bir göçtür; yedek ve ayrı spec olmadan yapılmaz.

## Adlandırma

- **Python (PEP 8)**: modül/fonksiyon/değişken → `snake_case`; sınıf → `PascalCase`;
  sabit → `UPPER_SNAKE_CASE`; modül içi yardımcı → `_önek`.
- Modül dosya adı küçük harf `snake_case` (ör. `risk_sizing.py`); ekran kodu `*_ui.py`,
  saf hesap kodu Streamlit içermeyen ayrı modüldür.
- **Alan terimleri EN kod adıyla** yazılır; terim ve karşılığı ilgili spec'in **Context → Terimler** bölümündedir.
- Boolean adları soru gibi yazılır (`is_approved`, `has_position`).
- Sonuç taşıyan veri yapıları `@dataclass(frozen=True)` olur (ör. `ReportOutcome`, `SizeResult`).

## Katmanlar

- Ayrı mimari belgesi yoktur; modül haritası kök dizindeki Python modülleridir
  (bkz. `../AGENTS.md` Altın Kural 3). Yeni modül veya katman kararı ilgili spec'te alınır.
- Doğrulama girişte yapılır; geçersiz girdi hesap katmanına ulaşmaz.

## Hata Yönetimi

- Beklenen iş hataları için anlamlı sonuç nesnesi (ör. `status`/`code` + Türkçe `reason`) döner;
  exception ile akış kontrol edilmez.
- Beklenmeyen ve altyapı hataları (depo, ağ) sınırda yakalanıp anlamlı bir hata türüne çevrilir
  (ör. `StorageAccessError`) ve `logger.py` ile loglanır.
- **Kullanıcıya teknik hata metni / stack trace sızmaz** (ekran tarafı için bkz. `frontend.md`).
  Ekran, hata durumunda anlaşılır Türkçe uyarı gösterir ve geri kalanı çalışmaya devam eder.
- Beklenen durumlar uygulama terimleriyle adlandırılır (ör. `BULUNAMADI`, `GECERSIZ_ISTEK`);
  HTTP durum kodlarıyla anlatılmaz.
- Loglar anlamlı ve ayıklanabilir olur; log'a sır (ör. `db_url` parolası) ve kişisel veri yazılmaz.

## Genel

- Bir kural yalnızca burada; başka dosyalar link verir (tek gerçek kaynağı).
- Ölü kod, yorum satırına alınmış kod bırakılmaz; geçmiş `git`tedir.
