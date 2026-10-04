# Ekran (UI) Konvansiyonları

> Yığın: **Streamlit** (`app.py`). Ekran kodu Python'dadır; ayrı bir web arayüzü projesi yoktur.

## Temel Kurallar

- **Akışı `app.py` kurar, mantık ayrı modüllerde durur.** Hesap ve karar kodu Streamlit
  içermeyen saf modüllerdedir (ör. `risk_sizing.py`, `strategy_candidates.py`); ekran
  parçaları `*_ui.py` dosyalarındadır (ör. `risk_ui.py`, `candidate_ui.py`). `*_ui.py`
  içinde `streamlit` yalnız `render_*` fonksiyonlarında içe aktarılır; gösterilecek metin
  `build_*_view` gibi saf fonksiyonlarda üretilir ve test edilir.
- **Her ekran parçasında üç durum tasarlanır ve test edilir:** veri **yüklenirken** (ilerleme /
  bekleme), **boş** (kayıt yok), **hata** (kayıt deposu, ağ ya da hesap hatası).
- **Kullanıcıya teknik hata metni sızmaz.** Stack trace, ham veritabanı/ağ hatası ve sınıf
  adları gösterilmez; anlaşılır Türkçe uyarı verilir ve ekranın geri kalanı çalışır
  (bkz. `docs/conventions.md`).
- **Engel varken sonuç uydurulmaz.** Doğrulanamayan değer sıfır gösterilmez; neden metniyle
  ("hesaplanamıyor", "fiyat alınamadı…") belirtilir.
- **Widget anahtarları (`key=`) kararlıdır ve varlık/dönem bilgisini taşır** (ör.
  `risk_min:{symbol}`); aynı anahtar iki yerde kullanılmaz.
- **Dilimler spec'e bağlıdır.** Her ekran/özellik `specs/` altındaki bir spec'e ve kabul
  kriterine bağlanır.

## Para / Yüzde

- Para ve yüzde `Decimal` ile hesaplanır; kayan noktada hesap yapılmaz. Biçimleme yalnız
  gösterimde yapılır ve sunum yuvarlaması hesap girdisi olamaz.
- Kullanıcıdan gelen metin girdisi (ondalık virgül dahil) `Decimal`e çevrilmeden önce
  doğrulanır; geçersiz girdi hesap katmanına ulaşmaz.

## Kalite Kapıları (CI)

- Testler (`pytest`) yeşil olmadan PR merge edilmez (bkz. `docs/git.md`, `docs/testing.md`).
- Ekran davranışı `streamlit.testing.v1.AppTest` ile sınanır; her kabul kriterinin testi
  ve gerekli yerde ekran görüntüsü kanıtı `docs/evidence/` altında tutulur.
- `app.py` CRLF satır sonlarını korur; dosyayı satır sonu bozmadan düzenleyin.
