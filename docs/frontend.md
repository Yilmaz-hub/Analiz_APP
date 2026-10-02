# Frontend Konvansiyonları (Streamlit)

> Yığın: **Python 3 + Streamlit** (`app.py`). Eski React/Vite iskeleti bu depoya ait değildi ve
> kaldırıldı. Test ve ekran kanıtı kuralları `testing.md`'dedir; bu dosya yalnız arayüz yapısını anlatır.

## Temel Kurallar

- **`app.py` akışı kurar, iş kuralı yazmaz.** Hesap Streamlit'siz, saf modüllerdedir; ekran metni
  ve düzeni `*_ui.py` modüllerinde (ör. `trading_ui.py`).
- **Görünüm modeli + ince çizim.** `*_ui.py` önce testlenebilir bir görünüm modeli üretir
  (ör. `build_decision_panel`), ardından yalnız onu `st.*` ile çizen küçük bir fonksiyon çağrılır.
- **Her ekranda üç durum tasarlanır ve test edilir:** yükleme, boş, hata.
- **Ölçülemeyen değer sıfır değil "hesaplanamıyor" görünür**; eksik maliyet gibi güveni azaltan
  durumlar sonucun yanında, ek tıklama gerektirmeden durur.
- **Kullanıcıya teknik hata metni sızmaz.** Stack trace, makine kodu ve ham istisna gösterilmez;
  anlaşılır Türkçe mesaj verilir (bkz. `conventions.md` → Hata Yönetimi).
- **Dilimler mini-spec'le koşar.** Her ekran/özellik `specs/` altındaki bir spec'e bağlıdır.

## Streamlit'e özgü

- Düğmeyle üretilen sonuç `st.session_state` içinde tutulur; yoksa sonuç içindeki bir filtre
  veya `date_input` yeniden çalıştırmada sonucu siler. Widget anahtarları kapsamı içerir
  (ör. `ts:BTC-USD:quantity_step`: varlık + alan).
- Dar ekranda uzun metin kesilebilir: metrikler en çok iki sütun, açıklayıcı etiket değerde
  değil etikette durur. Mobil/masaüstü görünümü spec 0001'deki görünüm modu kontrolünden gelir.

## Para / Yüzde

- Para ve yüzde `Decimal` taşınır; sunum yuvarlaması yalnız çizim katmanında yapılır ve hesap girdisi
  olamaz (bkz. `conventions.md`). Para birimleri birbirine eklenmez, ayrı gösterilir.

## Kalite Kapıları (CI)

- Testler yeşil olmadan PR merge edilmez (bkz. `git.md`, `testing.md`).
- Arayüz davranışı `streamlit.testing.v1.AppTest` ile gerçek `app.py` üzerinde sürülür; her arayüz
  kriterine ekran görüntüsü kanıtı eklenir.
