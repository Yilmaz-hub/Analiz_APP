# Arayüz Konvansiyonları (Streamlit)

> Yığın: **Streamlit** + **Plotly**. Giriş noktası `app.py`; tekrar kullanılan parçalar
> `ui_components.py`, işlem paneli `trading_ui.py`, tema/CSS `theme.py`.
> Çalıştırma: `streamlit run app.py`.

## Temel Kurallar

- **Arayüz hesap yapmaz.** İş kuralları ve finansal hesaplar ilgili modülde yazılır; arayüz
  çağırır ve gösterir (bkz. `conventions.md` → Modül Yapısı). Panel verisi saf fonksiyonlarla
  hazırlanır (ör. `trading_ui.build_decision_panel`) ki AppTest olmadan da test edilebilsin.
- **Her ekranda üç durum:** **yükleme** (`st.spinner`), **boş** (ör. "İşlem sinyali yok",
  `render_no_records_state`), **hata** (`st.error` / `st.warning`). Üçü de test edilir.
- **Kullanıcıya teknik hata metni sızmaz.** Traceback / ham exception mesajı gösterilmez;
  anlaşılır Türkçe mesaj verilir, ayrıntı log'a gider (bkz. `conventions.md` → Hata Yönetimi).
- **Kayıt erişimi yoksa** kayıt değiştiren tüm kontroller gizlenir ve kullanıcı bilgilendirilir
  (bkz. `storage.md` → Erişim sorunu).
- Kullanıcıya görünen metinler Türkçedir.
- **Dilimler mini-spec'le koşar.** Her ekran değişikliği `specs/` altındaki bir spec'e bağlıdır.

## Durum ve Önbellek

- Rerun'lar arası korunması gereken seçimler widget `key`'i ya da `st.session_state` ile
  tutulur (ör. `key="view_mode"`).
- Ağdan gelen veri `@st.cache_data(ttl=…)` ile önbelleğe alınır (`data_fetchers.py`); TTL
  değerleri `config.py`'dedir.

## Görünüm Modu ve Grafik

- **Masaüstü / Mobil** görünüm modu `st.segmented_control` ile seçilir; varsayılan Masaüstü
  (spec 0001). Karar `ui_components.is_mobile_mode` üzerinden verilir.
- Grafik yüksekliği, yakınlaştırma ve legend yerleşimi moda göre `ui_components`
  yardımcılarıyla belirlenir (`chart_height`, `resolve_zoom_count`); masaüstü davranışı
  değiştirilirken regresyon testi eklenir (`tests/test_chart_layout.py`).
- Grafik çizilemiyorsa (`is_chart_renderable`) boş grafik yerine açıklayıcı mesaj gösterilir.

## Para / Yüzde

- `Decimal` değerler arayüze kadar `Decimal` taşınır; biçimleme tek yardımcıdan geçer
  (ör. `trading_ui._display_decimal`). Gösterim için yuvarlanan değer hesaba geri girmez.
- Kullanıcı girdisi (komisyon, gecikme) arayüzde değil doğrulayıcıda denetlenir
  (`trading_ui.validate_fee`, `validate_delay`).

## Kalite Kapıları (CI)

- `pytest` yeşil olmadan PR merge edilmez (bkz. `git.md`, `testing.md`).
- Her arayüz kriterine ekran görüntüsü + kritik akışa smoke test (bkz. `testing.md`).
