# Spec 0005 — Adım 4 ekran kanıtı

Ekran görüntüleri `harness.py` ile üretildi: gerçek `performance_ui.render_report_view` ve
`performance_report` / `evaluation_window` kodunu sabit veriyle çizer (canlı veri sağlayıcı
gerektirmez). Gerçek `app.py` akışı ayrıca `tests/test_performance_ui.py::test_smoke_backtest_screen_shows_reliable_performance_report`
ile sürülür.

| Dosya | Kanıtlanan |
|---|---|
| `missing_cost.png` | AC07, AC84 — maliyet eksik uyarısı ve "üst sınır" etiketi ana görünümde |
| `empty.png` | AC02, AC49, AC78 — boş dönem ve sıfır sermayede "hesaplanamıyor" |
| `error.png` | AC09, AC64 — geçersiz istek anlaşılır mesajla, teknik metin yok |
| `currencies.png` | AC10 — para birimleri ayrı bloklarda, tek toplam yok |
| `comparison.png` | AC47, AC62, AC88 — strateji ve al-tut değerlendirme diliminde, aynı anda başlar |

Yeniden üretim: `streamlit run docs/evidence/0005/adim4/harness.py` ve `?state=missing_cost|empty|error|currencies|comparison`.
