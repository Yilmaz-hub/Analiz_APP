"""App-level integration tests for spec 0001 (mobil görünüm modu).

These drive the real app.py through streamlit.testing.v1.AppTest with the
network data source mocked, covering the criteria that only exist at the app
layer: the view-mode control (var/iki seçenek/required/kalıcılık) and the
boş-durum vs hata-durumu message split.
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import data_fetchers

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _make_app(monkeypatch, gmd):
    """Build an AppTest for app.py with the network sources mocked.

    app.py binds `get_market_data` via `from data_fetchers import ...` at run
    time, so patching the source attribute before .run() injects the mock.
    """
    monkeypatch.setattr(data_fetchers, "get_market_data", gmd)
    monkeypatch.setattr(data_fetchers, "get_fear_greed_index", lambda: (50, "Neutral"))
    return AppTest.from_file(APP_PATH, default_timeout=120)


def _all_texts(element_list):
    return " ".join(str(getattr(e, "value", "")) for e in element_list)


# C1 / BULGU-6 — Mod kontrolü var, iki seçenek sunuyor, varsayılan Masaüstü.
def test_view_mode_control_present_with_two_options(monkeypatch, processed_df):
    at = _make_app(monkeypatch, lambda *a, **k: (processed_df, "Binance")).run()
    assert not at.exception
    assert len(at.segmented_control) == 1
    sc = at.segmented_control[0]
    assert list(sc.options) == ["Masaüstü", "Mobil"]
    assert sc.value == "Masaüstü"


# BULGU-3 — Kontrol required: aktif segmente ikinci dokunuş modu düşüremez.
def test_view_mode_control_is_required(monkeypatch, processed_df):
    at = _make_app(monkeypatch, lambda *a, **k: (processed_df, "Binance")).run()
    assert at.segmented_control[0].proto.required is True


# C9 / BULGU-6 — Oturum kalıcılığı: Mobil seçilip başka bir kontrol
# kullanıldığında (rerun) mod Mobil kalır, varsayılana dönmez.
def test_view_mode_persists_across_unrelated_rerun(monkeypatch, processed_df):
    at = _make_app(monkeypatch, lambda *a, **k: (processed_df, "Binance")).run()
    at.segmented_control[0].set_value("Mobil").run()
    assert at.session_state["view_mode"] == "Mobil"
    # Alakasız bir kontrolü değiştir -> yeni rerun.
    at.checkbox[0].set_value(not at.checkbox[0].value).run()
    assert at.session_state["view_mode"] == "Mobil"


# C-Hata / BULGU-4 — Fetch hatası kırmızı hata mesajı gösterir (boş durum değil).
def test_error_state_shows_connection_message(monkeypatch, processed_df):
    at = _make_app(monkeypatch, lambda *a, **k: (None, "Veri Alınamadı")).run()
    assert not at.exception
    assert "Veri alınamadı" in _all_texts(at.error)
    assert "gösterilecek veri yok" not in _all_texts(at.info)


# C-Boş / BULGU-4 — Gerçekten veri yoksa mavi bilgi kutusu (hata değil).
def test_empty_state_shows_no_data_message(monkeypatch, processed_df):
    at = _make_app(monkeypatch, lambda *a, **k: (None, "Veri Yok (Yahoo)")).run()
    assert not at.exception
    assert "gösterilecek veri yok" in _all_texts(at.info)
    assert "Veri alınamadı" not in _all_texts(at.error)
