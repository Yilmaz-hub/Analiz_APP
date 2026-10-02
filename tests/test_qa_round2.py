"""QA turu 2 (rapor 62c3609): B3/B4/B6 şimdiki head'de uygulama düzeyinde sınanır."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from app_helpers import make_app, texts

UTC = timezone.utc


def _taze(frame):
    frame = frame.copy()
    frame["Open"] = frame["Open"].clip(lower=frame["Low"], upper=frame["High"])
    dun = datetime.now(UTC).date() - timedelta(days=1)
    frame.index = pd.date_range(end=pd.Timestamp(dun), periods=len(frame), freq="D", tz="UTC")
    return frame


def _gunluk(at):
    for sb in at.selectbox:
        if sb.label == "Periyot:":
            return sb.set_value("1d").run()
    raise AssertionError("Periyot seçimi bulunamadı")


def _acik(store, fiyat):
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 500.0, "positions": [{
        "Coin": "Bitcoin (BTC)", "Giriş": fiyat, "Adet": 2.0, "Yatırım": 2 * fiyat, "Realized": 0.0,
        "Status": "ACTIVE", "Stop": fiyat * 0.1, "V1Verified": True,
        "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00", "Tarih": "2026-01-02"}]})


def test_b3_missing_ml_with_open_position_shows_no_personal_action(store, monkeypatch, processed_df):
    import ml_models
    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", lambda _df: None)
    frame = _taze(processed_df)
    _acik(store, float(frame["Close"].iloc[-1]))
    at = _gunluk(make_app(monkeypatch, frame).run())
    metin = texts(at)
    assert not at.exception
    assert "Pozisyonuna göre eylem:" not in metin
    assert "Güncel risk değerlendirilemiyor" in metin
    assert "yapay zekâ" in metin


def test_b4_zero_volume_names_the_component_everywhere_and_panel_gives_no_action(store, monkeypatch, processed_df):
    frame = _taze(processed_df)
    frame["Volume"] = 0.0
    _acik(store, float(frame["Close"].iloc[-1]))
    at = _gunluk(make_app(monkeypatch, frame).run())
    metin = texts(at)
    assert not at.exception
    assert "Karar bileşeni hazır değil: hacim" in metin
    assert "Pozisyonuna göre eylem:" not in metin
    assert "Günlük veri doğrulanamadı" not in metin


def test_b6_reason_lines_show_turkish_component_names_not_codes(store, monkeypatch, processed_df):
    import ml_models
    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", lambda _df: None)
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = _gunluk(make_app(monkeypatch, _taze(processed_df)).run())
    metin = texts(at)
    assert "Zorunlu karar bileşeni hesaplanamadı" not in metin and "BILESEN_HAZIR_DEGIL" not in metin
