"""Q14 (spec 0003): uygulama düzeyinde kabul akışları.

AC40/AC77: sinyal gerçek işlemi tek başına değiştirmez. Uçtan uca duman testi:
teyitli alış → panel TUT → SAT sinyali → TAMAMINI SAT → teyitli satış → pozisyon yok.
"""
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

import signal_engine
from app_helpers import make_app, texts
from position_journal import PositionJournal
from signal_engine import CompositeSignal

UTC = timezone.utc
VARLIK = {"Bitcoin (BTC)": "BTC-USD"}


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


def _sinyal(monkeypatch, verdict):
    def fake(df, *a, **k):
        sig = CompositeSignal(timeframe="1d", verdict=verdict)
        sig.confidence = 70
        return sig
    monkeypatch.setattr(signal_engine, "generate_stable_signal", fake)


def _tikla(at, etiket):
    for b in at.button:
        if b.label == etiket:
            return b.click().run()
    raise AssertionError(etiket)


def _acik_pozisyon(store, fiyat):
    store.write_doc(store.ASSETS_KEY, VARLIK)
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 500.0, "positions": [{
        "Coin": "Bitcoin (BTC)", "Giriş": fiyat, "Adet": 2.0, "Yatırım": 2 * fiyat, "Realized": 0.0,
        "Status": "ACTIVE", "Stop": fiyat * 0.1, "V1Verified": True,
        "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00", "Tarih": "2026-01-02",
        "JournalEventId": "seed-1"}]})
    journal = PositionJournal()
    journal.confirm_trade("seed-1", "BUY", 2, fiyat, datetime(2026, 1, 2, 10, tzinfo=UTC),
                          datetime(2026, 1, 2, 10, tzinfo=UTC), symbol="BTC-USD")


# AC40 — SAT sinyali gerçek pozisyonu kapatmaz.
def test_ac40_sell_signal_does_not_close_the_real_position(store, monkeypatch, processed_df):
    frame = _taze(processed_df)
    _acik_pozisyon(store, float(frame["Close"].iloc[-1]))
    _sinyal(monkeypatch, "SAT")
    before = store.read_doc(store.PORTFOLIO_KEY)

    at = _gunluk(make_app(monkeypatch, frame).run())

    assert not at.exception
    assert "Pozisyonuna göre eylem: **TAMAMINI SAT**" in texts(at)      # panel yönlendirir...
    assert store.read_doc(store.PORTFOLIO_KEY) == before                # ...ama kayıt değişmez
    assert PositionJournal().positions["BTC-USD"].quantity == 2


# AC77 — Teyit edilmeyen AL, gerçek işlem eklemez.
def test_ac77_buy_signal_without_confirmation_adds_no_real_trade(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIK)
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 500.0, "positions": []})
    _sinyal(monkeypatch, "AL")

    at = _gunluk(make_app(monkeypatch, _taze(processed_df)).run())

    assert not at.exception
    assert store.read_doc(store.PORTFOLIO_KEY) == {"balance": 500.0, "positions": []}
    assert PositionJournal().trades == []


def test_smoke_confirmed_buy_hold_sell_signal_exit_confirmed_sell_flat(store, monkeypatch, processed_df):
    frame = _taze(processed_df)
    fiyat = float(frame["Close"].iloc[-1])
    store.write_doc(store.ASSETS_KEY, VARLIK)
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1_000_000.0, "positions": []})
    _sinyal(monkeypatch, "BEKLE")

    # 1) Teyitli alış (dün 10:00 İstanbul).
    at = make_app(monkeypatch, frame).run()
    dun = date.today() - timedelta(days=1)
    at.number_input(key="buy_qty:BTC-USD").set_value(2.0)
    at.date_input(key="buy_date").set_value(dun)
    at.time_input(key="buy_time").set_value(time(10, 0))
    at = at.run()
    stop = [n for n in at.number_input if n.label == "Kuruma koyduğum stop"][0]
    stop.set_value(fiyat * 0.5).run()
    at = _tikla(at, "➕ Emri Gir / Ekle")
    assert not at.exception
    kayit = store.read_doc(store.PORTFOLIO_KEY)
    assert [p["Status"] for p in kayit["positions"]] == ["ACTIVE"]
    assert "BTC-USD" in PositionJournal().positions

    # 2) BEKLE iken panel: TUT.
    at = _gunluk(at)
    assert "Pozisyonuna göre eylem: **TUT**" in texts(at)

    # 3) SAT sinyali: panel TAMAMINI SAT der, kayıt yine değişmez.
    _sinyal(monkeypatch, "SAT")
    at = _gunluk(make_app(monkeypatch, frame).run())
    assert "Pozisyonuna göre eylem: **TAMAMINI SAT**" in texts(at)
    assert [p["Status"] for p in store.read_doc(store.PORTFOLIO_KEY)["positions"]] == ["ACTIVE"]

    # 4) Teyitli satış: pozisyon kapanır.
    at.date_input(key="sell_date").set_value(dun)
    at.time_input(key="sell_time").set_value(time(14, 0))
    at = at.run()
    at = _tikla(at, "Satışı Onayla")
    assert not at.exception
    assert [p["Status"] for p in store.read_doc(store.PORTFOLIO_KEY)["positions"]] == ["CLOSED_CONFIRMED"]
    assert "BTC-USD" not in PositionJournal().positions
