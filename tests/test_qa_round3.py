"""QA turu 3 (Y1–Y10): her düzeltme geri alınınca düşen test."""
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

import data_fetchers
import trade_confirmation as tc
from app_helpers import make_app, texts
from conftest import make_ohlcv

UTC = timezone.utc
D = Decimal


def _taze(frame, scale=1.0):
    frame = frame.copy()
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = frame[col] * scale
    frame["Open"] = frame["Open"].clip(lower=frame["Low"], upper=frame["High"])
    dun = datetime.now(UTC).date() - timedelta(days=1)
    frame.index = pd.date_range(end=pd.Timestamp(dun), periods=len(frame), freq="D", tz="UTC")
    return frame


def _gunluk(at):
    for sb in at.selectbox:
        if sb.label == "Periyot:":
            return sb.set_value("1d").run()
    raise AssertionError("Periyot seçimi bulunamadı")


def _tikla(at, etiket):
    for b in at.button:
        if b.label == etiket:
            return b.click().run()
    raise AssertionError(etiket)


def _iki_varlik(monkeypatch, processed_df, store):
    btc, eth = _taze(processed_df), _taze(processed_df, scale=0.002)
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"})
    at = make_app(monkeypatch, btc)
    monkeypatch.setattr(data_fetchers, "get_market_data",
                        lambda src, sym, tf: ((eth if sym == "ETH-USD" else btc), "Binance"))
    return at, float(btc["Close"].iloc[-1]), float(eth["Close"].iloc[-1])


# Y1 -----------------------------------------------------------------------------
def test_y1_buy_form_follows_the_selected_asset_price_and_quantity(store, monkeypatch, processed_df):
    at, btc_price, eth_price = _iki_varlik(monkeypatch, processed_df, store)
    at = at.run()
    assert at.number_input(key="buy_price:BTC-USD").value == pytest.approx(btc_price)
    for sb in at.selectbox:
        if sb.label == "Enstrüman:":
            sb.set_value("Ethereum (ETH)").run()
    at.run()
    assert at.number_input(key="buy_price:ETH-USD").value == pytest.approx(eth_price)
    assert at.number_input(key="buy_qty:ETH-USD").value == pytest.approx(1000 / eth_price, rel=1e-6)


# Y2 -----------------------------------------------------------------------------
def test_y2_sell_price_follows_the_selected_position(store, monkeypatch, processed_df):
    at, btc_price, _ = _iki_varlik(monkeypatch, processed_df, store)
    pos = lambda coin, giris, adet: {
        "Coin": coin, "Giriş": giris, "Adet": adet, "Yatırım": giris * adet, "Realized": 0.0,
        "Status": "ACTIVE", "Stop": giris * 0.1, "V1Verified": True,
        "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00", "Tarih": "2026-01-02"}
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 0.0, "positions": [
        pos("Ethereum (ETH)", 90.0, 2.0), pos("Bitcoin (BTC)", 50000.0, 0.5)]})
    at = at.run()
    at.selectbox(key="sell_sel").set_value("Ethereum (ETH)").run()
    assert at.number_input(key="sell_price:Ethereum (ETH)").value == 90.0
    at.selectbox(key="sell_sel").set_value("Bitcoin (BTC)").run()
    assert at.number_input(key="sell_price:Bitcoin (BTC)").value == pytest.approx(btc_price)


# Y4 -----------------------------------------------------------------------------
def test_y4_paper_runs_the_real_engine_in_strict_mode(tmp_path, monkeypatch):
    """M12: strict_components kaldırılırsa eksik ML sessizce ağırlık dağıtımına döner."""
    import ml_models, paper_trading, signal_engine
    from config import FileConfig
    from data_fetchers import process_data

    raw = make_ohlcv(seed=11, segments=[(250, 0.004)])
    raw["Open"] = raw["Open"].clip(lower=raw["Low"], upper=raw["High"])
    raw.index = pd.date_range(end=pd.Timestamp(datetime.now(UTC).date() - timedelta(days=1)),
                              periods=len(raw), freq="D", tz="UTC")
    frame = process_data(raw, "test")[0]
    monkeypatch.setattr(FileConfig, "PAPER_FILE", str(tmp_path / "p.json"))
    monkeypatch.setattr(data_fetchers, "get_market_data", lambda *a, **k: (frame, "T"))
    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", lambda _df: None)
    signal_engine._stable_cache.clear(); signal_engine._bar_score_cache.clear()
    paper_trading.run_paper_update({"BTC": "BTC-USD"})
    journal = paper_trading._load_state()["journal"]
    assert journal[0]["verdict"] == "BILESEN_YOK" and journal[0]["unavailable_components"] == ["ml"]


def test_y4_app_start_reconciles_a_half_finished_trade(store, monkeypatch, processed_df):
    """M25: açılıştaki reconcile çağrısı kaldırılırsa yarım işlem günlükte hiç görünmez."""
    from position_journal import PositionJournal
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 0.0, "positions": [{
        "Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Adet": 2.0, "Yatırım": 200.0, "Realized": 0.0,
        "Status": "ACTIVE", "Stop": 10.0, "V1Verified": True, "JournalEventId": "yarim-1",
        "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00", "Tarih": "2026-01-02"}]})
    at = make_app(monkeypatch, _taze(processed_df)).run()
    assert not at.exception
    assert PositionJournal().has_event("yarim-1")


def test_y4_cost_settings_reach_the_paper_book_end_to_end(store, monkeypatch, processed_df, tmp_path):
    """M26: maliyet ayarı sanal deftere aktarılmazsa iki motor farklı varsayımla çalışır."""
    import paper_trading, signal_engine
    from config import FileConfig
    from signal_engine import CompositeSignal
    monkeypatch.setattr(FileConfig, "PAPER_FILE", str(tmp_path / "p.json"))
    monkeypatch.setattr(signal_engine, "generate_stable_signal",
                        lambda *a, **k: CompositeSignal(timeframe="1d", verdict="BEKLE"))
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, _taze(processed_df)).run()
    for ad, deger in (("commission_pct", "0.1"), ("spread_bps", "20"), ("slippage_bps", "5"),
                      ("quantity_step", "0.001"), ("notional", "2500"), ("capital", "30000")):
        at.text_input(key=f"ts:BTC-USD:{ad}").set_value(deger)
    at = at.run()
    at = _tikla(at, "📸 Bugünü Kaydet / Güncelle")
    assert not at.exception
    book = paper_trading._load_state()["assets"]["Bitcoin (BTC)"]
    assert book["costs"] == {"spread_bps": "20", "slippage_bps": "5", "commission_pct": "0.1"}
    assert (book["quantity_step"], book["trade_notional"], book["initial_balance"]) == ("0.001", "2500", "30000")


# Y5 -----------------------------------------------------------------------------
def test_y5_sell_quantity_is_compared_at_eight_decimals():
    position = {"Adet": 1000 / 45000, "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00"}
    ok = tc.validate_sell(position=position, quantity=D("0.02222222"), price=D(50000),
                          executed_at=datetime(2026, 1, 3, tzinfo=UTC), now=datetime(2026, 1, 4, tzinfo=UTC))
    assert ok is None
    # Spec 0006 Q01: 8 hane karşılaştırması korunur; eksik miktar kısmi satıştır, fazlası reddedilir.
    partial = tc.validate_sell(position=position, quantity=D("0.02222221"), price=D(50000),
                               executed_at=datetime(2026, 1, 3, tzinfo=UTC), now=datetime(2026, 1, 4, tzinfo=UTC))
    assert partial is None
    over = tc.validate_sell(position=position, quantity=D("0.02222223"), price=D(50000),
                            executed_at=datetime(2026, 1, 3, tzinfo=UTC), now=datetime(2026, 1, 4, tzinfo=UTC))
    assert over == "MIKTAR_FAZLA"


# Y6 -----------------------------------------------------------------------------
def test_y6_unreadable_settings_disable_backtest_and_paper(store, monkeypatch, processed_df):
    import trade_settings
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = _gunluk(make_app(monkeypatch, _taze(processed_df)).run())
    def kirik(_asset):
        raise store.StorageAccessError("test")
    monkeypatch.setattr(trade_settings, "load_raw", kirik)
    at = at.run()
    assert not at.exception
    assert [b for b in at.button if b.label == "🚀 Backtest Başlat"][0].disabled
    assert [b for b in at.button if b.label == "📸 Bugünü Kaydet / Güncelle"][0].disabled
    assert "varsayımları okunamadı" in texts(at)


# Y8 -----------------------------------------------------------------------------
def test_y8_buy_decision_with_unknown_atr_is_blocked_not_an_exception():
    from trade_execution import Bar, BookState, TradeSettings, advance_daily_bar
    bar = Bar(datetime(2026, 9, 2), D(100), D(105), D(95), D(102))
    result = advance_daily_bar(BookState(D(10000)), bar, "AL", None, TradeSettings(D(1000), D(1)))
    assert result.blocked == "GECERSIZ_STOP" and not result.opened


# Y10 ----------------------------------------------------------------------------
def test_y10_panel_messages_are_not_duplicated():
    from trade_decisions import Position, Signal
    from trading_ui import PanelInput, build_decision_panel
    panel = build_decision_panel(PanelInput(
        signal=Signal.WAIT, position=Position.flat(), data_status="V1_DOGRULANMADI", fee=D(0),
        spread_bps=D(0), slippage_bps=D(0), confidence=D(50), decision_at=None, in_scope=False))
    assert panel.messages.count("V1_DOGRULANMADI") == 1
