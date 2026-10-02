"""Q2 / Q4 (spec 0003): panel eylemi `decide_action` ile aynı kuraldan çıkar.

Q2: açık pozisyonda stop delinmişse sinyal ne derse desin TAMAMINI SAT.
Q4: zararlı çıkıştan sonra bekleme sürerken AL, SATIN AL olarak gösterilmez.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from trade_decisions import Position, PositionState, Signal, decide_action
from trading_ui import (
    CODE_TEXT, PanelInput, bars_since_loss_exit, build_decision_panel,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 10, tzinfo=UTC)
D = Decimal


def _input(**changes):
    values = dict(signal=Signal.WAIT, position=Position.flat(), data_status="GECERLI",
                  fee=D(0), spread_bps=D(0), slippage_bps=D(0), confidence=D(60), decision_at=NOW)
    values.update(changes)
    return PanelInput(**values)


def _open(stop="90"):
    return Position(PositionState.OPEN, D(2), D(100), NOW, D(stop))


# ---- Q2 ----------------------------------------------------------------------
@pytest.mark.parametrize("signal", [Signal.BUY, Signal.WAIT, Signal.SELL])
def test_q2_pierced_stop_means_sell_everything_whatever_the_signal(signal):
    panel = build_decision_panel(_input(signal=signal, position=_open("90"), current_price=D("89")))
    assert panel.action == "TAMAMINI SAT" and panel.exit_reason == "STOP"
    assert "STOP_TEMASI" in panel.messages


def test_q2_price_exactly_on_stop_counts_as_touched():
    panel = build_decision_panel(_input(signal=Signal.WAIT, position=_open("90"), current_price=D("90")))
    assert panel.action == "TAMAMINI SAT"


def test_q2_price_above_stop_keeps_holding_on_wait():
    panel = build_decision_panel(_input(signal=Signal.WAIT, position=_open("90"), current_price=D("95")))
    assert panel.action == "TUT" and "STOP_TEMASI" not in panel.messages


def test_q2_unknown_current_price_does_not_claim_the_position_is_safe():
    panel = build_decision_panel(_input(signal=Signal.WAIT, position=_open("90"), current_price=None))
    assert panel.action is None and "GUNCEL_RISK_DEGERLENDIRILEMIYOR" in panel.messages


def test_q2_unknown_price_still_allows_an_explicit_sell_signal():
    panel = build_decision_panel(_input(signal=Signal.SELL, position=_open("90"), current_price=None))
    assert panel.action == "TAMAMINI SAT"


def test_q2_panel_action_matches_decide_action_for_every_open_combination():
    names = {"SELL": "TAMAMINI SAT", "HOLD": "TUT"}
    for signal in Signal:
        for price in (D("85"), D("90"), D("120")):
            decision = decide_action(signal, _open("90"), stop_touched=price <= D("90"))
            panel = build_decision_panel(_input(signal=signal, position=_open("90"), current_price=price))
            assert panel.action == names[decision.action.name], (signal, price)


def test_q2_new_codes_have_turkish_text():
    for code in ("STOP_TEMASI", "BEKLEME"):
        assert code in CODE_TEXT


# ---- Q4 ----------------------------------------------------------------------
def test_q4_buy_during_loss_cooldown_is_not_shown_as_buy():
    panel = build_decision_panel(_input(signal=Signal.BUY, bars_since_loss_exit=1))
    assert panel.action is None and "BEKLEME" in panel.messages


def test_q4_buy_after_the_cooldown_is_a_buy_again():
    assert build_decision_panel(_input(signal=Signal.BUY, bars_since_loss_exit=2)).action == "SATIN AL"


def test_q4_buy_without_any_loss_exit_is_a_buy():
    assert build_decision_panel(_input(signal=Signal.BUY, bars_since_loss_exit=None)).action == "SATIN AL"


def test_q4_cooldown_does_not_touch_an_open_position():
    panel = build_decision_panel(_input(signal=Signal.BUY, position=_open(), current_price=D(100),
                                        bars_since_loss_exit=0))
    assert panel.action == "TUT"


def _closed(realized, exit_at, coin="BTC"):
    return {"Coin": coin, "Status": "CLOSED_CONFIRMED", "Realized": realized,
            "Çıkış Zamanı": exit_at.isoformat()}


def _days(first, count):
    return pd.date_range(first, periods=count, freq="D", tz="UTC")


def test_q4_counts_only_closed_daily_bars_after_the_exit_day():
    exit_at = datetime(2026, 9, 5, 15, tzinfo=UTC)
    index = _days("2026-09-01", 9)                          # 1..9 Eylül
    today = datetime(2026, 9, 9, 12, tzinfo=UTC)            # 9 Eylül mumu henüz kapanmadı
    assert bars_since_loss_exit([_closed(-5.0, exit_at)], "BTC", index, today) == 3   # 6, 7, 8


def test_q4_profitable_or_breakeven_exit_does_not_start_a_cooldown():
    exit_at = datetime(2026, 9, 5, 15, tzinfo=UTC)
    today = datetime(2026, 9, 6, 12, tzinfo=UTC)
    index = _days("2026-09-01", 6)
    assert bars_since_loss_exit([_closed(5.0, exit_at)], "BTC", index, today) is None
    assert bars_since_loss_exit([_closed(0.0, exit_at)], "BTC", index, today) is None


def test_q4_uses_the_latest_exit_and_ignores_other_assets():
    old = _closed(-9.0, datetime(2026, 9, 1, 10, tzinfo=UTC))
    latest_win = _closed(4.0, datetime(2026, 9, 4, 10, tzinfo=UTC))
    other = _closed(-9.0, datetime(2026, 9, 5, 10, tzinfo=UTC), coin="ETH")
    index, today = _days("2026-09-01", 8), datetime(2026, 9, 8, 12, tzinfo=UTC)
    assert bars_since_loss_exit([old, latest_win, other], "BTC", index, today) is None


def test_q4_no_closed_position_means_no_cooldown():
    assert bars_since_loss_exit([], "BTC", _days("2026-09-01", 3), NOW) is None


# ---- Uygulama düzeyi ---------------------------------------------------------
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


def _portfoy(stop, **ekstra):
    pos = {"Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Adet": 2.0, "Yatırım": 200.0, "Realized": 0.0,
           "Status": "ACTIVE", "Stop": stop, "V1Verified": True,
           "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00", "Tarih": "2026-01-02"}
    pos.update(ekstra)
    return {"balance": 500.0, "positions": [pos]}


def test_q2_app_open_position_with_pierced_stop_is_not_shown_as_hold(store, monkeypatch, processed_df):
    from app_helpers import make_app, texts
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    stop = float(processed_df["Close"].iloc[-1]) * 2          # fiyat stopun altında
    store.write_doc(store.PORTFOLIO_KEY, _portfoy(stop))
    at = _gunluk(make_app(monkeypatch, _taze(processed_df)).run())
    assert not at.exception
    metin = texts(at)
    assert "TAMAMINI SAT" in metin and "stop seviyesine temas" in metin
    assert "Pozisyonuna göre eylem: **TUT**" not in metin


def test_q2_stop_alert_reads_the_stop_key_the_ui_writes(monkeypatch):
    import data_fetchers
    from portfolio import check_active_positions_auto_close
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", lambda *a, **k: 85.0)
    portfolio = _portfoy(90.0)
    _, alerts = check_active_positions_auto_close(portfolio, {"Bitcoin (BTC)": "BTC-USD"})
    assert alerts and alerts[0]["stop"] == 90.0 and alerts[0]["type"] == "STOP_TEMASI_TEYIT_BEKLIYOR"


def test_q2_stop_alert_falls_back_to_legacy_sl_key(monkeypatch):
    import data_fetchers
    from portfolio import check_active_positions_auto_close
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", lambda *a, **k: 85.0)
    portfolio = _portfoy(None, SL=90.0)
    _, alerts = check_active_positions_auto_close(portfolio, {"Bitcoin (BTC)": "BTC-USD"})
    assert alerts and alerts[0]["stop"] == 90.0


def test_q4_app_buy_signal_right_after_a_loss_exit_shows_cooldown_not_buy(store, monkeypatch, processed_df):
    import signal_engine
    from app_helpers import make_app, texts
    from signal_engine import CompositeSignal

    def always_buy(df, *a, **k):
        sig = CompositeSignal(timeframe="1d", verdict="AL")
        sig.confidence = 70
        return sig

    monkeypatch.setattr(signal_engine, "generate_stable_signal", always_buy)
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    kapanis = (datetime.now(UTC) - timedelta(days=1)).replace(hour=10)
    kayit = {"Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Adet": 0.0, "Yatırım": 0.0, "Realized": -20.0,
             "Status": "CLOSED_CONFIRMED", "Çıkış Zamanı": kapanis.isoformat(), "Tarih": "2026-01-02"}
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 500.0, "positions": [kayit]})
    at = make_app(monkeypatch, _taze(processed_df)).run()
    at.session_state["flat_confirmed:Bitcoin (BTC)"] = True
    at = _gunluk(at)
    assert not at.exception
    metin = texts(at)
    assert "Pozisyonuna göre eylem: **SATIN AL**" not in metin
    assert "Zararlı çıkıştan sonra bekleme sürüyor" in metin
