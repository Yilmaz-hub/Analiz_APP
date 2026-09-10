from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd


def test_okx_daily_request_uses_utc_day(monkeypatch):
    import data_fetchers

    captured = {}

    class Response:
        def json(self):
            return {"code": "1", "data": []}

    def fake_get(url, params, headers, timeout):
        captured.update(params)
        return Response()

    data_fetchers.fetch_okx_simple.clear()
    monkeypatch.setattr(data_fetchers.requests, "get", fake_get)
    data_fetchers.fetch_okx_simple("ETH-USD", "1d", 200)

    assert captured["bar"] == "1Dutc"


def test_okx_open_candle_marker_survives_fetch(monkeypatch):
    import data_fetchers

    class Response:
        def json(self):
            return {"code": "0", "data": [
                ["1788998400000", "100", "102", "99", "101", "10", "20", "30", "0"],
                ["1788912000000", "99", "101", "98", "100", "11", "21", "31", "1"],
            ]}

    data_fetchers.fetch_okx_simple.clear()
    monkeypatch.setattr(data_fetchers.requests, "get", lambda *args, **kwargs: Response())

    frame = data_fetchers.fetch_okx_simple("ETH-USD", "1d", 200)

    assert frame.attrs["provider_open"] == {pd.Timestamp("2026-09-10")}


def test_portfolio_stop_touch_is_alert_only(monkeypatch):
    import data_fetchers
    from portfolio import check_active_positions_auto_close

    portfolio = {"balance": 0, "positions": [{
        "Coin": "ETH", "Status": "ACTIVE", "SL": 90, "TP": 120,
        "Giris": 100, "Miktar": 1, "Yatırım": 100,
    }]}
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", lambda *args: 80)

    count, alerts = check_active_positions_auto_close(portfolio, {"ETH": "ETH-USD"})

    assert count == 0 and portfolio["positions"][0]["Status"] == "ACTIVE"
    assert alerts[0]["type"] == "STOP_TEMASI_TEYIT_BEKLIYOR"


def test_paper_v1_holds_until_stop_or_sell():
    from paper_trading import advance_v1_book
    from trade_execution import Bar

    at = datetime(2026, 9, 10, tzinfo=timezone.utc)
    book = {"cash": "9000", "position": {"quantity": "10", "entry": "100", "stop": "90"}, "trades": []}
    rising = Bar(at, Decimal("105"), Decimal("130"), Decimal("100"), Decimal("125"))
    held = advance_v1_book(book, "BEKLE", rising)
    assert held["position"]["stop"] == "90" and held["trades"] == []

    sold = advance_v1_book(held, "SAT", Bar(at + timedelta(days=1), Decimal("123"), Decimal("125"), Decimal("120"), Decimal("121")))
    assert sold["position"] is None and sold["trades"][0]["reason"] == "SAT"


def test_existing_paper_update_keeps_stop_fixed_and_ignores_target():
    from paper_trading import _step_book

    book = {"balance": 9000.0, "position": {
        "entry": 100.0, "entry_date": "2026-09-01", "qty": 10.0,
        "cost": 1000.0, "highest": 100.0, "sl": 90.0, "tp": 110.0,
    }, "cooldown": 0, "trades": []}

    _step_book(book, "BEKLE", 125.0, 4.0, "2026-09-10")
    assert book["position"]["sl"] == 90.0 and book["trades"] == []

    _step_book(book, "SAT", 123.0, 4.0, "2026-09-11")
    assert book["position"] is None and book["trades"][0]["reason"] == "SAT"


def test_paper_signal_executes_at_next_open_and_uses_decision_atr():
    from paper_trading import advance_pending_daily_decision

    book = {
        "balance": 10000.0, "position": None, "cooldown": 0, "trades": [],
        "quantity_step": 0.01,
        "pending": {"verdict": "AL", "atr": 4.0, "known_at": "2026-09-09"},
    }
    bar = {"date": "2026-09-10", "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0}

    advance_pending_daily_decision(book, bar)

    assert book["position"]["entry"] == 100.0
    assert book["position"]["sl"] == 90.0
    assert book["position"]["entry_date"] == "2026-09-10"


def test_paper_report_keeps_quote_currency_visible(monkeypatch):
    import paper_trading

    state = {
        "created": "2026-09-10", "journal": [
            {"asset": "THYAO", "price": 300}, {"asset": "ETH", "price": 2000},
        ],
        "assets": {
            "THYAO": {**paper_trading._blank_book(10000), "currency": "TRY", "first_price": 300},
            "ETH": {**paper_trading._blank_book(10000), "currency": "USD/USDT", "first_price": 2000},
        },
    }
    monkeypatch.setattr(paper_trading, "_load_state", lambda: state)

    report, _ = paper_trading.paper_report()

    assert set(report["Para Birimi"]) == {"TRY", "USD/USDT"}
    assert "Bakiye ($)" not in report.columns


def test_validated_signal_uses_all_closed_rows():
    from market_validation import MarketPolicy
    from signal_engine import generate_validated_signal

    index = pd.date_range(end="2026-09-09", periods=200, freq="D", tz="UTC")
    frame = pd.DataFrame({"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1000, "Source": "OKX"}, index=index)
    policy = MarketPolicy("CRYPTO", datetime(2026, 9, 9, tzinfo=timezone.utc), timedelta(minutes=5), 200)
    seen = {}

    def signal_factory(data, **kwargs):
        seen["rows"] = len(data)
        return "AL"

    result = generate_validated_signal(frame, datetime(2026, 9, 10, tzinfo=timezone.utc), policy, signal_factory=signal_factory)

    assert result == "AL" and seen["rows"] == 200


def test_v1_backtest_uses_next_open_fixed_stop_and_ignores_target():
    from technical_analysis import run_v1_strategy_backtest

    index = pd.date_range("2026-09-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "Open": [99, 100, 110, 120], "High": [101, 200, 115, 125],
        "Low": [98, 95, 105, 118], "Close": [100, 110, 112, 121],
        "ATR": [4, 5, 5, 5],
    }, index=index)
    decisions = {index[0]: "AL", index[1]: "BEKLE", index[2]: "SAT"}

    result = run_v1_strategy_backtest(frame, decisions, initial_cash=Decimal("1000"), trade_notional=Decimal("1000"))

    assert result["trades"] == [{
        "entry_at": index[1], "entry": "100", "exit_at": index[3], "exit": "120",
        "quantity": "10", "reason": "SAT", "pnl": "200",
    }]


def test_v1_backtest_waits_two_completed_bars_after_loss():
    from technical_analysis import run_v1_strategy_backtest

    index = pd.date_range("2026-09-01", periods=6, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "Open": [100, 100, 100, 100, 100, 100],
        "High": [101, 105, 101, 101, 101, 105],
        "Low": [99, 95, 89, 95, 95, 95],
        "Close": [100, 102, 90, 100, 100, 102],
        "ATR": [4, 4, 4, 4, 4, 4],
    }, index=index)
    decisions = {index[0]: "AL", index[2]: "AL", index[3]: "AL", index[4]: "AL"}

    result = run_v1_strategy_backtest(
        frame, decisions, initial_cash=Decimal("2000"), trade_notional=Decimal("1000")
    )

    assert result["trades"][0]["reason"] == "STOP"
    assert result["position"]["entry_at"] == index[5]
