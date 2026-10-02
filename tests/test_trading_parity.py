"""Ekran, geçmiş test ve sanal takip tutarlılığı — gerçek motorlarla (spec 0003, Q10).

Üç yüzey de aynı kurallardan geçer: panel (`build_decision_panel`), geçmiş test
(`run_v1_strategy_backtest`) ve sanal takip (`advance_pending_daily_decision`).
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

import paper_trading
from technical_analysis import build_v1_decisions, run_v1_strategy_backtest
from trade_decisions import Action, Position, PositionState, Signal, classify_exit, decide_action, initial_stop
from trading_ui import PanelInput, build_decision_panel

UTC = timezone.utc
NOW = datetime(2026, 9, 10, tzinfo=UTC)
D = Decimal


def _frame(rows):
    index = pd.date_range("2026-09-01", periods=len(rows), freq="D")
    return pd.DataFrame([dict(Open=o, High=h, Low=l, Close=c, ATR=a) for o, h, l, c, a in rows], index=index)


def _engines(rows, verdicts, step="1"):
    frame = _frame(rows)
    decisions = {frame.index[i]: v for i, v in verdicts.items()}
    backtest = run_v1_strategy_backtest(frame, decisions, initial_cash=D(10000), trade_notional=D(1000),
                                        quantity_step=D(step))
    book = paper_trading._blank_book(10000)
    book["trade_notional"], book["quantity_step"] = "1000", step
    for i in range(len(frame)):
        row, day = frame.iloc[i], str(frame.index[i].date())
        paper_trading.advance_pending_daily_decision(book, {
            "date": day, "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"])})
        book["pending"] = {"verdict": decisions.get(frame.index[i], "BEKLE"), "atr": float(row["ATR"]),
                           "known_at": day}
    return backtest, book


ENTRY_ROWS = [(99, 101, 98, 100, 4), (100, 105, 95, 102, 4), (102, 105, 98, 103, 4)]


def test_requirement_decision_parity_panel_backtest_and_paper_agree_on_entry():
    for signal, verdict in ((Signal.BUY, "AL"), (Signal.WAIT, "BEKLE"), (Signal.SELL, "SAT")):
        panel = build_decision_panel(PanelInput(
            signal=signal, position=Position.flat(), data_status="GECERLI", fee=D(0), spread_bps=D(0),
            slippage_bps=D(0), confidence=D(60), decision_at=NOW))
        backtest, book = _engines(ENTRY_ROWS, {0: verdict})
        decided_buy = decide_action(signal, Position.flat()).action is Action.BUY
        assert (panel.action == "SATIN AL") == decided_buy
        assert (backtest["position"] is not None) == decided_buy
        assert (book["position"] is not None) == decided_buy


def test_requirement_stop_parity_ac83_ac84_entry_100_atr_4_gives_90_everywhere():
    assert initial_stop(D(100), D(4), later_atr=D(6)) == D("90.0")           # AC84
    backtest, book = _engines([(99, 101, 98, 100, 4), (100, 105, 95, 102, 6), (102, 105, 98, 103, 6)], {0: "AL"})
    assert backtest["position"]["stop"] == D("90")
    assert D(book["position"]["sl"]) == D("90")
    panel_stop = build_decision_panel(PanelInput(
        signal=Signal.WAIT, position=Position(PositionState.OPEN, D(10), D(100), NOW, D("90")),
        data_status="GECERLI", fee=D(0), spread_bps=D(0), slippage_bps=D(0), confidence=D(60),
        decision_at=NOW, current_price=D(101))).stop
    assert panel_stop == D("90")


def test_requirement_target_parity_no_engine_sells_at_a_profit_target():
    rows = [(99, 101, 98, 100, 4), (100, 300, 95, 250, 4), (250, 400, 240, 380, 4)]
    backtest, book = _engines(rows, {0: "AL"})
    assert backtest["trades"] == [] and backtest["position"] is not None
    assert book["trades"] == [] and book["position"] is not None


def test_ac14_net_parity_gross_12_fees_2_is_net_10_and_both_engines_agree_on_net():
    assert classify_exit(D(12), fee_known=True, fees=D(2)).net == D(10)
    from trade_execution import CostAssumptions
    rows = [(99, 101, 98, 100, 4), (100, 105, 95, 110, 5), (120, 125, 118, 121, 5)]
    frame = _frame(rows)
    costs = CostAssumptions(D(0), D(0), D("0.1"))
    backtest = run_v1_strategy_backtest(frame, {frame.index[0]: "AL", frame.index[1]: "SAT"},
                                        initial_cash=D(10000), trade_notional=D(1000),
                                        quantity_step=D(1), costs=costs)
    book = paper_trading._blank_book(10000)
    book.update(trade_notional="1000", quantity_step="1",
                costs={"spread_bps": "0", "slippage_bps": "0", "commission_pct": "0.1"})
    for i in range(len(frame)):
        row, day = frame.iloc[i], str(frame.index[i].date())
        paper_trading.advance_pending_daily_decision(book, {
            "date": day, "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"])})
        book["pending"] = {"verdict": {0: "AL", 1: "SAT"}.get(i, "BEKLE"), "atr": 4.0, "known_at": day}
    assert D(backtest["trades"][0]["pnl"]) == D(book["trades"][0]["pnl"])


def test_ac66_a_missing_component_makes_the_bar_non_comparable_not_a_neutral_buy(processed_df, monkeypatch):
    import ml_models
    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", lambda _df: None)
    decisions = build_v1_decisions(processed_df.iloc[:260], include_ml=True)
    assert set(decisions.values()) == {"BILESEN_YOK"}
    panel = build_decision_panel(PanelInput(
        signal=Signal.BUY, position=Position.flat(), data_status="GECERLI", fee=D(0), spread_bps=D(0),
        slippage_bps=D(0), confidence=D(60), decision_at=NOW, missing_component="BILESEN_HAZIR_DEGIL"))
    assert not panel.comparable and panel.action is None


def test_ac67_internal_precision_is_not_rounded_to_cents():
    rows = [(99, 101, 98, 100, 4), (100.0033, 105, 95, 102, 4), (100.0066, 105, 95, 103, 4)]
    frame = _frame(rows)
    result = run_v1_strategy_backtest(frame, {frame.index[0]: "AL", frame.index[1]: "SAT"},
                                      initial_cash=D(10000), trade_notional=D(1000), quantity_step=D("0.00000001"))
    pnl = D(result["trades"][0]["pnl"])
    assert pnl != pnl.quantize(D("0.01"))


def test_ac91_ac92_paper_books_use_their_own_amounts_and_do_not_share_cash(store, monkeypatch, tmp_path):
    from config import FileConfig
    from signal_engine import CompositeSignal
    import data_fetchers, signal_engine
    from conftest import make_ohlcv

    monkeypatch.setattr(FileConfig, "PAPER_FILE", str(tmp_path / "paper.json"))

    def frame_for(*_a, **_k):
        from data_fetchers import process_data
        raw = make_ohlcv(seed=3, segments=[(250, 0.003)])
        raw["Open"] = raw["Open"].clip(lower=raw["Low"], upper=raw["High"])
        yesterday = datetime.now(UTC).date() - timedelta(days=1)
        raw.index = pd.date_range(end=pd.Timestamp(yesterday), periods=len(raw), freq="D", tz="UTC")
        return process_data(raw, "test")[0], "TEST"

    monkeypatch.setattr(data_fetchers, "get_market_data", frame_for)
    monkeypatch.setattr(signal_engine, "generate_stable_signal",
                        lambda *a, **k: CompositeSignal(timeframe="1d", verdict="BEKLE"))
    paper_trading.run_paper_update(
        {"ETH": "ETH-USD", "SOL": "SOL-USD"},
        paper_settings={"ETH": {"capital": "25000", "trade_notional": "2500"},
                        "SOL": {"capital": "5000", "trade_notional": "500"}})
    state = paper_trading._load_state()
    assert (state["assets"]["ETH"]["initial_balance"], state["assets"]["ETH"]["trade_notional"]) == ("25000", "2500")
    assert (state["assets"]["SOL"]["initial_balance"], state["assets"]["SOL"]["trade_notional"]) == ("5000", "500")
    assert state["assets"]["ETH"]["balance"] == "25000" and state["assets"]["SOL"]["balance"] == "5000"


def test_ac121_decisions_use_only_data_known_at_the_decision_time(processed_df, monkeypatch):
    import ml_models

    seen = []

    def spy(df):
        seen.append(df.index[-1])
        return None

    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", spy)
    frame = processed_df.iloc[:230]
    build_v1_decisions(frame, include_ml=True)
    # ML, her kararda yalnız o ana kadarki mumları görür: ileri tarihli satır sızmaz.
    assert seen == list(frame.index[199:])
