"""Q1/Q7/Q8 (spec 0003): geçmiş test ve sanal takip aynı veride aynı işlemi üretir.

İki gerçek motor çalıştırılır: `run_v1_strategy_backtest` ve
`advance_pending_daily_decision` (sanal takibin günlük adımı). Panel kararı da
aynı `decide_action` eşlemesine bağlanır.
"""
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

import paper_trading
from technical_analysis import run_v1_strategy_backtest
from trade_execution import CostAssumptions

D = Decimal


def _frame(rows):
    """rows: (open, high, low, close, atr)"""
    index = pd.date_range("2026-09-01", periods=len(rows), freq="D")
    return pd.DataFrame(
        [dict(Open=o, High=h, Low=l, Close=c, ATR=a) for o, h, l, c, a in rows], index=index)


def _paper(frame, decisions, *, cash, notional, step, costs=None):
    book = paper_trading._blank_book(cash)
    book["trade_notional"] = str(notional)
    book["quantity_step"] = None if step is None else str(step)
    if costs is not None:
        book["costs"] = {
            "spread_bps": None if costs.spread_bps is None else str(costs.spread_bps),
            "slippage_bps": None if costs.slippage_bps is None else str(costs.slippage_bps),
            "commission_pct": None if costs.commission_pct is None else str(costs.commission_pct),
        }
    for index in range(len(frame)):
        row = frame.iloc[index]
        day = str(frame.index[index].date())
        paper_trading.advance_pending_daily_decision(book, {
            "date": day, "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
        })
        book["pending"] = {"verdict": decisions.get(frame.index[index], "BEKLE"),
                           "atr": float(row["ATR"]), "known_at": day}
    return book


def _both(rows, verdicts, *, cash="10000", notional="1000", step="0.01", costs=None):
    frame = _frame(rows)
    decisions = {frame.index[i]: v for i, v in verdicts.items()}
    backtest = run_v1_strategy_backtest(
        frame, decisions, initial_cash=D(cash), trade_notional=D(notional),
        quantity_step=None if step is None else D(step), costs=costs)
    paper = _paper(frame, decisions, cash=cash, notional=notional, step=step, costs=costs)
    return backtest, paper


def _assert_same(backtest, paper):
    bt = [(t["entry"], t["exit"], t["quantity"], t["reason"], t["pnl"]) for t in backtest["trades"]]
    pp = [(D(t["entry"]), D(t["exit"]), D(t["qty"]), t["reason"], D(t["pnl"])) for t in paper["trades"]]
    assert [(D(a), D(b), D(c), r, D(p)) for a, b, c, r, p in bt] == pp
    assert D(backtest["cash"]) == D(paper["balance"])
    bt_open, pp_open = backtest["position"], paper["position"]
    assert (bt_open is None) == (pp_open is None)
    if bt_open is not None:
        assert (bt_open["entry"], bt_open["quantity"], bt_open["stop"]) == (
            D(pp_open["entry"]), D(pp_open["qty"]), D(pp_open["sl"]))


LOSS_ROWS = [  # AL -> alış 100; stop 90; 3. mumda açılış stopun altında -> zararlı çıkış
    (99, 101, 98, 100, 4),
    (100, 105, 95, 102, 4),
    (88, 95, 85, 90, 4),
    (100, 101, 99, 100, 4),
    (100, 101, 99, 100, 4),
    (100, 101, 99, 100, 4),
    (100, 105, 95, 102, 4),
    (100, 105, 95, 102, 4),
]


def test_q1_zero_cost_trade_is_identical_in_both_engines():
    rows = [(99, 101, 98, 100, 4), (100, 200, 95, 110, 5), (110, 115, 105, 112, 5), (120, 125, 118, 121, 5)]
    backtest, paper = _both(rows, {0: "AL", 2: "SAT"}, step="1", costs=CostAssumptions(D(0), D(0), D(0)))
    _assert_same(backtest, paper)
    assert backtest["trades"][0]["pnl"] == "200"        # 10 adet * (120 - 100)


def test_q1_hidden_fee_is_gone_unknown_commission_is_gross_and_unverified():
    rows = [(99, 101, 98, 100, 4), (100, 200, 95, 110, 5), (110, 115, 105, 112, 5), (120, 125, 118, 121, 5)]
    backtest, paper = _both(rows, {0: "AL", 2: "SAT"}, step="1")
    _assert_same(backtest, paper)
    assert backtest["trades"][0]["pnl"] == "200"        # brüt: 97,9 gibi sessiz kesinti yok
    assert backtest["net_verified"] is False
    assert paper["trades"][0]["net_verified"] is False


def test_q1_known_commission_is_charged_on_both_sides_identically():
    rows = [(99, 101, 98, 100, 4), (100, 200, 95, 110, 5), (110, 115, 105, 112, 5), (120, 125, 118, 121, 5)]
    costs = CostAssumptions(D(0), D(0), D("0.1"))
    backtest, paper = _both(rows, {0: "AL", 2: "SAT"}, step="1", costs=costs)
    _assert_same(backtest, paper)
    # 1000 alış + 1 komisyon, 1200 satış - 1,2 komisyon => net 197,8
    assert D(backtest["trades"][0]["pnl"]) == D("197.8")
    assert backtest["net_verified"] is True and paper["trades"][0]["net_verified"] is True
    assert D(backtest["cash"]) == D("10197.8")


def test_q1_spread_and_slippage_apply_identically_without_commission():
    rows = [(99, 101, 98, 100, 4), (100, 105, 95, 110, 5), (110, 115, 105, 112, 5), (120, 125, 118, 121, 5)]
    costs = CostAssumptions(D(100), D(25), None)
    backtest, paper = _both(rows, {0: "AL", 2: "SAT"}, step="0.01", costs=costs)
    _assert_same(backtest, paper)
    assert D(backtest["trades"][0]["entry"]) == D("100.75")     # AC111
    assert D(backtest["trades"][0]["exit"]) == D("119.25") or D(backtest["trades"][0]["exit"]) == D("120") * D("0.9925")
    assert backtest["net_verified"] is False


def test_q1_loss_cooldown_blocks_two_completed_bars_in_both_engines():
    verdicts = {0: "AL", 2: "AL", 3: "AL", 4: "AL", 5: "AL"}
    backtest, paper = _both(LOSS_ROWS, verdicts, step="1")
    _assert_same(backtest, paper)
    assert backtest["trades"][0]["reason"] == "STOP"
    assert backtest["position"]["entry_at"] == _frame(LOSS_ROWS).index[5]   # çıkıştan sonraki üçüncü açılış


def test_q1_known_commission_makes_flat_trade_a_loss_and_starts_cooldown_in_both():
    rows = [(100, 101, 99, 100, 4), (100, 105, 99, 100, 4), (100, 101, 99, 100, 4),
            (100, 105, 99, 100, 4), (100, 101, 99, 100, 4), (100, 101, 99, 100, 4)]
    verdicts = {0: "AL", 1: "SAT", 2: "AL", 3: "AL"}
    costs = CostAssumptions(D(0), D(0), D("0.1"))
    backtest, paper = _both(rows, verdicts, step="1", costs=costs)
    _assert_same(backtest, paper)
    assert D(backtest["trades"][0]["pnl"]) < 0 and len(backtest["trades"]) == 1
    assert backtest["position"] is None        # bekleme: 2. ve 3. karar uygulanmadı


def test_q8_unknown_quantity_step_buys_nothing_and_says_why_in_both_engines():
    rows = [(99, 101, 98, 100, 4), (100, 105, 95, 110, 5), (110, 115, 105, 112, 5)]
    backtest, paper = _both(rows, {0: "AL"}, step=None)
    assert backtest["total_trades"] == 0 and backtest["position"] is None
    assert backtest["blocked"] == {"MIKTAR_ADIMI_BILINMIYOR": 1}
    assert paper["position"] is None and paper["last_block"] == "MIKTAR_ADIMI_BILINMIYOR"
    assert D(backtest["cash"]) == D(paper["balance"]) == D("10000")


def test_q8_small_asset_quantity_is_floored_to_the_step_not_fractional():
    rows = [(270, 280, 265, 275, 8), (270, 280, 265, 275, 8), (270, 280, 265, 275, 8)]
    backtest, paper = _both(rows, {0: "AL"}, step="1", notional="1000")
    _assert_same(backtest, paper)
    assert backtest["position"]["quantity"] == D("3")           # 3,7037... değil


def test_q7_non_positive_initial_stop_blocks_entry_in_both_engines():
    rows = [(10, 11, 9, 10, 4), (10, 11, 9, 10, 4), (10, 11, 9, 10, 4)]      # 10 - 2,5*4 = 0
    backtest, paper = _both(rows, {0: "AL"}, step="1")
    assert backtest["position"] is None and paper["position"] is None
    assert backtest["blocked"] == {"GECERSIZ_STOP": 1}


def test_q7_entry_bar_stop_exits_once_at_stop_level_in_both_engines():
    rows = [(99, 101, 98, 100, 4), (100, 101, 89, 95, 4), (95, 96, 94, 95, 4)]
    backtest, paper = _both(rows, {0: "AL"}, step="1")
    _assert_same(backtest, paper)
    assert [t["reason"] for t in backtest["trades"]] == ["STOP"] and backtest["trades"][0]["exit"] == "90.0"


def test_q7_open_gap_below_stop_exits_at_open_with_single_sale_in_both_engines():
    rows = [(99, 101, 98, 100, 4), (100, 105, 95, 102, 4), (80, 85, 78, 82, 4), (82, 85, 80, 83, 4)]
    backtest, paper = _both(rows, {0: "AL", 1: "SAT"}, step="1")
    _assert_same(backtest, paper)
    assert len(backtest["trades"]) == 1 and backtest["trades"][0]["exit"] == "80"
    assert backtest["trades"][0]["reason"] == "STOP"          # AC108: SAT beklerken bile gerekçe stop


def test_q7_explicit_zero_costs_differ_from_unknown_only_in_verification():
    rows = [(99, 101, 98, 100, 4), (100, 200, 95, 110, 5), (110, 115, 105, 112, 5), (120, 125, 118, 121, 5)]
    zero, _ = _both(rows, {0: "AL", 2: "SAT"}, step="1", costs=CostAssumptions(D(0), D(0), D(0)))
    unknown, _ = _both(rows, {0: "AL", 2: "SAT"}, step="1")
    assert zero["trades"] == unknown["trades"]
    assert zero["net_verified"] is True and unknown["net_verified"] is False
    assert unknown["spread_known"] is False and zero["spread_known"] is True


def test_q7_cost_limit_rejects_entry_at_10000_bps_but_not_9999():
    from trade_execution import TradeSettings, enter_position

    def attempt(slippage):
        settings = TradeSettings(D(1000), D("0.01"), CostAssumptions(D(0), D(slippage), None))
        return enter_position(cash=D(10000), open_price=D(100), atr=D("0.001"), settings=settings)

    assert attempt(10000).reason == "GECERSIZ_MALIYET" and not attempt(10000).executed
    assert attempt(9999).executed                          # AC112


def test_q7_half_spread_plus_slippage_counts_toward_the_cost_limit():
    from trade_execution import fill_price

    assert fill_price(D(100), "BUY", CostAssumptions(D(2), D(9999), None)) is None      # 1 + 9999
    assert fill_price(D(100), "BUY", CostAssumptions(D(2), D(9998), None)) is not None
    assert fill_price(D(100), "BUY", CostAssumptions(D(-1), D(0), None)) is None
