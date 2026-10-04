"""Spec 0005 Rev 5 — B6: kâr koruma adaylarının V1 referansıyla karşılaştırması (AC95, R13)."""
import copy
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

import profit_protection as pp
import protection_ui
from technical_analysis import run_v1_strategy_backtest
from trade_execution import CostAssumptions

D = Decimal
ZERO_COSTS = CostAssumptions(D("0"), D("0"), D("0"))


def _frame():
    index = pd.date_range("2026-09-01", periods=7, freq="D", tz="UTC")
    rows = [  # open, high, low, close
        (99, 101, 98, 100), (100, 105, 99, 104), (104, 112, 103, 110), (110, 121, 109, 120),
        (120, 121, 108, 109), (109, 111, 105, 108), (108, 108, 100, 100)]
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index).assign(ATR=4.0)


def _setup(costs=ZERO_COSTS):
    frame = _frame()
    decisions = {frame.index[0]: "AL", frame.index[5]: "SAT"}
    backtest = run_v1_strategy_backtest(frame, decisions, initial_cash=D("10000"), trade_notional=D("1000"),
                                        quantity_step=D("1"), costs=costs)
    return frame, decisions, backtest


def _by_rule(results):
    return {result.rule: result for result in results}


def test_ac95_each_candidate_is_simulated_on_the_same_trades_as_the_reference():
    """AC95 — Hedef, iz süren stop ve kademeli çıkış, V1 referansıyla aynı işlem üzerinde ayrı ayrı sonuç verir."""
    frame, decisions, backtest = _setup()
    assert [t["pnl"] for t in backtest["trades"]] == ["80"]          # V1: SAT ile 108'den çıkış
    results = _by_rule(pp.compare(frame, decisions, backtest, ZERO_COSTS, D("1")))
    assert set(results) == {pp.REFERENCE, pp.TARGET, pp.TRAILING, pp.SCALE_OUT}
    assert {rule: (r.closed_count, r.total_pnl) for rule, r in results.items()} == {
        pp.REFERENCE: (1, D("80")),     # 108'de SAT
        pp.TARGET: (1, D("200")),       # 2R = 120'de tamamı
        pp.TRAILING: (1, D("120")),     # en yüksek kapanış 120 − 2×4 = 112'de stop
        pp.SCALE_OUT: (1, D("90")),     # 1R'de yarısı 110 (+50) + kalan SAT 108 (+40)
    }
    assert results[pp.TARGET].expectancy == D("200")


def test_ac95_costs_are_applied_equally_to_every_rule():
    """AC95 — Maliyetler (komisyon, makas, kayma) tüm kurallara aynı biçimde uygulanır; fark yalnız kuraldan gelir."""
    costs = CostAssumptions(D("0"), D("0"), D("1"))                  # %1 komisyon
    frame, decisions, backtest = _setup(costs)
    results = _by_rule(pp.compare(frame, decisions, backtest, costs, D("1")))
    assert results[pp.REFERENCE].total_pnl == D(backtest["trades"][0]["pnl"])   # referans V1 motoruyla aynı
    assert results[pp.TARGET].total_pnl < D("200")


def test_ac95_comparison_never_touches_the_backtest_or_the_real_position():
    """AC95 — Karşılaştırma gerçek pozisyon kaydına ve backtest çıktısına yazmaz."""
    frame, decisions, backtest = _setup()
    before = copy.deepcopy(backtest)
    pp.compare(frame, decisions, backtest, ZERO_COSTS, D("1"))
    assert backtest == before


def test_ac95_view_lists_every_rule_and_labels_unknown_costs_as_upper_bound():
    """AC95 — Panel dört kuralı da listeler; maliyet bilinmiyorsa sonuçlar "üst sınır" etiketi taşır."""
    frame, decisions, backtest = _setup()
    known = " | ".join(protection_ui.build_protection_view(
        pp.compare(frame, decisions, backtest, ZERO_COSTS, D("1")), "USD", ZERO_COSTS).lines)
    for label in ("V1 referansı (sabit stop + SAT)", "Sabit hedef (2R)", "İz süren stop (2×ATR)",
                  "Kademeli çıkış (1R'de yarısı)"):
        assert label in known
    assert "üst sınır" not in known and "+200" in known.replace(" ", "")
    unknown_costs = CostAssumptions(None, None, D("0"))
    unknown = " | ".join(protection_ui.build_protection_view(
        pp.compare(frame, decisions, backtest, unknown_costs, D("1")), "USD", unknown_costs).lines)
    assert "(üst sınır)" in unknown
    assert "gerçek pozisyonunuzun stopunu ve miktarını değiştirmez" in known


def test_ac95_screen_shows_the_comparison_under_the_backtest(store, monkeypatch, processed_df):
    """AC95 — Backtest ekranında kâr koruma karşılaştırması görünür."""
    import technical_analysis
    from app_helpers import click, make_app, texts

    monkeypatch.setattr(technical_analysis, "build_v1_decisions",
                        lambda df, **k: {df.index[199]: "AL", df.index[230]: "SAT"})
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("0.001")
    at = click(at.run(), "🚀 Backtest Başlat")
    assert not at.exception
    shown = texts(at)
    assert "Kâr koruma karşılaştırması" in shown and "Sabit hedef (2R)" in shown
    assert "İz süren stop (2×ATR)" in shown
