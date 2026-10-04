"""Spec 0005 Rev 5 — QA bulguları: adaylar ve maliyet (B7, B10, B11, B12, B13)."""
import threading
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

import candidate_ui
import strategy_candidates as sc
from regime_classifier import YUKSELEN
from trade_execution import CostAssumptions

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
BREAKOUT, PULLBACK = sc.default_candidates("KRIPTO", date(2025, 1, 1), date(2026, 1, 1))
REF = sc.Metrics(Decimal("10"), Decimal("20"), Decimal("5"), 40)


def test_ac101_concurrent_history_writes_lose_nothing():
    """AC101 — Aday geçmişine iki yazıcı aynı anda yazdığında hiçbir kayıt kaybolmaz."""
    sc.preregister(BREAKOUT, NOW)
    barrier = threading.Barrier(2)
    errors = []

    def writer(label):
        try:
            barrier.wait()
            for index in range(25):
                sc.record_result(BREAKOUT, f"{label}{index}", sc.Verdict(sc.OLCUTU_KARSILADI), NOW)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(name,)) for name in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(sc.history()) == 50


def test_ac101_concurrent_preregistrations_keep_every_candidate():
    """AC101 — Eşzamanlı ön kayıtlar birbirinin kaydını silmez."""
    variants = [replace(BREAKOUT, settings={"lookback": str(10 + n)}) for n in range(12)]
    barrier = threading.Barrier(3)

    def writer(chunk):
        barrier.wait()
        for candidate in chunk:
            sc.preregister(candidate, NOW)

    threads = [threading.Thread(target=writer, args=(variants[i::3],)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(sc.start_run(candidate).ok for candidate in variants)


def _frame(n=60):
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    close = np.concatenate([np.linspace(100, 110, 30), np.linspace(110, 108, 10), np.linspace(108, 130, 20)])
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close}, index=index)


def test_ac102_candidate_settings_drive_the_decisions():
    """AC102 — Adayın `settings` değerleri kararı belirler; ayar değişince karar da değişir."""
    frame = _frame()
    regimes = pd.Series([YUKSELEN] * len(frame), index=frame.index, dtype=object)
    short = replace(BREAKOUT, settings={"lookback": "5"})
    long = replace(BREAKOUT, settings={"lookback": "35"})
    short_entries = [d for d, v in sc.candidate_decisions(frame, {}, short, regimes).items() if v == "AL"]
    long_entries = [d for d, v in sc.candidate_decisions(frame, {}, long, regimes).items() if v == "AL"]
    assert short_entries and short_entries != long_entries
    tolerant = replace(PULLBACK, settings={"ema": "5", "tolerance_pct": "10"})
    strict = replace(PULLBACK, settings={"ema": "5", "tolerance_pct": "0.01"})
    assert (sc.candidate_decisions(frame, {}, tolerant, regimes)
            != sc.candidate_decisions(frame, {}, strict, regimes))


def test_ac96_unknown_cost_candidate_cannot_be_declared_better():
    """AC96 — Maliyeti bilinmeyen sonuçla aday ölçütü "doğrulanamadı" (yetersiz veri) çıkar, "karşıladı" çıkmaz."""
    better = replace(REF, net_return_pct=Decimal("50"))
    assert sc.judge(better, REF).status == sc.OLCUTU_KARSILADI
    unknown = sc.judge(replace(better, costs_known=False), replace(REF, costs_known=False))
    assert unknown.status == sc.YETERSIZ_VERI and "üst sınır" in unknown.reason
    assert sc.judge(better, replace(REF, costs_known=False)).status == sc.YETERSIZ_VERI


def _backtest(**flags):
    return {"daily_equity": [{"date": date(2026, 1, 1), "equity": "10000"},
                             {"date": date(2026, 1, 2), "equity": "10100"}],
            "trades": [{"entry_at": date(2026, 1, 1), "exit_at": date(2026, 1, 2), "pnl": "100",
                        "entry": "100", "exit": "101", "quantity": "1", "reason": "SAT"}],
            "position": None, "initial_cash": "10000", **flags}


def test_ac96_report_is_upper_bound_when_spread_or_slippage_is_unknown():
    """AC96 — Komisyon biliniyor ama makas ya da kayma bilinmiyorsa rapor "üst sınır" olur; üçü de biliniyorsa etiket yoktur."""
    import performance_ui

    def usd(**flags):
        return performance_ui.report_from_backtest("ETH-USD", _backtest(**flags)).report.results["USD"]

    assert usd(net_verified=True, spread_known=True, slippage_known=True).is_upper_bound is False
    assert usd(net_verified=True, spread_known=False, slippage_known=True).is_upper_bound is True
    assert usd(net_verified=True, spread_known=True, slippage_known=False).is_upper_bound is True
    assert usd(net_verified=False, spread_known=True, slippage_known=True).is_upper_bound is True


def test_ac96_comparison_labels_strategy_and_hold_when_any_cost_is_unknown(trending_df):
    """AC96 — Karşılaştırma bloğunda strateji ve al-tut sonucu, maliyetlerden biri bilinmiyorsa "üst sınır" etiketi taşır."""
    import performance_ui

    index = trending_df.index
    decisions = {index[260]: "AL", index[290]: "SAT"}

    def labels(costs):
        view = performance_ui.comparison_from_backtest(
            "ETH-USD", trending_df, decisions, Decimal("1000"), Decimal("10000"), Decimal("0.001"), costs)
        return {label: value for label, value in view.blocks[0].metrics}

    unknown = labels(CostAssumptions(Decimal("0.1"), None, None))
    assert unknown["Strateji son sermaye"].endswith("(üst sınır)")
    assert unknown["Al-tut son sermaye"].endswith("(üst sınır)")
    known = labels(CostAssumptions(Decimal("5"), Decimal("2"), Decimal("0.1")))
    assert "(üst sınır)" not in known["Strateji son sermaye"] + known["Al-tut son sermaye"]


def test_ac97_evaluation_computes_entry_rules_on_the_full_history(monkeypatch, trending_df):
    """AC97 — Aday değerlendirmesi giriş kuralını tam geçmişte hesaplar (dilim yalnız sonuç ölçümünü sınırlar); koşucuyla aynı karar."""
    seen = []
    real = sc.candidate_decisions

    def spy(frame, v1, candidate, regimes, **kwargs):
        seen.append(len(frame))
        return real(frame, v1, candidate, regimes, **kwargs)

    monkeypatch.setattr(sc, "candidate_decisions", spy)
    outcome = candidate_ui.evaluate_candidates(
        "BTC-USD", trending_df, {trending_df.index[260]: "AL"}, notional=Decimal("1000"),
        capital=Decimal("10000"), quantity_step=Decimal("0.001"),
        costs=CostAssumptions(Decimal("1"), Decimal("1"), Decimal("0.1")), now=NOW)
    assert outcome.ok and seen == [len(trending_df)] * 2


def test_ac96_evaluation_marks_candidates_unverified_when_costs_unknown(trending_df):
    """AC96 — Maliyetleri bilinmeyen varlıkta aday değerlendirmesi "yetersiz veri" olur ve "üst sınır" nedenini yazar."""
    outcome = candidate_ui.evaluate_candidates(
        "BTC-USD", trending_df, {trending_df.index[260]: "AL", trending_df.index[280]: "SAT"},
        notional=Decimal("1000"), capital=Decimal("10000"), quantity_step=Decimal("0.001"),
        costs=None, now=NOW)
    assert outcome.ok
    assert {entry["status"] for entry in sc.history()} == {sc.YETERSIZ_VERI}
    assert all("üst sınır" in entry["reason"] for entry in sc.history())


def test_ac98_user_selection_is_stable_across_evaluation_periods_and_reversible():
    """AC98 — Aktif aday seçimi kalıcıdır, değerlendirme dönemi değişse de aynı sürümdür ve V1'e dönülebilir."""
    moved = replace(BREAKOUT, period_start=date(2026, 6, 1), period_end=date(2026, 9, 1))
    assert sc.fingerprint(moved) != sc.fingerprint(BREAKOUT)                 # ön kayıt dönemi içerir
    assert sc.strategy_fingerprint(moved) == sc.strategy_fingerprint(BREAKOUT)  # sürüm içermez
    assert sc.strategy_fingerprint(replace(BREAKOUT, settings={"lookback": "30"})) != \
        sc.strategy_fingerprint(BREAKOUT)
    assert sc.active_strategy() == "V1"
    sc.preregister(BREAKOUT, NOW)
    sc.choose_active_strategy(BREAKOUT)
    assert sc.active_strategy() == sc.strategy_fingerprint(moved)
    assert sc.find_by_strategy(sc.active_strategy()).name == BREAKOUT.name
    sc.choose_active_strategy(None)
    assert sc.active_strategy() == "V1"


def test_ac98_panel_buttons_select_and_revert_the_active_strategy(store, monkeypatch, processed_df):
    """AC98 — Panelde "Aktif yap" adayı seçer, "V1'e dön" geri alır; seçilen aday adıyla görünür."""
    import technical_analysis
    from app_helpers import click, make_app, texts

    monkeypatch.setattr(technical_analysis, "build_v1_decisions",
                        lambda df, **k: {df.index[199]: "AL", df.index[210]: "SAT"})
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("0.001")
    at = click(at.run(), "🚀 Backtest Başlat")
    assert "Aktif strateji: V1 (mevcut strateji)" in texts(at)
    at = click(at, "Aktif yap: Kırılım (20 gün)")
    assert not at.exception
    assert "Aktif strateji: Kırılım (20 gün) (kullanıcı seçimi)" in texts(at)
    assert sc.active_strategy() != "V1"
    at = click(at, "V1'e dön")
    assert "Aktif strateji: V1 (mevcut strateji)" in texts(at) and sc.active_strategy() == "V1"
