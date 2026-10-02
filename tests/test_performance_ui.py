"""Spec 0005 Adım 4 — rapor arayüzü (AC07, AC09, AC64, AC78, AC84 görünürlüğü)."""
from datetime import date, timedelta
from decimal import Decimal

from streamlit.testing.v1 import AppTest

import performance_ui
from performance_report import EquityPoint, ReportFilters, ReportTrade, build_report

D0 = date(2026, 1, 1)


def _equity(values):
    return [EquityPoint(D0 + timedelta(days=i), Decimal(str(v))) for i, v in enumerate(values)]


def _trade(pnl, cost_known=True):
    return ReportTrade("ETH-USD", "KRIPTO", "USD", Decimal(str(pnl)), D0 + timedelta(days=1),
                       "V1", cost_known)


def _view(equity, trades, filters=None):
    return performance_ui.build_report_view(
        build_report({"ETH-USD": equity}, trades, filters or ReportFilters()))


def _all_text(view):
    parts = list(view.messages)
    for block in view.blocks:
        parts.append(block.title)
        parts.extend(f"{label} {value}" for label, value in block.metrics)
        parts.extend(block.notes)
    return " | ".join(parts)


def test_ac07_missing_cost_note_is_in_main_view():
    """AC07 — Komisyonu bilinmeyen işlemde "maliyet eksik" açıklaması ek tıklama olmadan görünür."""
    view = _view(_equity([10000, 10150]), [_trade(100), _trade(50, cost_known=False)])
    text = _all_text(view)
    assert "maliyet eksik" in text.lower()
    assert "üst sınır" in text.lower()
    assert "1" in "".join(n for b in view.blocks for n in b.notes)  # eksik maliyetli işlem sayısı


def test_ac84_known_costs_have_no_upper_bound_label():
    """AC84 — Maliyetler bilinirken "üst sınır" etiketi yoktur."""
    text = _all_text(_view(_equity([10000, 10100]), [_trade(100)]))
    assert "üst sınır" not in text.lower()
    assert "maliyet eksik" not in text.lower()


def test_ac09_invalid_period_message_is_shown():
    """AC09 — Geçersiz tarih aralığı kullanıcıya açıklamayla gösterilir."""
    view = _view(_equity([10000, 10100]), [],
                 ReportFilters(start=D0 + timedelta(days=3), end=D0))
    assert view.blocks == []
    assert any("Başlangıç" in m for m in view.messages)


def test_ac64_unknown_market_message_has_no_technical_text():
    """AC64 — Bilinmeyen piyasa seçimi anlaşılır mesajla reddedilir; teknik metin sızmaz."""
    view = _view(_equity([10000, 10100]), [], ReportFilters(market="MARS"))
    text = _all_text(view)
    assert "piyasa" in text.lower()
    for leaked in ("Traceback", "Exception", "GECERSIZ_ISTEK", "400"):
        assert leaked not in text


def test_ac78_empty_period_shows_not_computable_not_zero():
    """AC78 — Boş dönemde getiri ve düşüş "hesaplanamıyor" görünür, 0 değil."""
    view = performance_ui.build_report_view(build_report(
        {"ETH-USD": _equity([10000, 10100])}, [],
        ReportFilters(start=D0 + timedelta(days=50), end=D0 + timedelta(days=60))))
    assert view.is_empty
    assert "örnek sayısı: 0" in _all_text(view).lower()


def test_ac78_unmeasurable_metrics_render_as_not_computable():
    """AC78 — Sıfır sermayede getiri ve işlemsiz dönemde beklenti "hesaplanamıyor" yazar."""
    view = _view(_equity([0, 500]), [])
    metrics = dict(view.blocks[0].metrics)
    assert metrics["Net getiri"] == "hesaplanamıyor"
    assert metrics["İşlem başına beklenti"] == "hesaplanamıyor"
    assert metrics["Kapanmış işlem"] == "0"


def test_ac10_each_currency_has_its_own_block():
    """AC10 — Her para birimi ayrı blokta gösterilir; tek parasal toplam yoktur."""
    outcome = build_report(
        {"ETH-USD": _equity([10000, 10100]), "THYAO.IS": _equity([10000, 10100])},
        [_trade(100), ReportTrade("THYAO.IS", "BIST", "TRY", Decimal("100"),
                                  D0 + timedelta(days=1))],
        ReportFilters())
    view = performance_ui.build_report_view(outcome)
    assert [b.title for b in view.blocks] == ["TRY", "USD"]


def _harness():
    from datetime import date, timedelta
    from decimal import Decimal
    import performance_ui
    from performance_report import EquityPoint, ReportFilters, ReportTrade, build_report
    d0 = date(2026, 1, 1)
    eq = [EquityPoint(d0 + timedelta(days=i), Decimal(v)) for i, v in enumerate(("10000", "10150"))]
    trades = [ReportTrade("ETH-USD", "KRIPTO", "USD", Decimal("100"), d0 + timedelta(days=1)),
              ReportTrade("ETH-USD", "KRIPTO", "USD", Decimal("50"), d0 + timedelta(days=1),
                          cost_known=False)]
    performance_ui.render_report_view(
        performance_ui.build_report_view(build_report({"ETH-USD": eq}, trades, ReportFilters())))


def test_ac07_rendered_screen_shows_missing_cost_without_click():
    """AC07 — Çizilen ekranda "maliyet eksik" uyarısı, hiçbir genişleticiye girmeden görünür."""
    at = AppTest.from_function(_harness).run()
    assert not at.exception
    shown = " ".join(str(w.value) for w in at.warning) + " ".join(str(m.value) for m in at.caption)
    assert "maliyet eksik" in shown.lower()
    assert [m.label for m in at.metric] == [
        "Net getiri", "En büyük düşüş", "Kapanmış işlem", "İşlem başına beklenti"]


def test_report_from_backtest_uses_daily_equity_and_flags_unknown_costs():
    """AC84 — Komisyonu bilinmeyen backtest: işlemler maliyet eksik sayılır, sonuç üst sınırdır."""
    import pandas as pd
    from technical_analysis import run_v1_strategy_backtest

    index = pd.date_range("2026-09-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "Open": [99, 100, 110, 120], "High": [101, 200, 115, 125],
        "Low": [98, 95, 105, 118], "Close": [100, 110, 112, 121], "ATR": [4, 5, 5, 5],
    }, index=index)
    result = run_v1_strategy_backtest(
        frame, {index[0]: "AL", index[1]: "BEKLE", index[2]: "SAT"}, initial_cash=Decimal("1000"),
        trade_notional=Decimal("1000"), quantity_step=Decimal("1"))
    outcome = performance_ui.report_from_backtest("ETH-USD", result)
    usd = outcome.report.results["USD"]
    assert usd.closed_count == 1
    assert usd.cost_missing_count == 1 and usd.is_upper_bound
    assert usd.net_return_pct == Decimal("20")


def test_report_from_backtest_unrecognised_symbol_is_rejected_politely():
    """AC64 — Piyasası tanınmayan varlık için rapor anlaşılır mesajla reddedilir."""
    outcome = performance_ui.report_from_backtest(
        "EURUSD=X", {"daily_equity": [], "trades": [], "net_verified": True})
    view = performance_ui.build_report_view(outcome)
    assert view.blocks == [] and "tanınmıyor" in view.messages[0]


def test_smoke_backtest_screen_shows_reliable_performance_report(store, monkeypatch, processed_df):
    """AC07 — Uygulamada backtest sonrası rapor bölümü, eksik maliyet uyarısıyla ana görünümde çıkar."""
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
    at = at.run()
    at = click(at, "🚀 Backtest Başlat")
    assert not at.exception
    assert "Güvenilir Performans Raporu" in " ".join(str(s.value) for s in at.subheader)
    assert "maliyet eksik" in texts(at).lower()
    assert {"Net getiri", "En büyük düşüş", "Kapanmış işlem"} <= {m.label for m in at.metric}
