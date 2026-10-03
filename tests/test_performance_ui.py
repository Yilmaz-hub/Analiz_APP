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
    assert metrics["İşlem başına beklenti (USD)"] == "hesaplanamıyor"
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
        "Net getiri (üst sınır)", "En büyük düşüş", "Kapanmış işlem",
        "İşlem başına beklenti (USD) (üst sınır)"]


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


def test_ac84_backtest_open_position_with_unknown_cost_is_upper_bound():
    """AC84 — Backtest sonunda maliyeti bilinmeyen açık pozisyon varsa rapor üst sınırdır."""
    import pandas as pd
    from technical_analysis import run_v1_strategy_backtest

    index = pd.date_range("2026-09-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "Open": [99, 100, 110], "High": [101, 112, 115],
        "Low": [98, 95, 105], "Close": [100, 110, 112], "ATR": [4, 5, 5],
    }, index=index)
    result = run_v1_strategy_backtest(
        frame, {index[0]: "AL", index[1]: "BEKLE"}, initial_cash=Decimal("1000"),
        trade_notional=Decimal("1000"), quantity_step=Decimal("1"))
    assert result["position"] is not None and not result["trades"]
    usd = performance_ui.report_from_backtest("ETH-USD", result).report.results["USD"]
    assert usd.closed_count == 0
    assert usd.is_upper_bound and usd.cost_missing_count == 1


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
    assert {"Net getiri (üst sınır)", "En büyük düşüş", "Kapanmış işlem"} <= {m.label for m in at.metric}
    assert {"Strateji son sermaye", "Al-tut son sermaye"} <= {m.label for m in at.metric}


def _comparison_frame(open_values=None, close_values=None):
    """12 günlük çubuk: ayar dilimi ilk 7, değerlendirme dilimi son 5 (index 7–11)."""
    import pandas as pd

    index = pd.date_range("2026-09-01", periods=12, freq="D", tz="UTC")
    opens = open_values or [100] * 12
    closes = close_values or [100] * 11 + [120]
    frame = pd.DataFrame({"Open": opens, "High": [max(o, c) + 1 for o, c in zip(opens, closes)],
                          "Low": [min(o, c) - 1 for o, c in zip(opens, closes)],
                          "Close": closes, "ATR": [4] * 12}, index=index)
    return frame, index


def _compare(frame, decisions, step=Decimal("1"), commission="0"):
    from trade_execution import CostAssumptions

    zero = Decimal("0")
    costs = CostAssumptions(zero, zero, Decimal(commission))
    return performance_ui.comparison_from_backtest(
        "ETH-USD", frame, decisions, Decimal("1000"), Decimal("1000"), step, costs)


def test_comparison_shows_strategy_and_buy_and_hold_on_same_capital():
    """AC47 — Strateji ile al-tut aynı sermayede yan yana; al-tut maliyet sonrası miktarla girer."""
    frame, index = _comparison_frame()
    view = _compare(frame, {index[7]: "AL"})
    metrics = dict(view.blocks[0].metrics)
    assert metrics["Al-tut son sermaye"] == "1200.00 USD"   # 10 adet × 120
    assert metrics["Al-tut kalıntı nakit"] == "0.00 USD"
    assert metrics["Strateji son sermaye"] == "1200.00 USD"  # aynı anda, aynı fiyattan girdi


def test_ac88_prices_before_evaluation_slice_do_not_change_comparison():
    """AC88 — Değerlendirme diliminden önceki fiyatlar değişince al-tut ve strateji karşılaştırması değişmez."""
    frame, index = _comparison_frame()
    decisions = {index[7]: "AL"}
    before = _compare(frame, decisions)
    altered = frame.copy()
    altered.iloc[:7, :4] = altered.iloc[:7, :4].values * 3
    after = _compare(altered, decisions)
    assert before.blocks[0].metrics == after.blocks[0].metrics


def test_ac88_decisions_inside_tuning_slice_are_not_used():
    """AC88 — Ayar dilimindeki kararlar karşılaştırmada işlem açmaz; ikisi de dilimin ilk açılışında başlar."""
    frame, index = _comparison_frame()
    with_old_decision = _compare(frame, {index[3]: "AL", index[6]: "AL"})
    metrics = dict(with_old_decision.blocks[0].metrics)
    assert metrics["Strateji son sermaye"] == "1000.00 USD"   # dilim içinde karar yok → nakitte
    assert metrics["Al-tut son sermaye"] == "1200.00 USD"
    notes = " ".join(with_old_decision.blocks[0].notes)
    assert "2026-09-09 açılışında başlar" in notes            # index[8], dilimin ilk kapanışından sonra


def test_comparison_short_history_is_flagged_not_sufficient_evidence():
    """AC54 — 250'den az tarihli geçmişte karşılaştırma "yetersiz geçmiş" etiketiyle gösterilir."""
    frame, index = _comparison_frame()
    notes = " ".join(_compare(frame, {index[7]: "AL"}).blocks[0].notes)
    assert "Yetersiz geçmiş" in notes


def test_comparison_without_known_step_explains_instead_of_guessing():
    """AC47 — Miktar adımı bilinmiyorsa al-tut uydurulmaz, neden gösterilir."""
    frame, index = _comparison_frame()
    view = _compare(frame, {index[7]: "AL"}, step=None)
    assert view.blocks == [] and "Miktar adımı bilinmiyor" in view.messages[0]


def test_comparison_evaluation_slice_too_short_gives_message():
    """AC57 — Değerlendirme dilimi karşılaştırmaya yetmiyorsa "yetersiz geçmiş" mesajı görünür."""
    frame, _ = _comparison_frame()
    view = _compare(frame.iloc[:2], {})
    assert view.blocks == [] and "Yetersiz geçmiş" in view.messages[0]


def _backtest_screen(store, monkeypatch, processed_df):
    import technical_analysis
    from app_helpers import click, make_app

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
    return click(at, "🚀 Backtest Başlat")


def _perf_widget(at, collection, label):
    return next(w for w in collection if w.label == label)


def test_ac09_screen_inverted_period_shows_explanation(store, monkeypatch, processed_df):
    """AC09 — Arayüzde başlangıç bitişten sonra seçilirse açıklama görünür, rapor çizilmez."""
    at = _backtest_screen(store, monkeypatch, processed_df)
    start = _perf_widget(at, at.date_input, "Başlangıç")
    end = _perf_widget(at, at.date_input, "Bitiş")
    start.set_value(end.max)
    end.set_value(end.min)
    at = at.run()
    assert not at.exception
    shown = " ".join(str(w.value) for w in at.warning)
    assert "Başlangıç tarihi bitiş tarihinden sonra olamaz" in shown


def test_ac65_screen_market_without_data_shows_empty_not_error(store, monkeypatch, processed_df):
    """AC65 — Arayüzde veri bulunmayan piyasa seçilince hata değil "örnek sayısı: 0" görünür."""
    at = _backtest_screen(store, monkeypatch, processed_df)
    _perf_widget(at, at.selectbox, "Piyasa").set_value("BIST")
    at = at.run()
    assert not at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Örnek sayısı: 0" in captions


def test_smoke_candidate_panel_evaluates_and_lists_candidates(store, monkeypatch, processed_df):
    """AC01, AC80 — Backtest ekranında aday paneli V1'i aktif gösterir; değerlendirme iki adayı ayrı listeler."""
    from app_helpers import click

    at = _backtest_screen(store, monkeypatch, processed_df)
    assert not at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Aktif strateji: V1 (mevcut strateji)" in captions
    at = click(at, "Adayları değerlendir")
    assert not at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Kırılım (20 gün) — KRIPTO:" in captions
    assert "Geri çekilme (EMA20, %1) — KRIPTO:" in captions
    assert "Aktif strateji: V1 (mevcut strateji)" in captions


def test_smoke_risk_panel_profile_save_drives_quantity(store, monkeypatch, processed_df):
    """AC23, AC81 — Ekranda profil yokken miktar önerilmez; profil kaydedilince miktar görünür."""
    from app_helpers import click

    at = _backtest_screen(store, monkeypatch, processed_df)
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Risk profili belirlenmedi" in captions and "Önerilen miktar" not in captions
    at.text_input(key="risk_per_trade").set_value("1")
    at.text_input(key="risk_total").set_value("3")
    at = click(at, "Risk profilini kaydet")
    assert not at.exception
    at = at.run()
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Önerilen miktar:" in captions
