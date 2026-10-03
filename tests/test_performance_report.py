"""Spec 0005 Adım 4 / S1 — performans raporu çekirdeği (AC02–AC10, 49, 50, 55, 64–66, 78, 84)."""
from datetime import date, timedelta
from decimal import Decimal

import market_map
from performance_report import (
    BULUNAMADI, GECERSIZ_ISTEK, OK,
    EquityPoint, OpenPosition, ReportFilters, ReportTrade, build_report, lookup_report,
)

D0 = date(2026, 1, 1)


def _series(values, start=D0):
    return [EquityPoint(start + timedelta(days=i), Decimal(str(v))) for i, v in enumerate(values)]


def _trade(pnl, symbol="ETH-USD", day=1, cost_known=True, version="V1"):
    market, currency = market_map.market_of(symbol)
    return ReportTrade(symbol, market, currency, Decimal(str(pnl)), D0 + timedelta(days=day),
                       version, cost_known)


def _report(equity_by_symbol, trades, filters=None):
    outcome = build_report(equity_by_symbol, trades, filters or ReportFilters())
    assert outcome.status == OK, outcome.reason
    return outcome.report


def test_ac02_empty_period_has_no_trades_and_no_expectancy():
    """AC02 — Boş durum: işlem yoksa sayı 0, beklenti yok değeri."""
    report = _report({"ETH-USD": _series([10000, 10000])}, [])
    result = report.results["USD"]
    assert result.closed_count == 0
    assert result.expectancy is None


def test_ac03_net_return_is_five_percent():
    """AC03 — 10.000 → 10.500 net getiri %5."""
    result = _report({"ETH-USD": _series([10000, 10500])}, []).results["USD"]
    assert result.net_return_pct == Decimal("5")


def test_ac04_max_drawdown_is_twenty_five_percent():
    """AC04 — 10.000/12.000/9.000/11.000 dizisinde maksimum düşüş %25."""
    result = _report({"ETH-USD": _series([10000, 12000, 9000, 11000])}, []).results["USD"]
    assert result.max_drawdown_pct == Decimal("25")


def test_ac05_expectancy_is_average_net_result():
    """AC05 — +100 ve −40 kapanmış işlem → beklenti +30, para biriminde."""
    trades = [_trade(100), _trade(-40, day=2)]
    result = _report({"ETH-USD": _series([10000, 10060, 10060])}, trades).results["USD"]
    assert result.expectancy == Decimal("30")
    assert result.closed_count == 2


def test_ac06_open_position_counts_in_equity_not_in_closed_count():
    """AC06 — Yalnız açık pozisyon: kapanmış 0, pozisyon değeri sermayede."""
    result = _report({"ETH-USD": _series([10000, 10400])}, []).results["USD"]
    assert result.closed_count == 0
    assert result.net_return_pct == Decimal("4")


def test_ac08_symbol_filter_uses_only_that_symbols_trades():
    """AC08 — ETH filtresinde metrikler yalnız ETH işlemleriyle hesaplanır."""
    trades = [_trade(100, "ETH-USD"), _trade(500, "THYAO.IS")]
    equity = {"ETH-USD": _series([10000, 10100]), "THYAO.IS": _series([10000, 10500])}
    report = _report(equity, trades, ReportFilters(symbol="ETH-USD"))
    assert set(report.results) == {"USD"}
    assert report.results["USD"].closed_count == 1
    assert report.results["USD"].net_return_pct == Decimal("1")
    assert report.sample_count == 1


def test_ac09_start_after_end_is_rejected_with_reason():
    """AC09 — Başlangıcı bitişinden sonra olan dönem reddedilir ve açıklama gösterilir."""
    outcome = build_report({"ETH-USD": _series([10000, 10100])}, [],
                           ReportFilters(start=D0 + timedelta(days=5), end=D0))
    assert outcome.status == GECERSIZ_ISTEK
    assert outcome.reason


def test_ac10_currencies_are_never_summed():
    """AC10 — 100 TRY ve 100 USD sonuç tek parasal toplam olarak gösterilmez."""
    trades = [_trade(100, "THYAO.IS"), _trade(100, "ETH-USD")]
    equity = {"THYAO.IS": _series([10000, 10100]), "ETH-USD": _series([10000, 10100])}
    report = _report(equity, trades)
    assert report.results["TRY"].expectancy == Decimal("100")
    assert report.results["USD"].expectancy == Decimal("100")
    assert all(Decimal("200") not in (r.expectancy,) for r in report.results.values())
    assert set(report.results) == {"TRY", "USD"}


def test_ac49_zero_or_negative_start_gives_no_return():
    """AC49 — Başlangıç sermayesi 0 veya negatif: getiri yüzdesi üretilmez."""
    for start in (0, -100):
        result = _report({"ETH-USD": _series([start, 500])}, []).results["USD"]
        assert result.net_return_pct is None


def test_ac50_drawdown_uses_daily_closes_only():
    """AC50 — Gün içi 8.000'e inip 11.000 kapanan gün: düşüş yalnız günlük kapanış dizisinden hesaplanır."""
    import pandas as pd
    from technical_analysis import run_v1_strategy_backtest
    from trade_execution import CostAssumptions

    zero = Decimal("0")
    rows = [  # (open, high, low, close, atr); 2. günde gün içi dip 91 ama kapanış 105
        (99, 101, 98, 100, 4), (100, 105, 95, 102, 4), (102, 110, 91, 105, 4), (105, 112, 104, 110, 4)]
    index = pd.date_range("2026-09-01", periods=len(rows), freq="D")
    frame = pd.DataFrame([dict(Open=o, High=h, Low=l, Close=c, ATR=a) for o, h, l, c, a in rows],
                         index=index)
    result = run_v1_strategy_backtest(
        frame, {index[0]: "AL"}, initial_cash=Decimal("10000"), trade_notional=Decimal("1000"),
        quantity_step=Decimal("1"), costs=CostAssumptions(zero, zero, zero))
    curve = result["daily_equity"]
    assert [p["date"] for p in curve] == list(index)
    # 10 adet @100 alındı (1. gün açılış); her gün nakit + adet × kapanış.
    assert [p["equity"] for p in curve] == [Decimal(v) for v in (10000, 10020, 10050, 10100)]
    points = [EquityPoint(p["date"].date(), p["equity"]) for p in curve]
    drawdown = _report({"ETH-USD": points}, []).results["USD"].max_drawdown_pct
    assert drawdown == Decimal("0")  # gün içi 91 dibi dizide yok


def test_ac55_single_day_period_is_accepted():
    """AC55 — Başlangıcı bitişine eşit dönem reddedilmez; sonuç o günün verisiyle üretilir."""
    day = D0 + timedelta(days=1)
    outcome = build_report({"ETH-USD": _series([10000, 10500, 10700])}, [],
                           ReportFilters(start=day, end=day))
    assert outcome.status == OK
    result = outcome.report.results["USD"]
    assert result.net_return_pct == Decimal("0")
    assert result.max_drawdown_pct == Decimal("0")


def test_ac64_unknown_market_is_rejected_without_technical_text():
    """AC64 — Tanımlı olmayan piyasa değeri doğrulama hatası olarak reddedilir, teknik metin sızmaz."""
    outcome = build_report({"ETH-USD": _series([10000, 10100])}, [], ReportFilters(market="MARS"))
    assert outcome.status == GECERSIZ_ISTEK
    assert outcome.reason
    assert "Traceback" not in outcome.reason and "Error" not in outcome.reason


def test_ac65_defined_filter_without_data_gives_empty_result_not_error():
    """AC65 — Tanımlı ama verisi olmayan filtre: hata değil boş sonuç, örnek sayısı 0."""
    outcome = build_report({"ETH-USD": _series([10000, 10100])}, [_trade(10)],
                           ReportFilters(market="BIST"))
    assert outcome.status == OK
    assert outcome.report.sample_count == 0
    assert outcome.report.results == {}


def test_ac66_missing_record_returns_not_found():
    """AC66 — Olmayan değerlendirme kaydı istendiğinde bulunamadı sonucu döner."""
    outcome = lookup_report({}, "yok-boyle-bir-kayit")
    assert outcome.status == BULUNAMADI
    assert outcome.reason


def test_ac78_empty_period_has_no_return_and_no_drawdown():
    """AC78 — Ölçülemeyen metrik 0 değil yok değeri taşır; boş dönemde getiri ve düşüş de yok."""
    outcome = build_report({"ETH-USD": _series([10000, 10100])}, [],
                           ReportFilters(start=D0 + timedelta(days=50), end=D0 + timedelta(days=60)))
    assert outcome.status == OK
    assert outcome.report.results == {}
    assert outcome.report.sample_count == 0


def test_ac84_missing_cost_marks_upper_bound_and_counts_trades():
    """AC84 — Maliyeti bilinmeyen işlem: metrik üst sınır, eksik maliyetli işlem sayısı görünür."""
    trades = [_trade(100), _trade(50, day=2, cost_known=False)]
    result = _report({"ETH-USD": _series([10000, 10150, 10150])}, trades).results["USD"]
    assert result.is_upper_bound is True
    assert result.cost_missing_count == 1
    assert result.closed_count == 2  # örneklemden düşmez


def test_ac84_known_costs_are_not_upper_bound():
    """AC84 — Tüm maliyetler bilinirken üst sınır etiketi yok."""
    result = _report({"ETH-USD": _series([10000, 10100])}, [_trade(100)]).results["USD"]
    assert result.is_upper_bound is False
    assert result.cost_missing_count == 0


def test_ac84_open_position_with_unknown_cost_marks_upper_bound():
    """AC84 — Maliyeti bilinmeyen açık pozisyon: kapanmış işlem olmasa da metrik üst sınırdır."""
    outcome = build_report({"ETH-USD": _series([10000, 10500])}, [], ReportFilters(),
                           open_positions=[OpenPosition("ETH-USD", D0, cost_known=False)])
    result = outcome.report.results["USD"]
    assert result.is_upper_bound is True
    assert result.cost_missing_count == 1
    assert result.closed_count == 0  # açık pozisyon kapanmış sayılmaz (AC06)


def test_ac84_open_position_with_known_cost_is_not_upper_bound():
    """AC84 — Maliyeti bilinen açık pozisyon üst sınır etiketi doğurmaz."""
    outcome = build_report({"ETH-USD": _series([10000, 10500])}, [], ReportFilters(),
                           open_positions=[OpenPosition("ETH-USD", D0, cost_known=True)])
    assert outcome.report.results["USD"].is_upper_bound is False


def test_ac84_open_position_outside_filter_is_not_counted():
    """AC84 — Filtre dışındaki varlığın açık pozisyonu başka varlığın raporunu üst sınır yapmaz."""
    outcome = build_report(
        {"ETH-USD": _series([10000, 10500]), "BTC-USD": _series([5000, 5100])}, [],
        ReportFilters(symbol="ETH-USD"),
        open_positions=[OpenPosition("BTC-USD", D0, cost_known=False)])
    assert outcome.report.results["USD"].is_upper_bound is False


def test_market_map_known_symbols():
    """Piyasa/para birimi tablosu: tanınan semboller ve tanınmayan."""
    assert market_map.market_of("ETH-USD") == ("KRIPTO", "USD")
    assert market_map.market_of("THYAO.IS") == ("BIST", "TRY")
    assert market_map.market_of("GRAM_TRY") == ("ALTIN", "TRY")
    assert market_map.market_of("XAU_GOLD") == ("ALTIN", "USD")
    assert market_map.market_of("AAPL") == ("ABD", "USD")
    assert market_map.market_of("EURUSD=X") is None
    assert market_map.market_of("") is None
