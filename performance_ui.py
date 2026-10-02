"""Performans raporu arayüzü (spec 0005, Adım 4).

Hesap `performance_report.py`'dedir; burası yalnız gösterim metnini üretir ve
Streamlit'e çizer. Ölçülemeyen metrik "hesaplanamıyor" yazar (0 değil), maliyeti
eksik sonuç "üst sınır" etiketiyle ve eksik işlem sayısıyla ana görünümde durur.
Sunum yuvarlaması yalnız burada ve yalnız gösterim içindir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal

import market_map
from trading_ui import describe_code
from evaluation_window import EvaluationInput, buy_and_hold, comparable
from performance_report import (
    GECERSIZ_ISTEK, OK, CurrencyResult, EquityPoint, ReportFilters, ReportOutcome,
    ReportTrade, build_report,
)

NOT_COMPUTABLE = "hesaplanamıyor"
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class MetricBlock:
    title: str
    metrics: list[tuple[str, str]]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportView:
    blocks: list[MetricBlock]
    messages: list[str]
    sample_count: int = 0
    is_empty: bool = False


def _pct(value: Decimal | None) -> str:
    return NOT_COMPUTABLE if value is None else f"%{value.quantize(_CENT, ROUND_HALF_EVEN)}"


def _money(value: Decimal | None, currency: str) -> str:
    if value is None:
        return NOT_COMPUTABLE
    return f"{value.quantize(_CENT, ROUND_HALF_EVEN)} {currency}"


def _bounded(text: str, upper_bound: bool) -> str:
    return text + " (üst sınır)" if upper_bound and text != NOT_COMPUTABLE else text


def _block(result: CurrencyResult) -> MetricBlock:
    bound = result.is_upper_bound
    notes, warnings = [], []
    if result.is_upper_bound:
        text = (f"Maliyet eksik: {result.cost_missing_count} işlemin maliyeti bilinmiyor; "
                "sonuçlar üst sınırdır, temiz net sonuç değildir.")
        warnings.append(text)
        notes.append(text)
    return MetricBlock(
        title=result.currency,
        metrics=[
            ("Net getiri", _bounded(_pct(result.net_return_pct), bound)),
            ("En büyük düşüş", _pct(result.max_drawdown_pct)),
            ("Kapanmış işlem", str(result.closed_count)),
            ("İşlem başına beklenti",
             _bounded(_money(result.expectancy, result.currency), bound)),
        ],
        notes=notes,
        warnings=warnings,
    )


def build_report_view(outcome: ReportOutcome) -> ReportView:
    if outcome.status != OK:
        return ReportView(blocks=[], messages=[outcome.reason])
    report = outcome.report
    if not report.results:
        return ReportView(blocks=[], messages=[
            f"Seçili filtre ve dönemde veri yok. Örnek sayısı: {report.sample_count}."],
            sample_count=report.sample_count, is_empty=True)
    return ReportView(blocks=[_block(r) for _, r in sorted(report.results.items())],
                      messages=[f"Örnek sayısı: {report.sample_count}."],
                      sample_count=report.sample_count)


def report_from_backtest(symbol: str, backtest: dict,
                         filters: ReportFilters | None = None) -> ReportOutcome:
    """V1 backtest çıktısından rapor üretir (günlük kapanış sermaye dizisi + kapanmış işlemler)."""
    info = market_map.market_of(symbol)
    if info is None:
        return ReportOutcome(GECERSIZ_ISTEK, "Bu varlığın piyasası tanınmıyor; rapor üretilemedi.")
    market, currency = info
    equity = [EquityPoint(_as_day(p["date"]), Decimal(p["equity"])) for p in backtest["daily_equity"]]
    cost_known = bool(backtest.get("net_verified"))
    trades = [
        ReportTrade(symbol, market, currency, Decimal(t["pnl"]), _as_day(t["exit_at"]),
                    "V1", cost_known)
        for t in backtest["trades"]
    ]
    return build_report({symbol: equity}, trades, filters or ReportFilters())


def comparison_from_backtest(symbol: str, frame, backtest: dict, quantity_step, costs) -> ReportView:
    """Strateji ile al-tut'u aynı sermaye ve dönemde yan yana gösterir (eşit koşul denetimli)."""
    info = market_map.market_of(symbol)
    currency = info[1] if info else ""
    curve = backtest["daily_equity"]
    capital = Decimal(backtest["initial_cash"])
    days = [_as_day(p["date"]) for p in curve]
    side = EvaluationInput(capital, days[0], days[-1], days, external_cash_flow=False)
    verdict = comparable(side, side)
    if not verdict.ok:
        return ReportView(blocks=[], messages=[verdict.reason])
    hold = buy_and_hold(capital, Decimal(str(frame["Open"].iloc[0])),
                        Decimal(str(frame["Close"].iloc[-1])), quantity_step, costs)
    if not hold.executed:
        return ReportView(blocks=[], messages=[
            f"Al-tut referansı hesaplanamadı: {describe_code(hold.reason)}"])
    strategy_final = Decimal(curve[-1]["equity"])
    suffix = " (üst sınır)" if hold.is_upper_bound else ""
    block = MetricBlock(
        title="Strateji ve Al-Tut (aynı sermaye ve dönem)",
        metrics=[
            ("Strateji son sermaye", _money(strategy_final, currency)),
            ("Al-tut son sermaye", _money(hold.final_equity, currency) + suffix),
            ("Al-tut kalıntı nakit", _money(hold.leftover_cash, currency)),
        ],
        notes=["Al-tut, ilk açılışta maliyet sonrası alınabilen miktarla girer; "
               "geçmiş sonuç gelecekteki kazanç olasılığı değildir."],
    )
    return ReportView(blocks=[block], messages=[])


def _as_day(value) -> date:
    return value.date() if hasattr(value, "date") else value


def render_report_view(view: ReportView) -> None:
    import streamlit as st

    for message in view.messages:
        (st.warning if not view.blocks and not view.is_empty else st.caption)(message)
    for block in view.blocks:
        st.markdown(f"**{block.title}**")
        for warning in block.warnings:
            st.warning(warning)
        columns = st.columns(len(block.metrics))
        for column, (label, value) in zip(columns, block.metrics):
            column.metric(label, value)
