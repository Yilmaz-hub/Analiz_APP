"""Performans raporu çekirdeği (spec 0005, Adım 4 / S1).

Saf hesap modülü: Streamlit içermez, para ve yüzde `Decimal`dir. Beklenen iş
hataları istisna değil `ReportOutcome` durumu olarak döner (`GECERSIZ_ISTEK`,
`BULUNAMADI`); mesajlarda teknik metin bulunmaz. Ölçülemeyen metrik `None`
(yok değeri) taşır, sıfır değil.

Sermaye dizisi sembol başına **günlük kapanış** değerleridir (açık pozisyon
dahil). Aynı para birimindeki semboller yalnız hepsinin değeri bulunan ortak
günlerde toplanır. Para birimleri hiçbir zaman birbirine eklenmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Mapping, Sequence

import market_map
from config import PerformanceConfig

OK = "OK"
GECERSIZ_ISTEK = "GECERSIZ_ISTEK"
BULUNAMADI = "BULUNAMADI"

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


@dataclass(frozen=True)
class EquityPoint:
    day: date
    equity: Decimal


@dataclass(frozen=True)
class ReportTrade:
    """Kapanmış tek işlem; `pnl` bilinen maliyetler sonrası net sonuçtur."""
    symbol: str
    market: str
    currency: str
    pnl: Decimal
    closed_on: date
    strategy_version: str = "V1"
    cost_known: bool = True


@dataclass(frozen=True)
class OpenPosition:
    """Dönem sonunda açık pozisyon; maliyeti bilinmiyorsa rapor üst sınırdır (R03)."""
    symbol: str
    opened_on: date
    cost_known: bool = True


@dataclass(frozen=True)
class ReportFilters:
    market: str | None = None
    symbol: str | None = None
    currency: str | None = None
    strategy_version: str | None = None
    start: date | None = None
    end: date | None = None


@dataclass(frozen=True)
class CurrencyResult:
    currency: str
    net_return_pct: Decimal | None
    max_drawdown_pct: Decimal | None
    closed_count: int
    expectancy: Decimal | None
    cost_missing_count: int
    is_upper_bound: bool


@dataclass(frozen=True)
class Report:
    sample_count: int
    results: dict[str, CurrencyResult] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportOutcome:
    status: str
    reason: str = ""
    report: Report | None = None


def _rejected(reason: str) -> ReportOutcome:
    return ReportOutcome(GECERSIZ_ISTEK, reason)


def net_return(points: Sequence[EquityPoint]) -> Decimal | None:
    """Günlük kapanış sermaye dizisinin net getirisi (%); ölçülemiyorsa `None`."""
    return _net_return(points)


def max_drawdown(points: Sequence[EquityPoint]) -> Decimal | None:
    """Günlük kapanış sermaye dizisinin en büyük düşüşü (%); ölçülemiyorsa `None`."""
    return _max_drawdown(points)


def lookup_report(records: Mapping[str, Report], record_id: str) -> ReportOutcome:
    """Kayıtlı değerlendirmeyi getirir; yoksa `BULUNAMADI`."""
    record = records.get(record_id)
    if record is None:
        return ReportOutcome(BULUNAMADI, "İstenen değerlendirme kaydı bulunamadı.")
    return ReportOutcome(OK, "", record)


def _combine_equity(series_list: Sequence[Sequence[EquityPoint]]) -> list[EquityPoint]:
    """Seriler yalnız ortak günlerde toplanır (hepsinde değeri olan günler)."""
    maps = [{point.day: point.equity for point in series} for series in series_list]
    if not maps:
        return []
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    return [EquityPoint(day, sum((m[day] for m in maps), _ZERO)) for day in sorted(common)]


def _net_return(points: Sequence[EquityPoint]) -> Decimal | None:
    if not points:
        return None
    start, end = points[0].equity, points[-1].equity
    if start <= 0:
        return None
    return (end / start - 1) * _HUNDRED


def _max_drawdown(points: Sequence[EquityPoint]) -> Decimal | None:
    if not points:
        return None
    peak = points[0].equity
    worst = _ZERO
    for point in points:
        peak = max(peak, point.equity)
        if peak <= 0:
            return None
        worst = max(worst, (peak - point.equity) / peak * _HUNDRED)
    return worst


def _in_period(day: date, filters: ReportFilters) -> bool:
    return (filters.start is None or day >= filters.start) and (
        filters.end is None or day <= filters.end)


def _symbol_selected(symbol: str, filters: ReportFilters) -> bool:
    info = market_map.market_of(symbol)
    if info is None:
        return False
    market, currency = info
    return ((filters.symbol is None or symbol == filters.symbol)
            and (filters.market is None or market == filters.market)
            and (filters.currency is None or currency == filters.currency))


def build_report(equity_by_symbol: Mapping[str, Sequence[EquityPoint]],
                 trades: Sequence[ReportTrade], filters: ReportFilters,
                 open_positions: Sequence[OpenPosition] = ()) -> ReportOutcome:
    if filters.market is not None and filters.market not in PerformanceConfig.MARKETS:
        return _rejected("Bilinmeyen piyasa seçildi; tanımlı piyasalardan birini seçin.")
    if filters.start is not None and filters.end is not None and filters.start > filters.end:
        return _rejected("Başlangıç tarihi bitiş tarihinden sonra olamaz.")

    selected = [
        t for t in trades
        if _in_period(t.closed_on, filters)
        and (filters.symbol is None or t.symbol == filters.symbol)
        and (filters.market is None or t.market == filters.market)
        and (filters.currency is None or t.currency == filters.currency)
        and (filters.strategy_version is None or t.strategy_version == filters.strategy_version)
    ]

    by_currency_series: dict[str, list[Sequence[EquityPoint]]] = {}
    for symbol, series in equity_by_symbol.items():
        if not _symbol_selected(symbol, filters):
            continue
        period = [p for p in series if _in_period(p.day, filters)]
        if period:
            currency = market_map.market_of(symbol)[1]
            by_currency_series.setdefault(currency, []).append(period)

    open_missing: dict[str, int] = {}
    for position in open_positions:
        if (position.cost_known or not _symbol_selected(position.symbol, filters)
                or (filters.end is not None and position.opened_on > filters.end)):
            continue
        currency = market_map.market_of(position.symbol)[1]
        open_missing[currency] = open_missing.get(currency, 0) + 1

    currencies = set(by_currency_series) | {t.currency for t in selected}
    results: dict[str, CurrencyResult] = {}
    for currency in sorted(currencies):
        points = _combine_equity(by_currency_series.get(currency, []))
        own = [t for t in selected if t.currency == currency]
        missing = sum(1 for t in own if not t.cost_known) + open_missing.get(currency, 0)
        results[currency] = CurrencyResult(
            currency=currency,
            net_return_pct=_net_return(points),
            max_drawdown_pct=_max_drawdown(points),
            closed_count=len(own),
            expectancy=(sum((t.pnl for t in own), _ZERO) / len(own)) if own else None,
            cost_missing_count=missing,
            is_upper_bound=missing > 0,
        )
    return ReportOutcome(OK, "", Report(sample_count=len(selected), results=results))
