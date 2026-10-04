"""Değerlendirme dönemi, al-tut referansı ve eşit-koşul denetimi (spec 0005, Adım 4 / S2).

Saf hesap modülü; para `Decimal`dir. Karar kaynağı Q01:

* Ortak tarihlerde son `LOOKBACK_YEARS` yıl alınır; ilk %60'ı ayar seçimine,
  kalanı dokunulmamış değerlendirmeye gider. Bölme tam değilse ayar dilimi
  **aşağı yuvarlanır** (kalan tarihler değerlendirmeye).
* 250'den az uygun tarih "yetersiz geçmiş"tir; bölme yine yapılır ama sonuç
  yeterli kanıt olarak sunulmaz.
* Al-tut ilk uygun açılışta, maliyet sonrası alınabilen miktarla girer; bölünmeyen
  kalıntı nakit olarak dönem sonu sermayesine eklenir.
* Karşılaştırma yalnız başlangıç sermayesi ve dönem **tam eşitse** ve dışarıdan
  nakit hareketi yoksa yapılır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from typing import Sequence

from config import PerformanceConfig
from trade_execution import CostAssumptions, commission_fee, execute_purchase, fill_price


@dataclass(frozen=True)
class SplitResult:
    tuning: list[date]
    evaluation: list[date]
    sufficient_history: bool


@dataclass(frozen=True)
class PeriodSelection:
    dates: list[date]
    insufficient_history: bool = False
    reason: str = ""


@dataclass(frozen=True)
class EvaluationInput:
    initial_capital: Decimal
    start: date
    end: date
    dates: Sequence[date] = field(default_factory=list)
    external_cash_flow: bool = False


@dataclass(frozen=True)
class Comparability:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class BuyHoldResult:
    executed: bool
    reason: str = ""
    quantity: Decimal = Decimal("0")
    leftover_cash: Decimal = Decimal("0")
    final_equity: Decimal | None = None
    is_upper_bound: bool = False


def _years_back(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 Şubat
        return day.replace(year=day.year - years, day=28)


def split_dates(dates: Sequence[date]) -> SplitResult:
    ordered = sorted(set(dates))
    if ordered:
        cutoff = _years_back(ordered[-1], PerformanceConfig.LOOKBACK_YEARS)
        ordered = [day for day in ordered if day >= cutoff]
    tuning_count = int((Decimal(len(ordered)) * PerformanceConfig.TUNING_SHARE)
                       .to_integral_value(rounding=ROUND_FLOOR))
    return SplitResult(
        tuning=ordered[:tuning_count],
        evaluation=ordered[tuning_count:],
        sufficient_history=len(ordered) >= PerformanceConfig.MIN_HISTORY_DAYS,
    )


def select_period(asset_dates: Sequence[date], start: date, end: date) -> PeriodSelection:
    """Dönemdeki (sınırlar dahil) varlık tarihleri; hiç yoksa "yetersiz geçmiş"."""
    chosen = sorted(day for day in set(asset_dates) if start <= day <= end)
    if not chosen:
        return PeriodSelection([], True, "Yetersiz geçmiş: seçili dönemde varlığın verisi yok.")
    return PeriodSelection(chosen)


def comparable(first: EvaluationInput, second: EvaluationInput) -> Comparability:
    if first.external_cash_flow or second.external_cash_flow:
        return Comparability(False, "Dönemde dışarıdan nakit hareketi var; karşılaştırma uygun değil.")
    if first.initial_capital != second.initial_capital:
        return Comparability(False, "Başlangıç sermayeleri eşit değil; eşit koşullarda karşılaştırılamaz.")
    if (first.start, first.end) != (second.start, second.end):
        return Comparability(False, "Dönemler eşit değil; eşit koşullarda karşılaştırılamaz.")
    if not set(first.dates) & set(second.dates):
        return Comparability(False, "Ortak işlem tarihi yok; karşılaştırma yapılamaz.")
    return Comparability(True)


def buy_and_hold(capital: Decimal, first_open: Decimal, last_close: Decimal,
                 quantity_step: Decimal | None, costs: CostAssumptions) -> BuyHoldResult:
    """İlk uygun açılışta maliyet sonrası alınabilen miktarla al-tut; dönem sonu sermaye değeri.

    Komisyon bilinmiyorsa sıfır kabul edilmez: sonuç `is_upper_bound` ile etiketlenir.
    """
    upper_bound = costs.commission_pct is None
    cash = Decimal(capital)
    price = fill_price(Decimal(first_open), "BUY", costs)
    if price is None:
        return BuyHoldResult(False, "GECERSIZ_MALIYET", leftover_cash=cash, is_upper_bound=upper_bound)
    commission_rate = Decimal("0") if costs.commission_pct is None else Decimal(costs.commission_pct) / 100
    affordable = cash / (Decimal("1") + commission_rate)
    if quantity_step is None or Decimal(quantity_step) <= 0:
        return BuyHoldResult(False, "MIKTAR_ADIMI_BILINMIYOR", leftover_cash=cash,
                             is_upper_bound=upper_bound)
    step = Decimal(quantity_step)
    quantity = ((affordable / price) / step).to_integral_value(rounding=ROUND_FLOOR) * step
    fee = commission_fee(quantity * price, costs)
    purchase = execute_purchase(cash, quantity * price, price, fee=fee, quantity_step=step)
    if not purchase.executed:
        return BuyHoldResult(False, purchase.reason, leftover_cash=cash, is_upper_bound=upper_bound)
    return BuyHoldResult(
        True, "", purchase.quantity, purchase.cash_after,
        purchase.quantity * Decimal(last_close) + purchase.cash_after, upper_bound)
