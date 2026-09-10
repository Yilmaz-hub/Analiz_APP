from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR


@dataclass(frozen=True)
class Bar:
    at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class CostAssumptions:
    spread_bps: Decimal | None
    slippage_bps: Decimal | None
    commission_pct: Decimal | None


@dataclass(frozen=True)
class Validation:
    is_valid: bool = False
    is_complete: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    at: datetime | None = None
    base_price: Decimal | None = None
    reason: str = ""
    exit_count: int = 0


@dataclass(frozen=True)
class Purchase:
    executed: bool = False
    reason: str = ""
    quantity: Decimal = Decimal("0")
    spent: Decimal = Decimal("0")
    cash_after: Decimal = Decimal("0")


def _finite(value):
    return value is not None and Decimal(value).is_finite()


def validate_costs(costs):
    values = (costs.spread_bps, costs.slippage_bps, costs.commission_pct)
    complete = all(value is not None for value in values)
    if not complete:
        return Validation(True, False, "MALIYET_BILINMIYOR")
    if not all(_finite(value) for value in values):
        return Validation(False, True, "SONLU_OLMAYAN_MALIYET")
    spread, slippage, commission = map(Decimal, values)
    valid = spread >= 0 and slippage >= 0 and Decimal("0") <= commission < Decimal("100")
    valid = valid and spread / Decimal("2") + slippage < Decimal("10000")
    return Validation(valid, True, "" if valid else "GECERSIZ_MALIYET")


def adjusted_price(base, side, costs, *, costs_included=False, confirmed_real=False):
    price = Decimal(base)
    if costs_included or confirmed_real:
        return price
    validation = validate_costs(costs)
    if not validation.is_valid or not validation.is_complete:
        return price
    effect = (Decimal(costs.spread_bps) / Decimal("2") + Decimal(costs.slippage_bps)) / Decimal("10000")
    result = price * (Decimal("1") + effect if side == "BUY" else Decimal("1") - effect)
    if result <= 0:
        raise ValueError("GECERSIZ_GERCEKLESME_FIYATI")
    return result


def execute_next_open(signal_known_at, next_bar, side):
    if next_bar.at <= signal_known_at:
        raise ValueError("SINYALDEN_ONCE_ISLEM")
    return Fill(next_bar.at, next_bar.open, side, 1)


def evaluate_stop(bar, stop, active_at, *, info_target=None, pending_sell=False, entered_at=None):
    level = Decimal(stop)
    if level <= 0 or active_at > bar.at or (entered_at is not None and entered_at > bar.at):
        return None
    if bar.open <= level:
        return Fill(bar.at, bar.open, "STOP", 1)
    if bar.low <= level:
        return Fill(bar.at, level, "STOP", 1)
    return None


def execute_purchase(cash, notional, price, *, fee, quantity_step, initial_stop=None, minimum_quantity=None, minimum_notional=None):
    cash, notional, price, fee = map(Decimal, (cash, notional, price, fee))
    if initial_stop is not None and Decimal(initial_stop) <= 0:
        return Purchase(False, "GECERSIZ_STOP", cash_after=cash)
    if quantity_step is None or Decimal(quantity_step) <= 0:
        return Purchase(False, "MIKTAR_ADIMI_BILINMIYOR", cash_after=cash)
    step = Decimal(quantity_step)
    quantity = ((notional / price) / step).to_integral_value(rounding=ROUND_FLOOR) * step
    spent = quantity * price
    if quantity <= 0 or (minimum_quantity is not None and quantity < Decimal(minimum_quantity)):
        return Purchase(False, "ASGARI_MIKTAR", cash_after=cash)
    if minimum_notional is not None and spent < Decimal(minimum_notional):
        return Purchase(False, "ASGARI_TUTAR", cash_after=cash)
    if spent + fee > cash:
        return Purchase(False, "YETERSIZ_NAKIT", cash_after=cash)
    return Purchase(True, "", quantity, spent, cash - spent - fee)


def round_stop_up(raw_stop, price_step, entry):
    raw, step, entry = map(Decimal, (raw_stop, price_step, entry))
    if raw <= 0 or step <= 0:
        return None
    rounded = (raw / step).to_integral_value(rounding=ROUND_CEILING) * step
    return rounded if rounded < entry else None
