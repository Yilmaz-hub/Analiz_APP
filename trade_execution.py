from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from trade_decisions import (
    LOSS_COOLDOWN_BARS, Action, Position, PositionState, Signal, classify_exit, decide_action,
    initial_stop as compute_initial_stop, signal_from_verdict,
)

#: `initial_stop` verilmediyse stop denetimi atlanır; açıkça None verilirse
#: GECERSIZ_STOP döner (AC110).
_UNSET = object()


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


def execute_purchase(cash, notional, price, *, fee, quantity_step, initial_stop=_UNSET, minimum_quantity=None, minimum_notional=None):
    cash, notional, price, fee = map(Decimal, (cash, notional, price, fee))
    if initial_stop is not _UNSET and (initial_stop is None or Decimal(initial_stop) <= 0):
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


# --- Ortak işlem motoru (spec 0003, Q1/Q7/Q8) --------------------------------
# Geçmiş test, sanal takip ve karar paneli aynı kurallardan geçer: aşağıdaki
# işlevler tek gerçek kaynaktır; motorlar kendi fiyat/komisyon/bekleme
# hesabını yapmaz.

@dataclass(frozen=True)
class TradeSettings:
    """Kullanıcının işlem varsayımları. Maliyette None = bilinmiyor, 0 = açıkça sıfır."""
    notional: Decimal
    quantity_step: Decimal | None = None
    costs: CostAssumptions = CostAssumptions(None, None, None)

    @property
    def net_verified(self):
        """Doğrulanmış net sonuç yalnız komisyon biliniyorsa sunulur."""
        return self.costs.commission_pct is not None


def fill_price(base, side, costs):
    """Makas ve kayma uygulanmış model fiyatı; komisyondan bağımsızdır.

    Bilinmeyen (None) makas/kayma etkisiz sayılır ama sıfır maliyet diye
    sunulmaz (çağıran `TradeSettings`/bayraklarla gösterir). Geçersiz maliyette
    ya da yarım makas + kayma 10.000 baz puana ulaşırsa None döner (AC112).
    """
    price = Decimal(base)
    parts = []
    for value, half in ((costs.spread_bps, True), (costs.slippage_bps, False)):
        if value is None:
            continue
        amount = Decimal(value)
        if not amount.is_finite() or amount < 0:
            return None
        parts.append(amount / Decimal("2") if half else amount)
    effect_bps = sum(parts, Decimal("0"))
    if effect_bps >= Decimal("10000"):
        return None
    effect = effect_bps / Decimal("10000")
    result = price * (Decimal("1") + effect if side == "BUY" else Decimal("1") - effect)
    return result if result > 0 else None


def commission_fee(amount, costs):
    """İşlem tutarına uygulanan komisyon; bilinmiyorsa 0 (brüt model)."""
    if costs.commission_pct is None:
        return Decimal("0")
    return Decimal(amount) * Decimal(costs.commission_pct) / Decimal("100")


@dataclass(frozen=True)
class Entry:
    executed: bool = False
    reason: str = ""
    quantity: Decimal = Decimal("0")
    price: Decimal | None = None
    fee: Decimal = Decimal("0")
    stop: Decimal | None = None
    cash_after: Decimal = Decimal("0")


def enter_position(*, cash, open_price, atr, settings):
    """Açılışta alım: model fiyatı, başlangıç stopu ve miktar tek yerde.

    Komisyon alış tutarına ek olarak nakitten düşülür (AC94/AC95). Başlangıç
    stopu pozitif değilse (AC110) ya da miktar adımı bilinmiyorsa (AC102) alım
    yapılmaz ve neden döner.
    """
    cash = Decimal(cash)
    price = fill_price(open_price, "BUY", settings.costs)
    if price is None:
        return Entry(False, "GECERSIZ_MALIYET", cash_after=cash)
    try:
        stop = compute_initial_stop(price, atr)
    except ArithmeticError:
        stop = None
    step = settings.quantity_step
    fee = Decimal("0")
    if step is not None and Decimal(step) > 0:
        quantity = ((Decimal(settings.notional) / price) / Decimal(step)).to_integral_value(
            rounding=ROUND_FLOOR) * Decimal(step)
        fee = commission_fee(quantity * price, settings.costs)
    purchase = execute_purchase(
        cash, settings.notional, price, fee=fee, quantity_step=step, initial_stop=stop)
    if not purchase.executed:
        return Entry(False, purchase.reason, cash_after=cash)
    return Entry(True, "", purchase.quantity, price, fee, stop, purchase.cash_after)


def resolve_exit(bar, stop, signal, *, entry_at):
    """Açık pozisyonun bu mumdaki çıkışı (yoksa None).

    Önce açılış fazı (açılış stopta ya da altındaysa STOP, değilse SAT sinyali
    açılışta satar), sonra gün içi faz (düşük stopa değerse stop seviyesinde).
    Karar `decide_action` ile verilir; fiyat `evaluate_stop` ile bulunur.
    """
    level = Decimal(stop)
    position = Position(PositionState.OPEN, stop=level)
    touch = evaluate_stop(bar, level, entry_at, entered_at=entry_at)
    opening = decide_action(
        signal, position, stop_touched=touch is not None and bar.open <= level)
    if opening.action is Action.SELL:
        if opening.reason == "STOP":
            return touch
        return Fill(bar.at, bar.open, "SAT", 1)
    intraday = decide_action(Signal.WAIT, position, stop_touched=touch is not None)
    return touch if intraday.action is Action.SELL else None


@dataclass(frozen=True)
class OpenPosition:
    entry_at: object
    entry: Decimal
    quantity: Decimal
    cost: Decimal          # alış tutarı + alış komisyonu
    entry_fee: Decimal
    stop: Decimal


@dataclass(frozen=True)
class ClosedTrade:
    entry_at: object
    entry: Decimal
    exit_at: object
    exit: Decimal
    quantity: Decimal
    reason: str
    pnl: Decimal           # komisyon biliniyorsa net, bilinmiyorsa brüt
    net_verified: bool
    provisional: bool      # bekleme kararı brüt farka göre verildi (AC86)


@dataclass
class BookState:
    cash: Decimal
    position: OpenPosition | None = None
    cooldown: int = 0


@dataclass
class StepResult:
    trades: list
    opened: bool = False
    blocked: str | None = None


def _close(state, position, base_price, at, reason, settings):
    exit_price = fill_price(base_price, "SELL", settings.costs) or Decimal(base_price)
    proceeds = position.quantity * exit_price
    exit_fee = commission_fee(proceeds, settings.costs)
    state.cash += proceeds - exit_fee
    gross = proceeds - (position.cost - position.entry_fee)
    fees = position.entry_fee + exit_fee
    outcome = classify_exit(gross, fee_known=settings.net_verified, fees=fees)
    if outcome.cooldown_required:
        state.cooldown = LOSS_COOLDOWN_BARS
    state.position = None
    return ClosedTrade(
        position.entry_at, position.entry, at, exit_price, position.quantity, reason,
        gross - fees, settings.net_verified, outcome.provisional,
    )


def advance_daily_bar(state, bar, verdict, atr, settings):
    """Bir günlük mumu ilerletir: önceki kapanmış mumun kararı bu mumun açılışında uygulanır.

    `verdict` ve `atr` karar mumuna aittir (sonraki mumun ATR'si stopu oynatmaz,
    AC84). Çıkış, bekleme, giriş ve aynı mumda stop bu tek işlevden geçer.
    """
    signal = signal_from_verdict(verdict)
    result = StepResult([])
    position = state.position
    if position is not None:
        fill = resolve_exit(bar, position.stop, signal, entry_at=position.entry_at)
        if fill is not None:
            result.trades.append(_close(state, position, fill.base_price, bar.at, fill.reason, settings))
        return result

    if state.cooldown > 0:
        state.cooldown -= 1
        return result
    decision = decide_action(signal, Position.flat())
    if decision.action is not Action.BUY:
        return result

    entry = enter_position(cash=state.cash, open_price=bar.open, atr=atr, settings=settings)
    if not entry.executed:
        result.blocked = entry.reason
        return result
    state.cash = entry.cash_after
    state.position = OpenPosition(
        bar.at, entry.price, entry.quantity, entry.quantity * entry.price + entry.fee,
        entry.fee, entry.stop,
    )
    result.opened = True
    same_bar = resolve_exit(bar, entry.stop, Signal.WAIT, entry_at=bar.at)
    if same_bar is not None:
        result.trades.append(_close(state, state.position, same_bar.base_price, bar.at,
                                    same_bar.reason, settings))
    return result
