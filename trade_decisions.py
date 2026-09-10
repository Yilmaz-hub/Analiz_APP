from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_contracts import Action, Decision, ExitClassification, Position, PositionState, Signal


@dataclass(frozen=True)
class Validation:
    is_valid: bool
    protection: object | None = None
    reason: str = ""


@dataclass(frozen=True)
class Protection:
    level: Decimal
    kind: str
    effective_at: datetime | None = None

    @classmethod
    def initial(cls, level):
        return cls(Decimal(level), "INITIAL")

    @classmethod
    def updated(cls, level, effective_at):
        return cls(Decimal(level), "UPDATED", effective_at)

    def validate(self, entry_price):
        entry = Decimal(entry_price)
        valid = self.level > 0 and (self.kind == "UPDATED" or self.level < entry)
        return Validation(valid, self if valid else None, "" if valid else "GECERSIZ_STOP")


def decide_action(
    signal: Signal,
    position: Position,
    *,
    stop_touched: bool = False,
    target_enabled: bool = False,
    cooldown_bars: int = 2,
    suggested_stop: Decimal | None = None,
):
    if position.state in (PositionState.UNKNOWN, PositionState.INVALID):
        return Decision(Action.NONE, "POZISYON_BILINMIYOR", protection=None)
    active_stop = position.stop if position.state is PositionState.OPEN else None
    if position.state is PositionState.OPEN and stop_touched and active_stop is not None:
        return Decision(Action.SELL, "STOP", active_stop, 1)
    if position.state is PositionState.OPEN:
        if signal is Signal.SELL:
            return Decision(Action.SELL, "SAT_SINYALI", active_stop, 1)
        return Decision(Action.HOLD, "TREND_KORUNUYOR", active_stop)
    if signal is Signal.BUY and cooldown_bars >= 2:
        return Decision(Action.BUY, "AL_SINYALI")
    return Decision(Action.NONE, "BEKLEME" if signal is Signal.BUY else "POZISYON_YOK")


def validate_position(position):
    if position.state is not PositionState.OPEN:
        return Validation(position.state is PositionState.FLAT)
    if (position.quantity is None or position.quantity <= 0 or
            position.entry_price is None or position.entry_price <= 0 or
            position.executed_at is None):
        return Validation(False, reason="GECERSIZ_POZISYON")
    protection = None
    if position.stop is not None:
        candidate = Protection.initial(position.stop)
        protection = candidate if candidate.validate(position.entry_price).is_valid else None
    return Validation(True, protection)


def classify_exit(gross: Decimal, *, fee_known: bool, fees: Decimal = Decimal("0")):
    gross = Decimal(gross)
    if fee_known:
        net = gross - Decimal(fees)
        return ExitClassification(net, net < 0)
    return ExitClassification(None, gross < 0, True)


def initial_stop(entry_price, atr, *, later_atr=None):
    """Freeze the entry-decision ATR; a later ATR never moves the first stop."""
    value = Decimal(entry_price) - Decimal("2.5") * Decimal(atr)
    return value if value > 0 else None


__all__ = ["Action", "Position", "PositionState", "Protection", "Signal",
           "classify_exit", "decide_action", "initial_stop", "validate_position"]
