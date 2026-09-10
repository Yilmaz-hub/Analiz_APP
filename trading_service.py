from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from trade_decisions import Action, Position, Signal, decide_action, initial_stop


@dataclass(frozen=True)
class TradingContext:
    signal: Signal
    position: Position
    data: object
    components: dict
    cutoff: datetime
    settings_version: str

    @property
    def identity(self):
        return (self.cutoff.astimezone(timezone.utc).isoformat(), self.signal.value,
                self.position.state.value, self.settings_version)


@dataclass
class TradingSettings:
    capital: Decimal = Decimal("10000")
    trade_notional: Decimal = Decimal("1000")
    costs: dict = field(default_factory=dict)

    def cost_for(self, market):
        return self.costs.get(market)


class PaperAccounts:
    def __init__(self, default_capital):
        self._default = Decimal(default_capital)
        self._balances = {}

    def debit(self, symbol, amount):
        self._balances[symbol] = self.balance(symbol) - Decimal(amount)

    def balance(self, symbol):
        return self._balances.get(symbol, self._default)


class DecisionLedger:
    def __init__(self):
        self._seen = set()

    def record(self, identity):
        if identity in self._seen:
            return False
        self._seen.add(identity)
        return True


@dataclass(frozen=True)
class ModeResult:
    action: Action
    stop: Decimal | None
    target_enabled: bool
    net: Decimal | None
    comparable: bool


def evaluate_all_modes(context, *, entry=None, atr=None, gross=None, fees=None):
    ready = bool(getattr(context.data, "is_valid", False)) and all(
        value in ("READY", "NEUTRAL") for value in context.components.values()
    )
    decision = decide_action(context.signal, context.position) if ready else None
    stop = initial_stop(entry, atr) if entry is not None and atr is not None else None
    net = Decimal(gross) - Decimal(fees) if gross is not None and fees is not None else None
    result = ModeResult(decision.action if decision else Action.NONE, stop, False, net, ready)
    return {mode: result for mode in ("screen", "backtest", "paper")}


def point_in_time_value(observations, cutoff):
    known = [value for observed_at, value in observations if observed_at <= cutoff]
    return known[-1] if known else None


@dataclass(frozen=True)
class ProviderResult:
    status: str
    is_success: bool
    action: object = None


def evaluate_provider(operation):
    try:
        return ProviderResult("BASARILI", True, operation())
    except TimeoutError:
        return ProviderResult("SAGLAYICI_ZAMAN_ASIMI", False)
    except Exception:
        return ProviderResult("SAGLAYICI_HATASI", False)
