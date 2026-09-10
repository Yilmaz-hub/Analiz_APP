from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class Signal(str, Enum):
    BUY = "AL"
    WAIT = "BEKLE"
    SELL = "SAT"


class Action(str, Enum):
    BUY = "AL"
    HOLD = "TUT"
    SELL = "SAT"
    NONE = "EYLEM_YOK"


class PositionState(str, Enum):
    UNKNOWN = "BILINMIYOR"
    FLAT = "YOK"
    OPEN = "ACIK"
    INVALID = "GECERSIZ"


@dataclass(frozen=True)
class Position:
    state: PositionState
    quantity: Decimal | None = None
    entry_price: Decimal | None = None
    executed_at: datetime | None = None
    stop: Decimal | None = None

    @classmethod
    def flat(cls):
        return cls(PositionState.FLAT)


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str = ""
    active_stop: Decimal | None = None
    exit_count: int = 0
    protection: object | None = None


@dataclass(frozen=True)
class ExitClassification:
    net: Decimal | None
    cooldown_required: bool
    provisional: bool = False
