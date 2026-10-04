"""Sanal kâr koruma adayları (spec 0005, Adım 6 / S5, Q05 karar kaydı).

Üç aday **ayrı** denenir, birlikte etkinleştirilmez; R = giriş − başlangıç stopu:

* `HEDEF_2R`: fiyat giriş + 2R'ye ulaşınca tamamı satılır.
* `IZ_SUREN_2ATR`: stop = o güne kadarki en yüksek kapanış − 2 × ATR; yalnız yukarı
  taşınır, başlangıç stopunun altına inmez.
* `KADEMELI_1R`: giriş + 1R'de pozisyonun yarısı (miktar adımına aşağı) satılır,
  kalan V1 kurallarıyla (stop / SAT) çıkar.

Seviyeler gün kapanışında hesaplanır, ertesi günün açılışından itibaren geçerlidir
(≥, `trade_execution.evaluate_stop`). Her adayda V1 çıkışı korunur: stop teması ve
SAT uyarısının ertesi açılışı. Simülasyon gerçek pozisyon kaydına **yazmaz** (R13);
gerçek pozisyondan yalnız değer kopyası okunur.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Iterable, Mapping

from trade_execution import Bar, evaluate_stop

TARGET = "HEDEF_2R"
TRAILING = "IZ_SUREN_2ATR"
SCALE_OUT = "KADEMELI_1R"

TARGET_R = Decimal("2")
TRAIL_ATR = Decimal("2")
SCALE_R = Decimal("1")
SCALE_FRACTION = Decimal("0.5")
_ZERO = Decimal("0")
_TICK = timedelta(microseconds=1)


@dataclass(frozen=True)
class VirtualPosition:
    entry: Decimal
    quantity: Decimal
    stop: Decimal
    entry_at: datetime


@dataclass(frozen=True)
class VirtualFill:
    at: datetime
    quantity: Decimal
    price: Decimal
    reason: str


@dataclass(frozen=True)
class VirtualResult:
    fills: list[VirtualFill]
    open_quantity: Decimal
    closed_positions: int
    stop_path: list[Decimal] = field(default_factory=list)


def from_portfolio(position: Mapping) -> VirtualPosition:
    """Gerçek portföy kaydından salt okunur değer kopyası."""
    return VirtualPosition(
        entry=Decimal(str(position["Giriş"])), quantity=Decimal(str(position["Adet"])),
        stop=Decimal(str(position["Stop"])),
        entry_at=datetime.fromisoformat(position["Gerçekleşme Zamanı"]))


def format_quantity(quantity: Decimal, step: Decimal) -> str:
    return str(quantity.quantize(step))


class VirtualBook:
    """Sanal pozisyonun kapanan/kalan miktarı; son parça kapanınca tek kapanmış işlem."""

    def __init__(self, position: VirtualPosition, *, quantity_step: Decimal):
        self.position = position
        self.step = quantity_step
        self.open_quantity = position.quantity
        self.closed_positions = 0
        self.fills: list[VirtualFill] = []

    def sell(self, quantity: Decimal, price: Decimal, at: datetime, reason: str) -> Decimal:
        quantity = min(quantity, self.open_quantity)
        if quantity <= 0:
            return _ZERO
        self.open_quantity -= quantity
        self.fills.append(VirtualFill(at, quantity, price, reason))
        if self.open_quantity == 0:
            self.closed_positions += 1
        return quantity

    def sell_fraction(self, fraction: Decimal, price: Decimal, at: datetime, reason: str) -> Decimal:
        raw = self.open_quantity * fraction
        quantity = (raw / self.step).to_integral_value(rounding=ROUND_FLOOR) * self.step
        return self.sell(quantity, price, at, reason)

    def sell_all(self, price: Decimal, at: datetime, reason: str) -> Decimal:
        return self.sell(self.open_quantity, price, at, reason)


def simulate(position: VirtualPosition, bars: Iterable[tuple[Bar, Decimal]], rule: str, *,
             quantity_step: Decimal, sell_signals: Iterable[datetime] = ()) -> VirtualResult:
    """Girişten sonraki günlük mumlar (`(mum, ATR)`) üzerinde tek adayı sanal işletir."""
    if rule not in (TARGET, TRAILING, SCALE_OUT):
        raise ValueError(f"Bilinmeyen kâr koruma adayı: {rule}")
    book = VirtualBook(position, quantity_step=quantity_step)
    risk = position.entry - position.stop
    target = position.entry + TARGET_R * risk
    scale_level = position.entry + SCALE_R * risk
    stop, stop_active_at = position.stop, position.entry_at
    signals = set(sell_signals)
    highest: Decimal | None = None
    scaled = False
    pending_sell = False
    path: list[Decimal] = []

    for bar, atr in bars:
        if book.open_quantity == 0:
            break
        path.append(stop)
        if pending_sell:
            book.sell_all(bar.open, bar.at, "SAT")
            break
        hit = evaluate_stop(bar, stop, stop_active_at)
        if hit is not None:
            book.sell_all(hit.base_price, bar.at, "STOP")
            break
        if rule == TARGET and bar.high >= target:
            book.sell_all(bar.open if bar.open >= target else target, bar.at, "HEDEF")
            break
        if rule == SCALE_OUT and not scaled and bar.high >= scale_level:
            price = bar.open if bar.open >= scale_level else scale_level
            book.sell_fraction(SCALE_FRACTION, price, bar.at, "KADEME")
            scaled = True
        if rule == TRAILING:
            highest = bar.close if highest is None else max(highest, bar.close)
            candidate = highest - TRAIL_ATR * Decimal(atr)
            if candidate > stop:
                stop, stop_active_at = candidate, bar.at + _TICK
        if bar.at in signals:
            pending_sell = True
    return VirtualResult(book.fills, book.open_quantity, book.closed_positions, path)
