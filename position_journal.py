"""Gerçek işlem günlüğü (spec 0003).

Günlük kalıcı kayıt deposunda (`storage`) tutulur; yerel dosyaya yazılmaz, çünkü
yayın ortamının diski kalıcı değildir (Q13 / spec 0004). Okuma başarısız olursa
`StorageAccessError` yükselir: boş günlükle devam etmek, bir sonraki yazmada
gerçek işlemlerin üzerine yazılmasına yol açardı.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

import storage
from storage import StorageAccessError


@dataclass(frozen=True)
class Trade:
    event_id: str
    side: str
    quantity: Decimal
    price: Decimal
    executed_at: datetime
    recorded_at: datetime
    fee: Decimal | None = None
    symbol: str = ""


@dataclass(frozen=True)
class RealPosition:
    quantity: Decimal
    entry_price: Decimal
    executed_at: datetime


@dataclass(frozen=True)
class PaperTrade:
    side: str
    quantity: Decimal
    price: Decimal
    spread_bps: Decimal | None


@dataclass(frozen=True)
class LegacyRecord:
    record_id: str
    status: str


class PositionJournal:
    def __init__(self, key=storage.JOURNAL_KEY):
        self.key = key
        self.trades = []
        self.paper_trades = []
        self.legacy_records = []
        self.position = None
        self.positions = {}
        self.cooldown_bars = 0
        self.verified_real_pnl = Decimal("0")
        self._paper_spread = None
        payload = storage.read_doc(self.key)  # erişilemezse StorageAccessError
        if payload is not None:
            try:
                self._load(payload)
            except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
                raise StorageAccessError("İşlem günlüğü okunamadı.") from exc

    def apply_signal(self, signal):
        return False

    def confirm_trade(self, event_id, side, quantity, price, executed_at, recorded_at, *, fee=None, symbol=""):
        quantity, price = Decimal(quantity), Decimal(price)
        if quantity <= 0 or price <= 0 or not event_id or any(t.event_id == event_id for t in self.trades):
            return False
        trade = Trade(event_id, side, quantity, price, executed_at, recorded_at,
                      None if fee is None else Decimal(fee), str(symbol))
        snapshot = self._snapshot()
        self.trades.append(trade)
        if side == "BUY":
            self.positions[trade.symbol] = RealPosition(quantity, price, executed_at)
        elif side == "SELL":
            self.positions.pop(trade.symbol, None)
        self.position = self.positions.get("")
        self._save_or_restore(snapshot)
        return True

    def has_event(self, event_id):
        return any(trade.event_id == event_id for trade in self.trades)

    def correct_trade(self, event_id, **changes):
        for index, trade in enumerate(self.trades):
            if trade.event_id == event_id:
                normalized = {key: Decimal(value) if key in {"quantity", "price", "fee"} else value
                              for key, value in changes.items()}
                snapshot = self._snapshot()
                self.trades[index] = replace(trade, **normalized)
                self._rebuild_positions()
                self._save_or_restore(snapshot)
                return True
        return False

    def record_paper(self, side, quantity, price, *, spread_bps=None):
        selected = self._paper_spread if spread_bps is None else Decimal(spread_bps)
        self.paper_trades.append(PaperTrade(side, Decimal(quantity), Decimal(price), selected))

    def ignore_signal(self, signal):
        return False

    def add_fee(self, event_id, fee):
        return self.correct_trade(event_id, fee=fee)

    def set_cooldown(self, bars):
        snapshot = self._snapshot()
        self.cooldown_bars = int(bars)
        self._save_or_restore(snapshot)

    def set_paper_quantity_step(self, step):
        self._paper_quantity_step = Decimal(step)

    def set_paper_costs(self, *, spread_bps=None):
        self._paper_spread = None if spread_bps is None else Decimal(spread_bps)

    def import_legacy(self, records):
        self.legacy_records.extend(
            LegacyRecord(str(item.get("id", "")), "ESKI_TEYITSIZ") for item in records
        )

    def _snapshot(self):
        return (list(self.trades), dict(self.positions), self.position, self.cooldown_bars)

    def _save_or_restore(self, snapshot):
        """Yazma başarısızsa bellek son kayıtlı duruma döner ve hata yükselir."""
        try:
            self._save()
        except StorageAccessError:
            self.trades, self.positions, self.position, self.cooldown_bars = snapshot
            raise

    def _save(self):
        payload = {
            "trades": [{
                "event_id": t.event_id, "side": t.side, "quantity": str(t.quantity),
                "price": str(t.price), "executed_at": t.executed_at.isoformat(),
                "recorded_at": t.recorded_at.isoformat(),
                "fee": None if t.fee is None else str(t.fee),
                "symbol": t.symbol,
            } for t in self.trades],
            "cooldown_bars": self.cooldown_bars,
        }
        storage.write_doc(self.key, payload)

    def _load(self, payload):
        self.trades = [Trade(
            item["event_id"], item["side"], Decimal(item["quantity"]), Decimal(item["price"]),
            datetime.fromisoformat(item["executed_at"]), datetime.fromisoformat(item["recorded_at"]),
            None if item["fee"] is None else Decimal(item["fee"]),
            item.get("symbol", ""),
        ) for item in payload.get("trades", [])]
        self.cooldown_bars = int(payload.get("cooldown_bars", 0))
        self._rebuild_positions()

    def _rebuild_positions(self):
        self.positions = {}
        for trade in sorted(self.trades, key=lambda item: (item.executed_at, item.recorded_at)):
            if trade.side == "BUY":
                self.positions[trade.symbol] = RealPosition(trade.quantity, trade.price, trade.executed_at)
            elif trade.side == "SELL":
                self.positions.pop(trade.symbol, None)
        self.position = self.positions.get("")
