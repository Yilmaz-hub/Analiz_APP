"""Gerçek işlem teyidi (spec 0003, Q3): alış ve satış kayıtları.

Kullanıcı kurumda gerçekleşen işlemi kendi zamanı, miktarı ve fiyatıyla teyit
eder. Bu modül formun arkasındaki kuralları tek yerde tutar:

* **Doğrulama önce.** Bakiye ve pozisyon, işlem doğrulanmadan değişmez.
* **Kimlik işlemden türer.** `event_id`; varlık, yön, miktar, fiyat ve UTC işlem
  zamanından üretilir. Aynı teyit iki kez girilirse aynı kimliği alır ve tekrar
  olarak reddedilir; farklı işlem (farklı miktar/fiyat/zaman) ayrı kimlik alır.
* **Sıra: önce portföy, sonra günlük.** Arada bir yazma koparsa yarım kalan işlem
  `reconcile` ile tamamlanır; ikisinde de varsa gerçek tekrardır.
"""
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from storage import StorageAccessError

ISTANBUL = ZoneInfo("Europe/Istanbul")


@dataclass(frozen=True)
class Outcome:
    ok: bool
    code: str = ""
    event_id: str = ""


def _plain(value):
    return format(Decimal(value).normalize(), "f")


def make_event_id(symbol, side, quantity, price, executed_at):
    """İşlemin kendi alanlarından türeyen kimlik (saat dilimi fark etmez)."""
    stamp = executed_at.astimezone(timezone.utc).isoformat()
    digest = hashlib.sha256(
        f"{symbol}|{side}|{_plain(quantity)}|{_plain(price)}|{stamp}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{symbol}:{side}:{digest}"


def istanbul_to_utc(day, clock):
    """Kullanıcının girdiği İstanbul tarih/saatini UTC'ye çevirir."""
    return datetime.combine(day, clock, tzinfo=ISTANBUL).astimezone(timezone.utc)


def _decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def validate_buy(*, quantity, price, stop, executed_at, now, balance=None):
    """Alış teyidinin geçersizlik kodu; geçerliyse None. `balance` verilirse nakit denetlenir."""
    quantity, price, stop = _decimal(quantity), _decimal(price), _decimal(stop)
    if quantity is None or quantity <= 0:
        return "GECERSIZ_MIKTAR"
    if price is None or price <= 0:
        return "GECERSIZ_FIYAT"
    if executed_at > now:
        return "GELECEK_ZAMAN"
    if stop is None or stop <= 0 or stop >= price:
        return "GECERSIZ_STOP"
    if balance is not None and quantity * price > Decimal(str(balance)):
        return "YETERSIZ_BAKIYE"
    return None


def validate_sell(*, position, quantity, price, executed_at, now):
    """Satış teyidinin geçersizlik kodu; V1'de satış pozisyonun tamamını kapatır (R03)."""
    quantity, price = _decimal(quantity), _decimal(price)
    if quantity is None or quantity <= 0:
        return "GECERSIZ_MIKTAR"
    if price is None or price <= 0:
        return "GECERSIZ_FIYAT"
    if executed_at > now:
        return "GELECEK_ZAMAN"
    # Eski kayıtlarda kesirli adet uzun olabilir (1000/45000); kurum 8 haneye yuvarlar,
    # bu yüzden karşılaştırma 8 haneye yuvarlanarak yapılır (Y5).
    step = Decimal("0.00000001")
    held = _decimal(position.get("Adet"))
    if held is None or quantity.quantize(step) != held.quantize(step):
        return "MIKTAR_POZISYONLA_ESIT_DEGIL"
    entry = position.get("Gerçekleşme Zamanı")
    if entry and executed_at < datetime.fromisoformat(entry):
        return "CIKIS_GIRISTEN_ONCE"
    return None


def _journal_has(journal, event_id):
    return any(trade.event_id == event_id for trade in journal.trades)


def _portfolio_has(portfolio, event_id, key):
    return any(item.get(key) == event_id for item in portfolio.get("positions", []))


def confirm_buy(portfolio, journal, save, *, coin, symbol, quantity, price, stop,
                executed_at, now, use_balance=True, is_limit=False, tarih=""):
    """Alışı doğrular, önce portföye, sonra günlüğe yazar.

    `save()` portföyü kalıcılaştırıp başarıyı döner (başarısızsa bellekteki
    değişikliği kendisi geri alır).
    """
    balance = portfolio.get("balance", 0.0) if use_balance else None
    code = validate_buy(quantity=quantity, price=price, stop=stop, executed_at=executed_at,
                        now=now, balance=balance)
    if code:
        return Outcome(False, code)
    quantity, price = Decimal(str(quantity)), Decimal(str(price))
    event_id = make_event_id(symbol, "BUY", quantity, price, executed_at)
    in_portfolio = _portfolio_has(portfolio, event_id, "JournalEventId")
    in_journal = _journal_has(journal, event_id)
    if in_portfolio and (in_journal or is_limit):
        return Outcome(False, "TEKRAR_TEYIT", event_id)

    if not in_portfolio:
        invested = float(quantity * price)
        if use_balance:
            portfolio["balance"] = float(Decimal(str(portfolio.get("balance", 0.0))) - Decimal(str(invested)))
        portfolio.setdefault("positions", []).append({
            "Coin": coin, "Giriş": float(price), "Adet": float(quantity), "Yatırım": invested,
            "Realized": 0.0, "Status": "PENDING" if is_limit else "ACTIVE",
            "Tarih": tarih or executed_at.astimezone(ISTANBUL).strftime("%Y-%m-%d"),
            "Stop": float(Decimal(str(stop))), "Gerçekleşme Zamanı": executed_at.isoformat(),
            "V1Verified": not is_limit, "JournalEventId": event_id,
        })
        if not save():
            return Outcome(False, "KAYIT_YAZILAMADI", event_id)

    if not is_limit and not in_journal:
        try:
            journal.confirm_trade(event_id, "BUY", quantity, price, executed_at,
                                  datetime.now(timezone.utc), fee=None, symbol=symbol)
        except StorageAccessError:
            return Outcome(False, "GUNLUK_YAZILAMADI", event_id)
    return Outcome(True, "", event_id)


def confirm_sell(portfolio, journal, save, *, position, symbol, quantity, price,
                 executed_at, now):
    """Satışı doğrular; pozisyonu önce portföyde kapatır, sonra günlüğe yazar."""
    code = validate_sell(position=position, quantity=quantity, price=price,
                         executed_at=executed_at, now=now)
    if code:
        return Outcome(False, code)
    quantity, price = Decimal(str(quantity)), Decimal(str(price))
    event_id = make_event_id(symbol, "SELL", quantity, price, executed_at)
    if _portfolio_has(portfolio, event_id, "JournalExitEventId"):
        return Outcome(False, "TEKRAR_TEYIT", event_id)

    proceeds = quantity * price
    cost_basis = Decimal(str(position.get("Giriş", 0.0))) * quantity
    portfolio["balance"] = float(Decimal(str(portfolio.get("balance", 0.0))) + proceeds)
    position["Adet"] = 0.0
    position["Yatırım"] = float(Decimal(str(position.get("Yatırım", 0.0))) - cost_basis)
    position["Realized"] = float(Decimal(str(position.get("Realized", 0.0))) + proceeds - cost_basis)
    position["Status"] = "CLOSED_CONFIRMED"
    position["Gerçekleşen Çıkış"] = float(price)
    position["Çıkış Adedi"] = float(quantity)
    position["Çıkış Zamanı"] = executed_at.isoformat()
    position["JournalExitEventId"] = event_id
    if not save():
        return Outcome(False, "KAYIT_YAZILAMADI", event_id)

    if not _journal_has(journal, event_id):
        try:
            journal.confirm_trade(event_id, "SELL", quantity, price, executed_at,
                                  datetime.now(timezone.utc), fee=None, symbol=symbol)
        except StorageAccessError:
            return Outcome(False, "GUNLUK_YAZILAMADI", event_id)
    return Outcome(True, "", event_id)


def reconcile(portfolio, journal, coin_map):
    """Portföyde teyitli ama günlükte olmayan işlemleri günlüğe tamamlar.

    Portföy yazıldıktan sonra günlük yazımı koparsa kalan yarım işlemin
    tamamlanma yoludur. Tamamlanan işlem sayısını döner; erişim hatasında
    StorageAccessError yükselir.
    """
    completed = 0
    for item in portfolio.get("positions", []):
        if not item.get("V1Verified") or item.get("Status") not in ("ACTIVE", "CLOSED_CONFIRMED"):
            continue
        symbol = coin_map.get(item.get("Coin"), item.get("Coin"))
        buy_id = item.get("JournalEventId")
        if buy_id and not _journal_has(journal, buy_id) and item.get("Gerçekleşme Zamanı"):
            quantity = item.get("Çıkış Adedi", item.get("Adet"))
            if _decimal(quantity) and _decimal(quantity) > 0:
                journal.confirm_trade(
                    buy_id, "BUY", Decimal(str(quantity)), Decimal(str(item["Giriş"])),
                    datetime.fromisoformat(item["Gerçekleşme Zamanı"]),
                    datetime.now(timezone.utc), fee=None, symbol=symbol)
                completed += 1
        exit_id = item.get("JournalExitEventId")
        if exit_id and not _journal_has(journal, exit_id):
            journal.confirm_trade(
                exit_id, "SELL", Decimal(str(item["Çıkış Adedi"])),
                Decimal(str(item["Gerçekleşen Çıkış"])),
                datetime.fromisoformat(item["Çıkış Zamanı"]),
                datetime.now(timezone.utc), fee=None, symbol=symbol)
            completed += 1
    return completed
