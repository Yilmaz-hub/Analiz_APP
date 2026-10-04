"""Dış nakit hareketleri (spec 0005, AC14).

Kullanıcının bakiyeyi elle değiştirmesi, strateji performansından değil dışarıdan gelen para
hareketidir. Zamanıyla kaydedilir; değerlendirme diliminde böyle bir gün varsa strateji
karşılaştırması uygun sayılmaz.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import storage

_KEY = "external_cash_flows"


def delta(old, new) -> Decimal:
    """Yeni − eski bakiye; kayan nokta artığı olmaması için `str` üzerinden `Decimal` (Altın Kural 2)."""
    return Decimal(str(new)) - Decimal(str(old))


def record(delta, at: datetime, note: str = "", reverses: str | None = None) -> str:
    """Hareketi kaydeder ve zaman damgasını döndürür. `reverses`: geri alınan hareketin zaman damgası
    (bakiye yazılamadığında yapılan telafi kaydı); geri alınan çift para hareketi sayılmaz."""
    entry = {"delta": str(Decimal(str(delta))), "at": at.isoformat(), "note": note}
    if reverses is not None:
        entry["reverses"] = reverses
    storage.update_doc(_KEY, lambda entries: [*(entries or []), entry])
    return entry["at"]


def flows() -> list[dict]:
    return list(storage.read_doc(_KEY) or [])


def flow_days() -> list[date]:
    entries = flows()
    reversed_ats = {entry["reverses"] for entry in entries if entry.get("reverses")}
    return sorted({datetime.fromisoformat(entry["at"]).date() for entry in entries
                   if not entry.get("reverses") and entry["at"] not in reversed_ats})
