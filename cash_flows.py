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


def record(delta, at: datetime, note: str = "") -> None:
    entry = {"delta": str(Decimal(str(delta))), "at": at.isoformat(), "note": note}
    storage.update_doc(_KEY, lambda entries: [*(entries or []), entry])


def flows() -> list[dict]:
    return list(storage.read_doc(_KEY) or [])


def flow_days() -> list[date]:
    return sorted({datetime.fromisoformat(entry["at"]).date() for entry in flows()})
