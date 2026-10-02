"""Sembolden piyasa ve para birimi çıkarımı (spec 0005, Adım 4).

Varlık kayıtlarında piyasa/para birimi alanı yoktur; tek kaynak bu açık
kurallardır. Tanınmayan sembol `None` döner — çağıran taraf onu sessizce bir
piyasaya koymaz.
"""
from __future__ import annotations

import re

_US_TICKER = re.compile(r"^[A-Z]{1,5}$")


def market_of(symbol: str) -> tuple[str, str] | None:
    """`(piyasa, para_birimi)`; tanınmıyorsa `None`."""
    text = str(symbol or "").strip().upper()
    if not text:
        return None
    if text == "XAU_GOLD":
        return ("ALTIN", "USD")
    if text == "GRAM_TRY":
        return ("ALTIN", "TRY")
    if text.endswith(".IS"):
        return ("BIST", "TRY")
    if text.endswith("-USD"):
        return ("KRIPTO", "USD")
    if text.endswith("-USDT"):
        return ("KRIPTO", "USDT")
    if _US_TICKER.match(text):
        return ("ABD", "USD")
    return None
