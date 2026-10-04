"""Sembolden piyasa ve para birimi çıkarımı (spec 0005, Adım 4).

Varlık kayıtlarında piyasa/para birimi alanı yoktur; tek kaynak bu açık
kurallardır. Tanınmayan sembol `None` döner — çağıran taraf onu sessizce bir
piyasaya koymaz.
"""
from __future__ import annotations

import re

_US_TICKER = re.compile(r"^[A-Z]{1,5}$")
# Tire olmadan yazılan kripto çifti (`LINKUSD`, `HBARUSDT`): en az iki karakterlik taban.
# ISO 4217 döviz ve değerli maden kodları kripto tabanı değildir (`GBPUSD` bir döviz çiftidir).
# Kripto ile çakışan kodlar (ör. MNT = Mantle) bilerek listede yoktur.
_NON_CRYPTO_BASES = frozenset({
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", "BAM", "BBD", "BDT",
    "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", "BTN", "BWP", "BYN", "BZD", "CAD",
    "CDF", "CHF", "CLP", "CNH", "CNY", "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP",
    "DZD", "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD", "GNF",
    "GTQ", "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR", "IQD", "IRR", "ISK", "JMD",
    "JOD", "JPY", "KES", "KGS", "KHR", "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP",
    "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MOP", "MRU", "MUR", "MVR",
    "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN",
    "PGK", "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR",
    "SDG", "SEK", "SGD", "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB",
    "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU", "UZS",
    "VES", "VND", "VUV", "WST", "XAF", "XAG", "XAU", "XCD", "XOF", "XPD", "XPF", "XPT", "YER",
    "ZAR", "ZMW", "ZWL"
})
_DASHLESS_CRYPTO = re.compile(r"^([A-Z0-9]{2,12}?)(USDT|USD)$")


def canonical_symbol(symbol: str) -> str:
    """Uygulamanın her yerinde kullanılan tek sembol yorumu (spec 0006 Q07/Q08).

    Tire olmadan yazılan kripto çiftleri (`LINKUSD`) varsayılan listedeki biçime
    (`LINK-USD`) çevrilir. Başka hiçbir sembol değişmez (`AAPL`, `THYAO.IS`,
    `EURUSD=X`, `XAU_GOLD`, `GRAM_TRY`). Depodaki kayıt yeniden yazılmaz; çeviri
    yalnız okuma anında yapılır.
    """
    text = str(symbol or "").strip().upper()
    if not text or any(mark in text for mark in "-/=._"):
        return text
    match = _DASHLESS_CRYPTO.match(text)
    if match is None or match.group(1) in _NON_CRYPTO_BASES:
        return text
    return f"{match.group(1)}-{match.group(2)}"


def crypto_parts(symbol: str) -> tuple[str, str] | None:
    """`(taban, kote)` — kripto değilse `None`. Örn. `LINKUSD` → `("LINK", "USD")`."""
    text = canonical_symbol(symbol)
    for quote in ("-USDT", "-USD"):
        if text.endswith(quote) and len(text) > len(quote):
            return text[:-len(quote)], quote[1:]
    return None


def binance_symbol(symbol: str) -> str | None:
    """Binance çifti (`LINKUSDT`); kripto değilse `None`. USD çiftleri USDT'ye bağlanır."""
    parts = crypto_parts(symbol)
    return None if parts is None else f"{parts[0]}USDT"


def okx_symbol(symbol: str) -> str | None:
    """OKX enstrüman kimliği (`LINK-USDT`); kripto değilse `None`."""
    parts = crypto_parts(symbol)
    return None if parts is None else f"{parts[0]}-USDT"


def market_of(symbol: str) -> tuple[str, str] | None:
    """`(piyasa, para_birimi)`; tanınmıyorsa `None`."""
    text = canonical_symbol(symbol)
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
