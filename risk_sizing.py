"""Risk bazlı miktar, toplam risk ve dönem kayıp sınırı (spec 0005, Adım 6 / S4).

Saf hesap + kalıcı ayar. Para ve yüzde `Decimal`dir; yuvarlama yalnız **aşağı**
yapılır (R10). Doğrulanamayan durumda miktar ya da "geçer" sonucu üretilmez,
nedeni döner (R12):

* miktar = aşağı_yuvarla(bütçe / (giriş − stop + birim maliyet), miktar adımı);
  nakit ve asgari işlem tutarı ayrıca sınırlar.
* Risk oranı yüzde ölçeğinde (iki ondalık, `conventions.md`) temsil edilir;
  0 < oran < 100, ölçekte sıfıra yuvarlanan girdi reddedilir (Q04).
* Toplam risk ve kayıp sınırı yalnız tek para biriminde hesaplanır; sınır dahildir.
* Kayıp dönemi kendiliğinden sıfırlanmaz; sıfırlama kayda geçen kullanıcı eylemidir.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Sequence

import storage

_PROFILE_KEY = "risk_profile"
_LOSS_KEY = "loss_period"
_MIN_KEY = "asset_minimums"
_PCT_SCALE = Decimal("0.01")
_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


@dataclass(frozen=True)
class SizeResult:
    quantity: Decimal | None
    reason: str = ""


@dataclass(frozen=True)
class RiskProfile:
    per_trade_pct: Decimal
    total_pct: Decimal


@dataclass(frozen=True)
class Gate:
    allowed: bool
    verified: bool
    reason: str = ""


def parse_ratio(value) -> tuple[Decimal | None, str]:
    """Yüzde oranını doğrular; `(oran, "")` ya da `(None, neden)`."""
    try:
        ratio = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, ValueError):
        return None, "Risk oranı sayı olmalı."
    if not ratio.is_finite() or ratio <= 0:
        return None, "Risk oranı 0'dan büyük olmalı."
    if ratio >= _HUNDRED:
        return None, "Risk oranı %100'den küçük olmalı."
    scaled = ratio.quantize(_PCT_SCALE, rounding=ROUND_DOWN)
    if scaled <= 0:
        return None, "Risk oranı yüzde ölçek sınırının (iki ondalık) altında kalıp sıfıra düşüyor; daha büyük bir oran girin."
    return scaled, ""


def save_profile(*, per_trade_pct: Decimal, total_pct: Decimal) -> None:
    for value in (per_trade_pct, total_pct):
        if parse_ratio(value)[0] is None:
            raise ValueError(parse_ratio(value)[1])
    storage.write_doc(_PROFILE_KEY, {"per_trade_pct": str(parse_ratio(per_trade_pct)[0]),
                                     "total_pct": str(parse_ratio(total_pct)[0])})


def load_profile() -> RiskProfile | None:
    payload = storage.read_doc(_PROFILE_KEY)
    if not payload:
        return None
    return RiskProfile(Decimal(payload["per_trade_pct"]), Decimal(payload["total_pct"]))


def size(budget: Decimal, entry: Decimal, stop: Decimal, *, unit_cost: Decimal = _ZERO,
         quantity_step: Decimal | None, cash: Decimal,
         minimum_notional: Decimal | None = None) -> SizeResult:
    if stop <= 0:
        return SizeResult(None, "Stop 0'dan büyük olmalı; miktar hesaplanmadı.")
    if stop >= entry:
        return SizeResult(None, "Stop girişin altında olmalı; miktar hesaplanmadı.")
    if quantity_step is None or quantity_step <= 0:
        return SizeResult(None, "Varlığın miktar adımı bilinmiyor; miktar önerilmez.")
    if budget <= 0 or cash <= 0:
        return SizeResult(None, "Sermaye veya nakit bilgisi eksik; miktar hesaplanmadı.")
    per_unit_risk = entry - stop + unit_cost
    by_risk = budget / per_unit_risk
    by_cash = cash / (entry + unit_cost)
    raw = min(by_risk, by_cash)
    quantity = (raw / quantity_step).to_integral_value(rounding=ROUND_FLOOR) * quantity_step
    if quantity <= 0:
        return SizeResult(None, "Hesaplanan miktar ürünün miktar adımının altında; miktar önerilmez.")
    if minimum_notional is not None and quantity * entry < minimum_notional:
        return SizeResult(None, "Hesaplanan tutar ürünün asgari işlem tutarının altında; miktar önerilmez.")
    return SizeResult(quantity)


def suggest(*, capital: Decimal, entry: Decimal, stop: Decimal, quantity_step: Decimal | None,
            cash: Decimal, unit_cost: Decimal = _ZERO,
            minimum_notional: Decimal | None = None) -> SizeResult:
    """Kayıtlı profile göre miktar; profil yoksa miktar yok (varsayılan oran atanmaz)."""
    profile = load_profile()
    if profile is None:
        return SizeResult(None, "Risk profili belirlenmedi: işlem başına ve toplam risk oranını "
                                "seçmeden miktar önerilmez.")
    if capital <= 0:
        return SizeResult(None, "Sermaye bilgisi eksik; miktar hesaplanmadı.")
    budget = capital * profile.per_trade_pct / _HUNDRED
    return size(budget, entry, stop, unit_cost=unit_cost, quantity_step=quantity_step,
                cash=cash, minimum_notional=minimum_notional)


def position_risk(entry: Decimal, stop: Decimal, quantity: Decimal) -> Decimal:
    """Açık pozisyonun stopa kadar riski; stop girişin üstündeyse 0 (negatif sayılmaz)."""
    return max(_ZERO, entry - stop) * quantity


def total_risk_check(limit: Decimal, open_risks: Sequence[tuple[str, Decimal]],
                     new_risk: Decimal, currency: str) -> Gate:
    currencies = {cur for cur, _ in open_risks} | {currency}
    if len(currencies) > 1:
        return Gate(False, False, "Açık pozisyonlarda birden fazla para birimi var; toplam risk "
                                  "doğrulanamadı, yeni giriş önerilmez.")
    total = sum((risk for _, risk in open_risks), _ZERO) + new_risk
    if total > limit:
        return Gate(False, True, f"Toplam risk {total} sınırı ({limit}) aşıyor; yeni giriş engellendi.")
    return Gate(True, True)


def set_loss_limit(limit: Decimal, currency: str, started_at: datetime) -> None:
    """Kayıp sınırını belirler.

    Açık bir dönem yokken yeni dönem başlatır. Açık dönem varken dönemi ve sıfırlama
    kayıtlarını **korur**: aynı değer yeniden kaydedilirse hiçbir şey değişmez, farklıysa
    değişiklik ayrı bir kayıt olarak eklenir (dönemi yalnız `reset_loss_period` yeniler;
    spec 0005 R11, AC94)."""
    def apply(period):
        if period is None:
            return {"limit": str(limit), "currency": currency,
                    "started_at": started_at.isoformat(), "history": []}
        if Decimal(period["limit"]) == limit and period["currency"] == currency:
            return period
        history = [*period["history"], {"action": "SINIR_DEGISTI", "at": started_at.isoformat(),
                                        "old": period["limit"], "new": str(limit),
                                        "currency": currency}]
        return {**period, "limit": str(limit), "currency": currency, "history": history}

    storage.update_doc(_LOSS_KEY, apply)


def loss_period() -> dict | None:
    return storage.read_doc(_LOSS_KEY)


def reset_loss_period(at: datetime) -> None:
    """Yalnız kullanıcının açık eylemiyle çağrılır; kapanan dönem kayıtta kalır."""
    def apply(period):
        if period is None:
            return None
        history = [*period["history"], {"action": "SIFIRLAMA", "at": at.isoformat(),
                                        "closed_period_start": period["started_at"]}]
        return {**period, "history": history, "started_at": at.isoformat()}

    if loss_period() is not None:
        storage.update_doc(_LOSS_KEY, apply)


def loss_gate(period_results: Sequence[tuple[str, Decimal]]) -> Gate:
    """Dönem içi gerçekleşmiş sonuçlarla yeni girişe izin; kayıp sınıra eşitse engel."""
    period = loss_period()
    if period is None:
        return Gate(True, True)
    currencies = {cur for cur, _ in period_results} | {period["currency"]}
    if len(currencies) > 1:
        return Gate(False, False, "Dönem sonuçlarında birden fazla para birimi var; kayıp sınırı "
                                  "doğrulanamadı, yeni giriş önerilmez.")
    net = sum((value for _, value in period_results), _ZERO)
    loss = -net if net < 0 else _ZERO
    if loss >= Decimal(period["limit"]):
        return Gate(False, True, f"Dönem kaybı {loss} sınıra ({period['limit']}) ulaştı; yeni giriş "
                                 "önerilmez. Açık pozisyonların stop ve SAT bilgisi geçerlidir.")
    return Gate(True, True)


def parse_min_notional(value) -> tuple[Decimal | None, str]:
    """Asgari işlem tutarı girdisi; boş = yok. `(tutar, "")` ya da `(None, neden)` (AC83)."""
    text = str(value).strip()
    if not text:
        return None, ""
    try:
        amount = Decimal(text.replace(",", "."))
    except (InvalidOperation, ValueError):
        return None, "Asgari işlem tutarı sayı olmalı."
    if not amount.is_finite() or amount <= 0:
        return None, "Asgari işlem tutarı 0'dan büyük olmalı."
    return amount, ""


def save_min_notional(asset: str, amount: Decimal | None) -> None:
    """Varlık başına kayıt; `None` kaydı kaldırır, diğer varlıklara dokunmaz."""
    def change(current):
        current = dict(current or {})
        if amount is None:
            current.pop(asset, None)
        else:
            current[asset] = str(amount)
        return current

    storage.update_doc(_MIN_KEY, change)


def load_min_notional(asset: str) -> Decimal | None:
    value = (storage.read_doc(_MIN_KEY) or {}).get(asset)
    return None if value is None else Decimal(value)
