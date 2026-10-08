"""Geçmiş test teşhisi: giriş, kâr alma, stop ve "girmeme" kararlarının ayrı ölçümü (spec 0016).

Hesap yalnız mevcut geçmiş test sonucundan ve fiyat verisinden türetilir; strateji kuralı değişmez.
Para/yüzde Decimal ile hesaplanır; gösterim yuvarlaması yalnız arayüzdedir.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_HUNDRED = Decimal("100")


@dataclass(frozen=True)
class Diagnostics:
    closed: int
    exits: dict                      # çıkış nedeni → adet (STOP, SAT, ...)
    went_green_closed_red: int       # kâra geçip zararla kapanan işlem
    avg_best_pct: Decimal | None     # işlem süresince görülen en yüksek seviye, girişe göre ort. %
    avg_realized_pct: Decimal | None # kapanış, girişe göre ort. %
    in_market_pct: Decimal | None    # pozisyondayken varlığın bileşik getirisi
    out_market_pct: Decimal | None   # dışarıdayken varlığın bileşik getirisi
    days_in: int
    days_out: int


def _day(value):
    return value.date() if hasattr(value, "date") else value


def _dec(value) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return number if number.is_finite() else None


def diagnose(frame, backtest: dict) -> Diagnostics | None:
    """Geçmiş test sonucunu dört soruya göre ölçer; veri yetersiz/geçersizse None."""
    if frame is None or len(frame) < 2 or not isinstance(backtest, dict) or "trades" not in backtest:
        return None
    days = [_day(ts) for ts in frame.index]
    highs = [_dec(v) for v in frame["High"]]
    closes = [_dec(v) for v in frame["Close"]]

    exits: dict = {}
    best, realized, green_red = [], [], 0
    held = [False] * len(days)
    for trade in backtest["trades"]:
        reason = str(trade.get("reason", "?"))
        exits[reason] = exits.get(reason, 0) + 1
        entry, exit_ = _dec(trade.get("entry")), _dec(trade.get("exit"))
        start, end = _day(trade["entry_at"]), _day(trade["exit_at"])
        span = [i for i, d in enumerate(days) if start <= d <= end]
        for i in span:
            held[i] = True
        window = [highs[i] for i in span if highs[i] is not None]
        if entry is None or entry <= 0 or exit_ is None or not window:
            continue
        best_pct = (max(window) / entry - 1) * _HUNDRED
        real_pct = (exit_ / entry - 1) * _HUNDRED
        best.append(best_pct)
        realized.append(real_pct)
        if best_pct > 0 and real_pct < 0:
            green_red += 1
    position = backtest.get("position")
    if position:
        start = _day(position["entry_at"])
        for i, d in enumerate(days):
            if d >= start:
                held[i] = True

    grow = {True: Decimal("1"), False: Decimal("1")}
    count = {True: 0, False: 0}
    for i in range(1, len(days)):
        prev, cur = closes[i - 1], closes[i]
        if prev is None or cur is None or prev <= 0:
            continue
        grow[held[i]] *= cur / prev
        count[held[i]] += 1

    def pct(flag):
        return (grow[flag] - 1) * _HUNDRED if count[flag] else None

    def mean(values):
        return sum(values, Decimal("0")) / len(values) if values else None

    return Diagnostics(len(backtest["trades"]), exits, green_red, mean(best), mean(realized),
                       pct(True), pct(False), count[True], count[False])


def explain(diag: Diagnostics) -> list[str]:
    """Teşhisin düz Türkçe yorumu; sayı yoksa yorum uydurulmaz."""
    lines = []
    if diag.closed == 0:
        return ["Kapanmış işlem yok; giriş ve çıkış kalitesi ölçülemedi."]
    stop, sat = diag.exits.get("STOP", 0), diag.exits.get("SAT", 0)
    lines.append(f"Çıkışlar: {stop} stop, {sat} SAT sinyali"
                 + (f", {diag.closed - stop - sat} diğer." if diag.closed - stop - sat else "."))
    if diag.avg_best_pct is not None and diag.avg_realized_pct is not None:
        given = diag.avg_best_pct - diag.avg_realized_pct
        lines.append(f"İşlemler ortalama %{diag.avg_best_pct:.2f} yükseğe çıktı ama ortalama "
                     f"%{diag.avg_realized_pct:.2f} ile kapandı: işlem başına %{given:.2f} geri verildi. "
                     "Kâr alma kuralı olmadığı için yükseliş korunamıyor olabilir.")
    if diag.went_green_closed_red:
        lines.append(f"{diag.went_green_closed_red} / {diag.closed} işlem kâra geçtikten sonra zararla kapandı.")
    if diag.out_market_pct is not None:
        if diag.out_market_pct < 0:
            lines.append(f"Dışarıda kalınan {diag.days_out} günde varlık %{diag.out_market_pct:.2f} değişti: "
                         "girmemek bu düşüşten korudu.")
        else:
            lines.append(f"Dışarıda kalınan {diag.days_out} günde varlık %{diag.out_market_pct:.2f} değişti: "
                         "girmemek bu yükselişi kaçırdı.")
    return lines
