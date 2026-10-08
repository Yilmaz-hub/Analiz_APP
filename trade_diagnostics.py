"""Geçmiş test teşhisi: giriş, kâr alma, stop ve "girmeme" kararlarının ayrı ölçümü (spec 0016, 0017).

Hesap yalnız mevcut geçmiş test sonucundan ve fiyat verisinden türetilir; strateji kuralı değişmez.
Para/yüzde Decimal ile hesaplanır; gösterim yuvarlaması yalnız arayüzdedir.

Spec 0017: ortalama yerine medyan (birkaç büyük kazanç ortalamayı şişirir); "kâra geçti" yalnız
en az 1R (giriş − başlangıç stopu) kâr görüldüyse sayılır; kazanan ve kaybeden işlemler ayrı;
her işlemin girişe karar verilen günkü piyasa koşulu gösterilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

_HUNDRED = Decimal("100")
REGIME_TEXT = {"YUKSELEN": "Yükselen", "DUSEN": "Düşen", "YATAY": "Yatay", None: "belirsiz"}


@dataclass(frozen=True)
class TradeRow:
    entry_day: object
    exit_day: object
    won: bool
    reason: str
    realized_pct: Decimal
    best_pct: Decimal
    best_r: Decimal | None          # görülen en iyi kâr, R cinsinden (R bilinmiyorsa None)
    regime: str | None              # girişe karar verilen günün piyasa koşulu


@dataclass(frozen=True)
class Diagnostics:
    closed: int
    exits: dict                          # çıkış nedeni → adet (STOP, SAT, ...)
    reached_1r_closed_red: int           # en az 1R kâr görüp zararla kapanan işlem
    r_known: int                         # R'si hesaplanabilen kapanmış işlem
    median_best_pct: Decimal | None      # işlem süresince görülen en yüksek seviye, medyan %
    median_realized_pct: Decimal | None  # kapanış, girişe göre medyan %
    in_market_pct: Decimal | None        # pozisyondayken varlığın bileşik getirisi
    out_market_pct: Decimal | None       # dışarıdayken varlığın bileşik getirisi
    days_in: int
    days_out: int
    rows: list = field(default_factory=list)


def _day(value):
    return value.date() if hasattr(value, "date") else value


def _dec(value) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return number if number.is_finite() else None


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def first_tradable_pos(frame, decisions) -> int | None:
    """Stratejinin ilk alım yapabileceği mumun konumu: ilk kararın bir sonraki mumu.

    Isınma döneminde (göstergeler oluşmadan) karar yoktur; bu günler "dışarıda kalma" sayılmaz.
    Karar verilmemişse 1 (ikinci mum); karar sonrası mum yoksa None."""
    if not decisions:
        return 1
    first = min(decisions)
    later = [i for i, ts in enumerate(frame.index) if ts > first]
    return later[0] if later else None


def _risk_unit(frame, entry_pos, entry) -> Decimal | None:
    """R = giriş − başlangıç stopu; stop, girişe karar verilen günün ATR'siyle hesaplanır (V1 kuralı)."""
    if "ATR" not in frame.columns or entry_pos < 1:
        return None
    atr = _dec(frame["ATR"].iloc[entry_pos - 1])
    if atr is None:
        return None
    from trade_execution import compute_initial_stop
    try:
        stop = compute_initial_stop(entry, atr)
    except ArithmeticError:
        return None
    if stop is None or stop <= 0 or stop >= entry:
        return None
    return entry - stop


def _regime_before(frame, entry_pos, classify):
    """Girişe karar verilen günün koşulu: yalnız o güne kadarki veriyle (sızıntı yok)."""
    if classify is None or entry_pos < 1:
        return None
    try:
        return classify(frame.iloc[:entry_pos])[0]
    except Exception:  # koşul hesaplanamazsa teşhis yine çalışır
        return None


def diagnose(frame, backtest: dict, decisions=None, classify=None) -> Diagnostics | None:
    """Geçmiş test sonucunu dört soruya göre ölçer; veri yetersiz/geçersizse None.

    `classify(frame) -> (koşul, sürüm)` verilirse her işlemin giriş koşulu hesaplanır
    (uygulamada `regime_classifier.classify_latest`)."""
    if frame is None or len(frame) < 2 or not isinstance(backtest, dict) or "trades" not in backtest:
        return None
    if not {"High", "Close"} <= set(frame.columns):
        return None
    begin = first_tradable_pos(frame, decisions)
    if begin is None:
        return None
    days = [_day(ts) for ts in frame.index]
    highs = [_dec(v) for v in frame["High"]]
    closes = [_dec(v) for v in frame["Close"]]

    exits: dict = {}
    rows, red_after_1r, r_known = [], 0, 0
    held = [False] * len(days)
    for trade in backtest["trades"]:
        reason = str(trade.get("reason", "?"))
        exits[reason] = exits.get(reason, 0) + 1
        entry, exit_ = _dec(trade.get("entry")), _dec(trade.get("exit"))
        start, end = _day(trade["entry_at"]), _day(trade["exit_at"])
        span = [i for i, d in enumerate(days) if start <= d <= end]
        for i in span:
            held[i] = True
        if entry is None or entry <= 0 or exit_ is None or not span:
            continue
        # Çıkış gününün yükseği çıkıştan sonra oluşmuş olabilir (SAT ertesi açılışta, stopta gün içi sıra
        # bilinmez); o gün yalnız çıkış fiyatıyla temsil edilir.
        window = [highs[i] for i in span if days[i] < end and highs[i] is not None] + [exit_]
        peak = max(window)
        best_pct = (peak / entry - 1) * _HUNDRED
        realized_pct = (exit_ / entry - 1) * _HUNDRED
        unit = _risk_unit(frame, span[0], entry)
        best_r = None if unit is None else (peak - entry) / unit
        if best_r is not None:
            r_known += 1
            if best_r >= 1 and exit_ < entry:
                red_after_1r += 1
        rows.append(TradeRow(start, end, exit_ > entry, reason, realized_pct, best_pct, best_r,
                             _regime_before(frame, span[0], classify)))
    position = backtest.get("position")
    if position:
        start = _day(position["entry_at"])
        for i, d in enumerate(days):
            if d >= start:
                held[i] = True

    grow = {True: Decimal("1"), False: Decimal("1")}
    count = {True: 0, False: 0}
    for i in range(max(begin, 1), len(days)):
        prev, cur = closes[i - 1], closes[i]
        if prev is None or cur is None or prev <= 0:
            continue
        grow[held[i]] *= cur / prev
        count[held[i]] += 1

    def pct(flag):
        return (grow[flag] - 1) * _HUNDRED if count[flag] else None

    return Diagnostics(len(backtest["trades"]), exits, red_after_1r, r_known,
                       median([r.best_pct for r in rows]), median([r.realized_pct for r in rows]),
                       pct(True), pct(False), count[True], count[False], rows)


def _group_line(title, rows):
    if not rows:
        return f"{title}: yok."
    regimes = {}
    for row in rows:
        name = REGIME_TEXT.get(row.regime, "belirsiz")
        regimes[name] = regimes.get(name, 0) + 1
    where = ", ".join(f"{count} {name}" for name, count in sorted(regimes.items(), key=lambda kv: -kv[1]))
    return (f"{title}: {len(rows)} işlem · medyan kapanış %{median([r.realized_pct for r in rows]):.2f} · "
            f"medyan en iyi seviye %{median([r.best_pct for r in rows]):.2f} · giriş koşulu: {where}.")


def explain(diag: Diagnostics) -> list[str]:
    """Teşhisin düz Türkçe yorumu; sayı yoksa yorum uydurulmaz."""
    lines = []
    stop, sat = diag.exits.get("STOP", 0), diag.exits.get("SAT", 0)
    if diag.closed == 0:
        lines.append("Kapanmış işlem yok; giriş ve çıkış kalitesi ölçülemedi.")
    else:
        lines.append(f"Çıkışlar: {stop} stop, {sat} SAT sinyali"
                     + (f", {diag.closed - stop - sat} diğer." if diag.closed - stop - sat else "."))
        winners = [r for r in diag.rows if r.won]
        losers = [r for r in diag.rows if not r.won]
        lines.append(_group_line("Kazananlar", winners))
        lines.append(_group_line("Kaybedenler", losers))
        if diag.r_known:
            lines.append(f"En az 1R kâr görüp zararla kapanan: {diag.reached_1r_closed_red} / {diag.r_known} "
                         "(1R = giriş ile başlangıç stopu arası). Bu sayı küçükse sorun kâr almak değil, "
                         "kaybeden girişlerdir.")
        if diag.closed < 30:
            lines.append(f"Yalnız {diag.closed} kapanmış işlem var; desen için az. Birkaç varlıkta "
                         "tekrar bakın.")
    if diag.out_market_pct is not None:
        if diag.out_market_pct == 0:
            lines.append(f"Dışarıda kalınan {diag.days_out} günde varlık değişmedi.")
        elif diag.out_market_pct < 0:
            lines.append(f"Dışarıda kalınan {diag.days_out} günde varlık %{diag.out_market_pct:.2f} değişti: "
                         "girmemek bu düşüşten korudu.")
        else:
            lines.append(f"Dışarıda kalınan {diag.days_out} günde varlık %{diag.out_market_pct:.2f} değişti: "
                         "girmemek bu yükselişi kaçırdı.")
    return lines


def trade_table(diag: Diagnostics) -> list[dict]:
    """İşlem başına satır (ekran tablosu için)."""
    return [{
        "Giriş": str(row.entry_day), "Çıkış": str(row.exit_day),
        "Sonuç": "Kazandı" if row.won else "Kaybetti", "Çıkış nedeni": row.reason,
        "Kapanış %": f"{row.realized_pct:.2f}", "En iyi %": f"{row.best_pct:.2f}",
        "En iyi (R)": "—" if row.best_r is None else f"{row.best_r:.2f}",
        "Giriş koşulu": REGIME_TEXT.get(row.regime, "belirsiz"),
    } for row in diag.rows]
