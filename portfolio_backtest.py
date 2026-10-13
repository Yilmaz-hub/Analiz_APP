"""Ağırlıklı portföy geçmiş testi (spec 0019, P1).

Seçilen varlıklar tek kasayı paylaşır. Her varlık, tek varlık motorunun aynı kurallarıyla
(`trade_execution.advance_daily_bar`: karar ertesi açılışta, sabit 2,5 ATR stop, SAT'ta tamamı satılır,
zararlı çıkıştan sonra bekleme, aynı mumda stop, maliyet modeli) işletilir; yalnız alım tutarı portföyden
gelir: girişte o günkü toplam varlığın varlığa ait ağırlığı kadar. Nakit yetmezse mevcut nakitle sınırlanır;
ağırlığın yarısından azı kalıyorsa o gün giriş yapılmaz.

Tut yöntemi (spec 0021): `hold` içindeki varlıklar karar ve stop kullanmaz; ilk işlem gününde o günkü toplam
varlığın ağırlığı kadar alınır ve dönem sonuna kadar tutulur (nakit yetmezse ertesi gün yeniden denenir).

Al-tut karşılaştırması: her varlık kendi ilk gününde başlangıç sermayesinin ağırlığı kadar alınır ve tutulur.
Para hesabı Decimal'dir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from typing import Mapping

from trade_execution import Bar, BookState, CostAssumptions, TradeSettings, advance_daily_bar

#: Varsayılan ağırlıklar (spec 0024): BTC/ETH %20, ons altın %15, DOGE ve BIST hisseleri %5, diğer %10.
MAJOR_BASES = frozenset({"BTC", "ETH"})
SMALL_BASES = frozenset({"DOGE"})
#: Miktar adımı bilinmeyen varlıkta kullanılan adım; sonuç "yaklaşık" etiketlenir.
FALLBACK_STEP = Decimal("0.00000001")
_ZERO = Decimal("0")


def default_weight(symbol: str) -> Decimal:
    """Sembolün varsayılan portföy ağırlığı (yüzde)."""
    import market_map

    info = market_map.market_of(symbol)
    if info is not None and info[0] == "BIST":           # spec 0021: dolar bazında güvenilir değil, küçük ağırlık
        return Decimal("5")
    base = market_map.canonical_symbol(symbol).split("-")[0]
    if base == "XAU_GOLD":
        return Decimal("15")
    if base in MAJOR_BASES:
        return Decimal("20")
    if base in SMALL_BASES:
        return Decimal("5")
    return Decimal("10")


@dataclass(frozen=True)
class PortfolioTrade:
    symbol: str
    entry_at: object
    exit_at: object
    reason: str
    pnl: Decimal


@dataclass
class PortfolioResult:
    equity: list = field(default_factory=list)          # [(gün, toplam varlık)]
    hold_equity: list = field(default_factory=list)     # [(gün, al-tut değeri)]
    trades: list = field(default_factory=list)          # PortfolioTrade
    contribution: dict = field(default_factory=dict)    # sembol → kapanmış işlemlerin net K/Z toplamı
    approximate: list = field(default_factory=list)     # miktar adımı bilinmeyen semboller
    blocked: dict = field(default_factory=dict)         # neden → adet
    holdings: dict = field(default_factory=dict)        # dönem sonunda açık pozisyonların değeri
    cash: Decimal = Decimal("0")                        # dönem sonu nakit

    @property
    def start(self):
        return self.equity[0][0] if self.equity else None

    @property
    def end(self):
        return self.equity[-1][0] if self.equity else None


def _day(value) -> date:
    return value.date() if hasattr(value, "date") else value


def _bar(frame, i) -> Bar:
    row = frame.iloc[i]
    return Bar(frame.index[i], Decimal(str(row["Open"])), Decimal(str(row["High"])),
               Decimal(str(row["Low"])), Decimal(str(row["Close"])))


def _finite(value) -> Decimal | None:
    number = Decimal(str(value))
    return number if number.is_finite() else None


#: Spec 0021: tek kasa dolardır; USDT dolar sayılır, TL varlıklar günlük USD/TRY kuruyla çevrilir.
POOL_CURRENCIES = {"USD": "USD", "USDT": "USD", "TRY": "TRY"}
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "ATR")


def to_usd(frame, fx_frame):
    """TL fiyatlı mum tablosunu dolara çevirir: her gün o günün (yoksa son bilinen) USD/TRY kapanışına bölünür.

    Kur takvim gününe göre eşlenir (saat dilimi farkı eşleşmeyi bozmaz); ilk kurdan önceki günler atılır.
    ATR de aynı kurla ölçeklenir (yaklaşık; günler arası kur oynaklığı ATR'ye eklenmez)."""
    import pandas as pd

    rates = pd.Series(fx_frame["Close"].values, index=[_day(ts) for ts in fx_frame.index])
    rates = rates[~rates.index.duplicated(keep="last")].sort_index()
    days = [_day(ts) for ts in frame.index]
    aligned = rates.reindex(sorted(set(days) | set(rates.index))).ffill().reindex(days)
    keep = aligned.notna().values & (aligned.values > 0)
    out = frame.loc[keep].copy()
    rate = aligned.values[keep]
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].values / rate
    return out


def portfolio_start(ready: Mapping, reference) -> tuple:
    """Spec 0025 R01: test başlangıcı referans varlıkların (BTC/ETH; yoksa kriptolar; yoksa hepsi) en erken işlem
    günüdür. Döner: (başlangıç, {varlık: ilk işlem günü}) — ikinci sözlük başlangıçtan sonra hazır olan, payı o
    güne kadar nakit sayılan varlıklardır."""
    chosen = [ready[name] for name in reference if name in ready] or list(ready.values())
    start = min(chosen)
    return start, {name: day for name, day in ready.items() if day > start}


def first_tradable_day(frame, decided_days):
    """İlk kararın bir sonraki mumu: strateji en erken bu açılışta alım yapabilir (ısınma dönemi hariç)."""
    if not decided_days:
        return None
    first = min(decided_days)
    later = [ts for ts in frame.index if ts > first]
    return later[0] if later else None


def run_portfolio(frames: Mapping, decisions: Mapping, weights: Mapping, *, capital: Decimal,
                  costs: CostAssumptions, quantity_steps: Mapping | None = None, start=None,
                  hold=(), available_from: Mapping | None = None) -> PortfolioResult:
    """`frames`: sembol → günlük mum tablosu (ATR dahil); `decisions`: sembol → {gün: AL/SAT/BEKLE};
    `weights`: sembol → yüzde. `start` verilirse sermaye eğrisi ve al-tut bu günden başlar (ısınma dönemi
    karşılaştırmaya girmez; spec 0019 QA). `hold`: stopsuz al-tut yöntemiyle tutulan semboller (spec 0021);
    bunlar için `decisions` gerekmez. `available_from`: sembol → varlığın işlem yapabildiği ilk gün (spec 0025);
    o güne kadar payı nakittir, tut alımı ve al-tut karşılaştırması o gün yapılır. Döner: `PortfolioResult`."""
    capital = Decimal(capital)
    quantity_steps = dict(quantity_steps or {})
    result = PortfolioResult()
    for s in frames:
        w = Decimal(str(weights.get(s, 0)))
        if w < _ZERO or w > Decimal("100"):
            raise ValueError(f"Varlık ağırlığı 0 ile 100 arasında olmalıdır: {s}=%{w:g}")
    total_weight = sum((Decimal(str(weights.get(s, 0))) for s in frames), _ZERO)
    if total_weight > Decimal("100"):
        raise ValueError(f"Portföy ağırlıkları toplamı %100'ü aşamaz (girilen: %{total_weight:g}).")
    if total_weight <= _ZERO:
        raise ValueError("Portföy ağırlıkları toplamı sıfırdan büyük olmalıdır.")
    symbols = [s for s in frames if Decimal(weights.get(s, 0)) > 0]
    steps = {}
    for s in symbols:
        step = quantity_steps.get(s)
        if step is None:
            result.approximate.append(s)
            step = FALLBACK_STEP
        steps[s] = Decimal(step)
    rate = Decimal("0") if costs.commission_pct is None else Decimal(costs.commission_pct) / 100
    from trade_execution import commission_fee, fill_price

    hold = {s for s in hold if s in symbols}
    held, paid, hold_blocked = {}, {}, set()                    # tut: miktar ve toplam maliyet (komisyon dahil)
    states = {s: BookState(_ZERO) for s in symbols}
    positions_close = {s: None for s in symbols}             # son bilinen kapanış (değerleme)
    index = {s: {ts: i for i, ts in enumerate(frames[s].index)} for s in symbols}
    days = sorted(set().union(*[set(frames[s].index) for s in symbols])) if symbols else []
    if start is not None:
        days = [t for t in days if t >= start]
    cash = capital
    result.contribution = {s: _ZERO for s in symbols}

    def record(symbol, trades):
        for item in trades:
            result.trades.append(PortfolioTrade(symbol, item.entry_at, item.exit_at, item.reason, item.pnl))
            result.contribution[symbol] += item.pnl

    def value_of(symbol):
        last = positions_close[symbol]
        if symbol in hold:
            return _ZERO if symbol not in held or last is None else held[symbol] * last
        position = states[symbol].position
        return _ZERO if position is None or last is None else position.quantity * last

    for t in days:
        active = [s for s in symbols if t in index[s] and index[s][t] >= 1]
        exited_today = set()
        # 1) açık pozisyonlar: çıkışlar (tüm varlıklarda girişlerden önce)
        for s in active:
            state = states[s]
            if s in hold or state.position is None:
                continue
            i = index[s][t]
            frame = frames[s]
            verdict = decisions[s].get(frame.index[i - 1], "BEKLE")
            atr = _finite(frame["ATR"].iloc[i - 1])
            state.cash = _ZERO
            step = advance_daily_bar(state, _bar(frame, i), verdict, atr,
                                     TradeSettings(_ZERO, steps[s], costs))
            record(s, step.trades)
            if step.trades:
                exited_today.add(s)       # tek varlık motoru gibi: çıkış mumunda başka işlem yok, bekleme ertesi gün başlar
            cash += state.cash
            state.cash = _ZERO
        # 2) giriş tutarı için toplam varlık (önceki kapanışlarla)
        equity_prev = cash + sum((value_of(s) for s in symbols), _ZERO)
        # 3a) tut yöntemi: henüz alınmadıysa ağırlığı kadar alınır (karar ve stop yok)
        for s in active:
            if s not in hold or s in held or (available_from and t < available_from.get(s, t)):
                continue
            target = equity_prev * Decimal(weights[s]) / 100
            notional = min(target, cash / (1 + rate))
            price = fill_price(Decimal(str(frames[s]["Open"].iloc[index[s][t]])), "BUY", costs)
            quantity = _ZERO if not price or price <= 0 else \
                (notional / price / steps[s]).to_integral_value(rounding=ROUND_FLOOR) * steps[s]
            if notional < target / 2 or quantity <= 0:
                if s not in hold_blocked:              # her tut varlığı için bir kez sayılır (her gün değil)
                    hold_blocked.add(s)
                    result.blocked["YETERSIZ_NAKIT"] = result.blocked.get("YETERSIZ_NAKIT", 0) + 1
                continue
            cost = quantity * price + commission_fee(quantity * price, costs)
            cash -= cost
            held[s], paid[s] = quantity, cost
        # 3b) yeni girişler
        for s in active:
            state = states[s]
            if s in hold or state.position is not None or s in exited_today:
                continue
            i = index[s][t]
            frame = frames[s]
            verdict = decisions[s].get(frame.index[i - 1], "BEKLE")
            atr = _finite(frame["ATR"].iloc[i - 1])
            if state.cooldown == 0 and verdict == "AL":
                target = equity_prev * Decimal(weights[s]) / 100
                notional = min(target, cash / (1 + rate))
                if notional < target / 2 or notional <= 0:
                    result.blocked["YETERSIZ_NAKIT"] = result.blocked.get("YETERSIZ_NAKIT", 0) + 1
                    continue
            else:
                notional = _ZERO
            state.cash = cash
            step = advance_daily_bar(state, _bar(frame, i), verdict, atr, TradeSettings(notional, steps[s], costs))
            if step.blocked:
                result.blocked[step.blocked] = result.blocked.get(step.blocked, 0) + 1
            record(s, step.trades)
            cash = state.cash
            state.cash = _ZERO
        # 4) gün sonu değerleme
        for s in active:
            positions_close[s] = _finite(frames[s]["Close"].iloc[index[s][t]]) or positions_close[s]
        result.equity.append((_day(t), cash + sum((value_of(s) for s in symbols), _ZERO)))

    result.holdings = {s: value_of(s) for s in symbols if states[s].position is not None or s in held}
    for s in held:                                              # tut: dönem sonu gerçekleşmemiş net K/Z
        result.contribution[s] += value_of(s) - paid[s]
    result.cash = cash
    result.hold_equity = _hold(frames, weights, symbols, steps, capital, costs, days, available_from or {})
    return result


def _hold(frames, weights, symbols, steps, capital, costs, days, available_from):
    """Al-tut: her varlık kendi ilk işlem gününde (`available_from`, yoksa ilk mumu) başlangıç sermayesi × ağırlık
    kadar alınır, tutulur; o güne kadar payı nakittir (spec 0025)."""
    from trade_execution import commission_fee, fill_price

    cash, held, last = capital, {}, {}
    curve = []
    for t in days:
        for s in symbols:
            frame = frames[s]
            if t not in frame.index or (s not in held and t < available_from.get(s, t)):
                continue
            if s not in held:
                price = fill_price(Decimal(str(frame["Open"].loc[t])), "BUY", costs) or Decimal(str(frame["Open"].loc[t]))
                budget = capital * Decimal(str(weights[s])) / 100
                rate = Decimal("0") if costs.commission_pct is None else Decimal(costs.commission_pct) / 100
                quantity = (budget / (1 + rate) / price / steps[s]).to_integral_value(rounding=ROUND_FLOOR) * steps[s]
                cash -= quantity * price + commission_fee(quantity * price, costs)
                held[s] = quantity
            last[s] = _finite(frame["Close"].loc[t]) or last.get(s, _ZERO)
        curve.append((_day(t), cash + sum((held[s] * last[s] for s in held), _ZERO)))
    return curve


def summary(curve: list) -> dict:
    """Toplam ve yıllık getiri (%), en büyük düşüş (%), süre (yıl)."""
    if len(curve) < 2 or curve[0][1] <= 0:
        return {"total": None, "annual": None, "drawdown": None, "years": 0}
    start, end = curve[0][1], curve[-1][1]
    years = Decimal((curve[-1][0] - curve[0][0]).days) / Decimal("365.25")
    peak, worst = curve[0][1], _ZERO
    for _, value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1)
    ratio = end / start
    annual = None
    if years > 0 and ratio > 0:
        annual = ((ratio.ln() / years).exp() - 1) * 100
    return {"total": (ratio - 1) * 100, "annual": annual, "drawdown": worst * 100, "years": years}
