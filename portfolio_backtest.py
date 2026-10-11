"""Ağırlıklı portföy geçmiş testi (spec 0019, P1).

Seçilen varlıklar tek kasayı paylaşır. Her varlık, tek varlık motorunun aynı kurallarıyla
(`trade_execution.advance_daily_bar`: karar ertesi açılışta, sabit 2,5 ATR stop, SAT'ta tamamı satılır,
zararlı çıkıştan sonra bekleme, aynı mumda stop, maliyet modeli) işletilir; yalnız alım tutarı portföyden
gelir: girişte o günkü toplam varlığın varlığa ait ağırlığı kadar. Nakit yetmezse mevcut nakitle sınırlanır;
ağırlığın yarısından azı kalıyorsa o gün giriş yapılmaz.

Al-tut karşılaştırması: her varlık kendi ilk gününde başlangıç sermayesinin ağırlığı kadar alınır ve tutulur
(ağırlık toplamı %100'ü aşarsa oranlar küçültülür). Para hesabı Decimal'dir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from typing import Mapping

from trade_execution import Bar, BookState, CostAssumptions, TradeSettings, advance_daily_bar

#: Varsayılan ağırlıklar (spec 0019 R04, kullanıcı onayı): BTC/ETH %20, DOGE %5, diğer kriptolar %10.
MAJOR_BASES = frozenset({"BTC", "ETH"})
SMALL_BASES = frozenset({"DOGE"})
#: Miktar adımı bilinmeyen varlıkta kullanılan adım; sonuç "yaklaşık" etiketlenir.
FALLBACK_STEP = Decimal("0.00000001")
_ZERO = Decimal("0")


def default_weight(symbol: str) -> Decimal:
    """Sembolün varsayılan portföy ağırlığı (yüzde)."""
    import market_map

    base = market_map.canonical_symbol(symbol).split("-")[0]
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


def run_portfolio(frames: Mapping, decisions: Mapping, weights: Mapping, *, capital: Decimal,
                  costs: CostAssumptions, quantity_steps: Mapping | None = None) -> PortfolioResult:
    """`frames`: sembol → günlük mum tablosu (ATR dahil); `decisions`: sembol → {gün: AL/SAT/BEKLE};
    `weights`: sembol → yüzde. Döner: `PortfolioResult`."""
    capital = Decimal(capital)
    quantity_steps = dict(quantity_steps or {})
    result = PortfolioResult()
    symbols = [s for s in frames if Decimal(weights.get(s, 0)) > 0]
    steps = {}
    for s in symbols:
        step = quantity_steps.get(s)
        if step is None:
            result.approximate.append(s)
            step = FALLBACK_STEP
        steps[s] = Decimal(step)
    rate = Decimal("0") if costs.commission_pct is None else Decimal(costs.commission_pct) / 100
    states = {s: BookState(_ZERO) for s in symbols}
    positions_close = {s: None for s in symbols}             # son bilinen kapanış (değerleme)
    index = {s: {ts: i for i, ts in enumerate(frames[s].index)} for s in symbols}
    days = sorted(set().union(*[set(frames[s].index) for s in symbols])) if symbols else []
    cash = capital
    result.contribution = {s: _ZERO for s in symbols}

    def record(symbol, trades):
        for item in trades:
            result.trades.append(PortfolioTrade(symbol, item.entry_at, item.exit_at, item.reason, item.pnl))
            result.contribution[symbol] += item.pnl

    def value_of(symbol):
        position = states[symbol].position
        last = positions_close[symbol]
        return _ZERO if position is None or last is None else position.quantity * last

    for t in days:
        active = [s for s in symbols if t in index[s] and index[s][t] >= 1]
        # 1) açık pozisyonlar: çıkışlar (tüm varlıklarda girişlerden önce)
        for s in active:
            state = states[s]
            if state.position is None:
                continue
            i = index[s][t]
            frame = frames[s]
            verdict = decisions[s].get(frame.index[i - 1], "BEKLE")
            atr = _finite(frame["ATR"].iloc[i - 1])
            state.cash = _ZERO
            step = advance_daily_bar(state, _bar(frame, i), verdict, atr,
                                     TradeSettings(_ZERO, steps[s], costs))
            record(s, step.trades)
            cash += state.cash
            state.cash = _ZERO
        # 2) giriş tutarı için toplam varlık (önceki kapanışlarla)
        equity_prev = cash + sum((value_of(s) for s in symbols), _ZERO)
        # 3) yeni girişler
        for s in active:
            state = states[s]
            if state.position is not None:
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

    result.holdings = {s: value_of(s) for s in symbols if states[s].position is not None}
    result.cash = cash
    result.hold_equity = _hold(frames, weights, symbols, steps, capital, costs, days)
    return result


def _hold(frames, weights, symbols, steps, capital, costs, days):
    """Al-tut: her varlık kendi ilk gününde (başlangıç sermayesi × ağırlık) alınır, tutulur."""
    from trade_execution import commission_fee, fill_price

    total = sum((Decimal(weights[s]) for s in symbols), _ZERO)
    scale = Decimal("1") if total <= 100 else Decimal("100") / total
    cash, held, last = capital, {}, {}
    curve = []
    for t in days:
        for s in symbols:
            frame = frames[s]
            if t not in frame.index:
                continue
            if s not in held:
                price = fill_price(Decimal(str(frame["Open"].loc[t])), "BUY", costs) or Decimal(str(frame["Open"].loc[t]))
                budget = capital * Decimal(weights[s]) * scale / 100
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
        annual = (Decimal(float(ratio) ** (1 / float(years))) - 1) * 100
    return {"total": (ratio - 1) * 100, "annual": annual, "drawdown": worst * 100, "years": years}
