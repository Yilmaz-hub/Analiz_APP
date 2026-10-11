"""Varlığa göre sinyal kuralı (spec 0022).

Günlük karar, varlığın yöntemine göre son kapanmış mumdan üretilir (`strategy_candidates.asset_method`):
BTC/ETH ve altın C (V1 AL + kapanış > 200 günlük ortalama; çıkış V1 SAT), altcoin kanal kırılımı 50/20,
hisse al-tut (zamanlama sinyali yok). Stop her kuralda aynıdır (2,5 ATR). V1 kararı bilgi olarak yanında yazılır.
"""
from __future__ import annotations

from dataclasses import dataclass

from trade_decisions import Signal

import strategy_candidates as sc

VARLIK = "VARLIK"
V1 = "V1"
_MODE_KEY = "signal_mode"
RULE_TEXT = {"TREND_SMA": "V1 + 200 günlük ortalama", "KANAL": "Kanal kırılımı 50/20", sc.TUT: "Al-tut"}
_SIGNAL_TEXT = {Signal.BUY: "AL", Signal.SELL: "SAT", Signal.WAIT: "BEKLE"}


@dataclass(frozen=True)
class RuleSignal:
    signal: Signal
    method: str | None       # None: varlığın kuralı yok (ör. döviz) → V1 kullanılır
    text: str | None


def signal_mode() -> str:
    """Kayıtlı sinyal kuralı seçimi; seçilmemişse varlığa göre (kullanıcı onayı 2026-10-11)."""
    import storage

    value = (storage.read_doc(_MODE_KEY) or {}).get("mode")
    return value if value in (VARLIK, V1) else VARLIK


def choose_signal_mode(mode: str) -> None:
    import storage

    if mode not in (VARLIK, V1):
        raise ValueError(mode)
    storage.update_doc(_MODE_KEY, lambda _: {"mode": mode})


def rule_signal(symbol: str, closed, v1_signal: Signal) -> RuleSignal:
    """`closed`: yalnız kapanmış günlük mumlar (sağlayıcının açık mumu hariç). `v1_signal`: aynı mumdaki V1 kararı."""
    method = sc.asset_method(symbol)
    if method is None or closed is None or len(closed) < 2:
        return RuleSignal(v1_signal, None, None)
    v1_text = _SIGNAL_TEXT[v1_signal]
    if method == sc.TUT:
        return RuleSignal(Signal.WAIT, method,
                          f"Sinyal kuralı: Al-tut — hisselerde zamanlama sinyali üretilmez; portföy ağırlığınızca "
                          f"tutulur (V1 bilgi: {v1_text}).")
    if method == "KANAL":
        import market_map

        candidate = sc.channel_candidate(market_map.market_of(symbol)[0])
        verdict = sc.candidate_decisions(closed, {}, candidate, None)[closed.index[-1]]
        signal = Signal.BUY if verdict == "AL" else Signal.SELL if verdict == "SAT" else Signal.WAIT
    else:
        average = closed["Close"].rolling(200).mean().iloc[-1]
        above = average == average and float(closed["Close"].iloc[-1]) > float(average)
        signal = (Signal.SELL if v1_signal is Signal.SELL
                  else Signal.BUY if v1_signal is Signal.BUY and above else Signal.WAIT)
    return RuleSignal(signal, method,
                      f"Sinyal kuralı: {RULE_TEXT[method]} → {_SIGNAL_TEXT[signal]} (V1 kararı: {v1_text}).")


def backtest_method(symbol: str) -> str | None:
    """Günlük backtest'te test edilen kural; "V1" modunda ya da kuralı olmayan varlıkta None (V1)."""
    return sc.asset_method(symbol) if signal_mode() == VARLIK else None


def backtest_label(method: str | None) -> str:
    """Backtest bilgi kutusu metni (spec 0023 R02); eski V1 metni yalnız V1 kullanıldığında."""
    if method == sc.TUT:
        return "Günlük test — kural: Al-tut; bu varlıkta zamanlama testi yapılmaz, yalnız al-tut sonucu gösterilir."
    rule = "Günlük V1 testi" if method is None else f"Günlük test — kural: {RULE_TEXT[method]}"
    return (f"{rule}; kapanmış mum kararını (ML kararlara girmez) sonraki açılışta uygular, 2,5 ATR başlangıç "
            "stopunu sabit tutar ve hedef/otomatik iz süren stopla satış yapmaz.")


def rule_decisions(symbol: str, frame, v1_decisions: dict) -> tuple[dict, str | None]:
    """Günlük backtest'in karar sözlüğü (spec 0023 R01). "V1" modunda ya da kuralı olmayan varlıkta V1 kararları
    aynen; `TUT` (hisse) için boş sözlük — zamanlama testi yapılmaz. C ve kanal kararları yalnız ilk V1 karar
    gününden itibaren tutulur (portföy testiyle aynı ısınma; al-tut karşılaştırması ve teşhis ısınmayı saymaz)."""
    method = backtest_method(symbol)
    if method is None:
        return v1_decisions, None
    if method == sc.TUT:
        return {}, method
    import market_map

    market = market_map.market_of(symbol)[0]
    if method == "KANAL":
        decisions = sc.candidate_decisions(frame, {}, sc.channel_candidate(market), None)
    else:
        decisions = sc.candidate_decisions(frame, v1_decisions, sc.trend_candidate(market), None)
    if not v1_decisions:
        return {}, method
    first = min(v1_decisions)
    return {day: verdict for day, verdict in decisions.items() if day >= first}, method


def render_mode_switch(st) -> str:
    """Karar panelindeki kural seçimi; değişince kaydedilir (tüm varlıklar için geçerli)."""
    current = signal_mode()
    labels = {VARLIK: "Varlığa göre (BTC/ETH/altın: V1 + 200 gün ort. · altcoin: kanal 50/20 · hisse: al-tut)",
              V1: "V1 (eski)"}
    picked = st.radio("Sinyal kuralı", [VARLIK, V1], index=[VARLIK, V1].index(current),
                      format_func=labels.get, horizontal=True, key="signal_mode_choice")
    if picked != current:
        choose_signal_mode(picked)
    return picked
