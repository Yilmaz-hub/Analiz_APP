"""Kâr koruma karşılaştırması paneli (spec 0005, Adım 6 / R13, AC95).

Hesap `profit_protection.compare`'dadır; burası gösterim satırlarını üretir. Karşılaştırma
sanaldır: gerçek pozisyonun stopunu ve miktarını değiştirmez.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Sequence

import profit_protection as pp

_CENT = Decimal("0.01")


@dataclass(frozen=True)
class ProtectionView:
    lines: list[str]


def _signed(value: Decimal) -> str:
    return f"{value.quantize(_CENT, ROUND_HALF_EVEN):+}"


def build_protection_view(results: Sequence[pp.RuleResult], currency: str, costs) -> ProtectionView:
    reference = next(result for result in results if result.rule == pp.REFERENCE)
    lines = []
    for result in results:
        bound = "" if result.costs_known else " (üst sınır)"
        if result.closed_count == 0:
            lines.append(f"{result.label}: kapanmış işlem yok")
            continue
        expectancy = _signed(result.expectancy)
        line = (f"{result.label}: kapanmış işlem {result.closed_count} · toplam {_signed(result.total_pnl)} "
                f"{currency}{bound} · işlem başına {expectancy} {currency}{bound}")
        if result.rule != pp.REFERENCE:
            line += f" · V1'e göre {_signed(result.total_pnl - reference.total_pnl)} {currency}"
        lines.append(line)
    if any(result.open_count for result in results):
        lines.append("Dönem sonunda kapanmamış sanal pozisyonlar sonuca girmez.")
    if not reference.costs_known:
        lines.append("Komisyon, makas ya da kayma bilinmiyor; sonuçlar üst sınırdır.")
    lines.append("Bu bir sanal karşılaştırmadır; gerçek pozisyonunuzun stopunu ve miktarını değiştirmez. "
                 "Geçmiş sonuç gelecekteki kazanç olasılığı değildir.")
    return ProtectionView(lines)


def render_protection_panel(source: dict, currency: str) -> None:
    import streamlit as st

    st.markdown("**🛡️ Kâr koruma karşılaştırması (sanal)**")
    if source["quantity_step"] is None:
        st.caption("Miktar adımı bilinmiyor; kâr koruma karşılaştırması yapılamadı.")
        return
    if not source["backtest"]["trades"]:
        st.caption("Backtest'te kapanmış işlem yok; karşılaştırılacak bir şey yok.")
        return
    results = pp.compare(source["frame"], source["decisions"], source["backtest"], source["costs"],
                         source["quantity_step"])
    for line in build_protection_view(results, currency, source["costs"]).lines:
        st.caption(line)
