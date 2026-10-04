"""Aday stratejiler paneli (spec 0005, Adım 5 / S3).

Hesap `strategy_candidates.py` ve `regime_classifier.py`'dedir; burası değerlendirme
akışını kurar ve gösterim metnini üretir. Adaylar ayrı değerlendirme diliminde (Q01),
V1 referansıyla aynı sermaye ve dönemde koşulur. Sonuç ne olursa olsun aktif strateji
değişmez; yalnız kullanıcının açık seçimi değiştirir (R09).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import market_map
import strategy_candidates as sc
from evaluation_window import split_dates
from regime_classifier import DUSEN, YATAY, YUKSELEN, classify_frame, classify_latest

STATUS_TEXT = {
    sc.OLCUTU_KARSILADI: "Ölçütü karşıladı",
    sc.OLCUTU_KARSILAMADI: "Ölçütü karşılamadı",
    sc.YETERSIZ_VERI: "Yetersiz veri",
    sc.DENEME: "Deneme — onaylı strateji değerlendirmesi değildir",
}
REGIME_TEXT = {YUKSELEN: "Yükselen", DUSEN: "Düşen", YATAY: "Yatay", None: "hesaplanamıyor"}


@dataclass(frozen=True)
class CandidateView:
    lines: list[str]


@dataclass(frozen=True)
class EvaluationOutcome:
    ok: bool
    reason: str = ""
    results: list[tuple[str, str]] = field(default_factory=list)


def _as_day(value) -> date:
    return value.date() if hasattr(value, "date") else value


def evaluate_candidates(symbol: str, frame, v1_decisions: dict, *, notional: Decimal,
                        capital: Decimal, quantity_step, costs, now: datetime) -> EvaluationOutcome:
    """Varsayılan adayları ön kaydeder, değerlendirme diliminde koşar ve geçmişe yazar."""
    from technical_analysis import run_v1_strategy_backtest

    info = market_map.market_of(symbol)
    if info is None:
        return EvaluationOutcome(False, "Bu varlığın piyasası tanınmıyor; adaylar değerlendirilmedi.")
    market = info[0]
    split = split_dates([_as_day(ts) for ts in frame.index])
    if len(split.evaluation) < 2:
        return EvaluationOutcome(False, "Yetersiz geçmiş: ayrı değerlendirme dilimi kurulamadı.")
    first, last = split.evaluation[0], split.evaluation[-1]
    inside = [_as_day(ts) >= first for ts in frame.index]
    window = frame[inside]
    regimes_full = classify_frame(frame)
    run = dict(initial_cash=Decimal(capital), trade_notional=Decimal(notional),
               quantity_step=quantity_step, costs=costs)
    reference = sc.metrics_from_backtest(run_v1_strategy_backtest(window, v1_decisions, **run))

    results = []
    for candidate in sc.default_candidates(market, first, last):
        sc.preregister(candidate, now)
        permit = sc.start_run(candidate)
        if not permit.ok:
            return EvaluationOutcome(False, permit.reason, results)
        # Giriş kuralı tam geçmişi görür (ör. önceki 20 gün); yalnız sonuç ölçümü dilimde başlar
        # ve koşucuyla aynı kararı üretir (AC97).
        decisions = sc.candidate_decisions(frame, v1_decisions, candidate, regimes_full)
        metrics = sc.metrics_from_backtest(run_v1_strategy_backtest(window, decisions, **run))
        verdict = sc.judge(metrics, reference, approved=sc.is_approved(candidate))
        sc.record_result(candidate, market, verdict, now)
        results.append((candidate.name, verdict.status))
    return EvaluationOutcome(True, "", results)


def build_candidate_view(frame) -> CandidateView:
    active = sc.active_strategy()
    if active == sc.REFERENCE_STRATEGY:
        lines = ["Aktif strateji: V1 (mevcut strateji)"]
    else:
        lines = [f"Aktif strateji: {sc.registered_name(active) or active} (kullanıcı seçimi)"]
    if frame is not None:
        label, version = classify_latest(frame)
        lines.append(f"Son günün piyasa koşulu: {REGIME_TEXT[label]} ({version})")
    entries = sc.history()
    if not entries:
        lines.append("Henüz değerlendirilmiş aday yok.")
    for entry in entries:
        text = f"{entry['name']} — {entry['market']}: {STATUS_TEXT[entry['status']]}"
        if entry["reason"]:
            text += f" ({entry['reason']})"
        lines.append(text)
    lines.append("Geçmiş sonuçlar gelecekteki kazanma olasılığı değildir; aday, siz seçmeden "
                 "aktif strateji olmaz.")
    return CandidateView(lines)


def render_candidate_panel(source: dict) -> None:
    import streamlit as st

    from storage import StorageAccessError

    try:
        _render_candidate_panel(st, source)
    except StorageAccessError:
        st.warning("Aday kayıtları okunamadı; kayıt deposu erişimini kontrol edin.")


def _render_candidate_panel(st, source: dict) -> None:
    from datetime import timezone

    st.markdown("**🧪 Aday Stratejiler (kırılım / geri çekilme)**")
    if st.button("Adayları değerlendir", key=f"cand_eval:{source['symbol']}"):
        outcome = evaluate_candidates(
            source["symbol"], source["frame"], source["decisions"], notional=source["notional"],
            capital=Decimal(source["backtest"]["initial_cash"]),
            quantity_step=source["quantity_step"], costs=source["costs"],
            now=datetime.now(timezone.utc))
        if not outcome.ok:
            st.warning(outcome.reason)
    for line in build_candidate_view(source["frame"]).lines:
        st.caption(line)
    info = market_map.market_of(source["symbol"])
    if info is not None:
        for candidate in sc.default_candidates(info[0], None, None):
            if st.button(f"Aktif yap: {candidate.name}", key=f"cand_pick:{source['symbol']}:{candidate.name}"):
                sc.preregister(candidate, datetime.now(timezone.utc))
                sc.choose_active_strategy(candidate)
                st.rerun()
    if sc.active_strategy() != sc.REFERENCE_STRATEGY and st.button("V1'e dön", key=f"cand_back:{source['symbol']}"):
        sc.choose_active_strategy(None)
        st.rerun()
