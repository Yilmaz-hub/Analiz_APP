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
from trading_contracts import Signal

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
                        capital: Decimal, quantity_step, costs, now: datetime,
                        trial_settings: dict | None = None) -> EvaluationOutcome:
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
    defaults = sc.default_candidates(market, first, last)
    trials = _trial_candidates(defaults, trial_settings or {})
    for candidate, default in [(c, None) for c in defaults] + trials:
        sc.preregister(candidate, now)
        permit = sc.start_run(candidate)
        if not permit.ok:
            return EvaluationOutcome(False, permit.reason, results)
        # Giriş kuralı tam geçmişi görür (ör. önceki 20 gün); yalnız sonuç ölçümü dilimde başlar
        # ve koşucuyla aynı kararı üretir (AC97).
        decisions = sc.candidate_decisions(frame, v1_decisions, candidate, regimes_full)
        metrics = sc.metrics_from_backtest(run_v1_strategy_backtest(window, decisions, **run))
        # Her piyasa kendi verisiyle ayrı yargılanır; başka piyasanın sonucu aktarılmaz (AC22).
        verdict = sc.judge_by_market({market: (metrics, reference)},
                                     approved=sc.is_approved(candidate))[market]
        effect = None
        if default is not None:
            effect = sc.single_filter_effect(default, candidate) or "COK"
        sc.record_result(candidate, market, verdict, now, filter_effect=effect)
        results.append((candidate.name, verdict.status))
    return EvaluationOutcome(True, "", results)


def _trial_candidates(defaults: list, trial: dict) -> list[tuple]:
    """Kullanıcının girdiği ayarlarla varsayılandan farklı deneme adayları (AC15).

    Yalnız adayın kendi ayar anahtarları dikkate alınır; varsayılanla aynı değer deneme üretmez."""
    from dataclasses import replace

    trials = []
    for default in defaults:
        changed = {key: str(value).strip() for key, value in trial.items()
                   if key in default.settings and str(value).strip()
                   and str(value).strip() != default.settings[key]}
        if changed:
            variant = replace(default, name=f"{default.name} — deneme",
                              settings={**default.settings, **changed})
            trials.append((variant, default))
    return trials


def active_entry_signal(frame, v1_signal: Signal) -> tuple[Signal, str | None]:
    """Aktif strateji bir adaysa canlı giriş sinyali onun kuralından gelir (R09, AC104).

    Yalnız kullanıcının açık seçimiyle değişir; V1 aktifken sinyal aynen kalır. Çıkış (SAT)
    V1 kurallarıyla aynıdır: aday yalnız girişi değiştirir. Aday kaydı bulunamazsa V1 kullanılır
    ve bu belirtilir."""
    active = sc.active_strategy()
    if active == sc.REFERENCE_STRATEGY:
        return v1_signal, None
    candidate = sc.find_by_strategy(active)
    if candidate is None:
        return v1_signal, "Aktif adayın kaydı bulunamadı; V1 kuralı kullanıldı."
    import pandas as pd

    label, _ = classify_latest(frame)
    regimes = pd.Series([None] * (len(frame) - 1) + [label], index=frame.index, dtype=object)
    last = frame.index[-1]
    verdict = sc.candidate_decisions(
        frame, {last: "SAT"} if v1_signal is Signal.SELL else {}, candidate, regimes)[last]
    signal = Signal.SELL if verdict == "SAT" else Signal.BUY if verdict == "AL" else Signal.WAIT
    return signal, (f"Aktif strateji: {candidate.name} — giriş sinyali bu aday kuralına göre; "
                    "çıkış V1 kurallarıyla.")


def management_note(position, active: str) -> str | None:
    """Açık gerçek pozisyon giriş anındaki sürümle yönetilir; sürüm farklıysa açıklama (Q08, AC77)."""
    import forward_tracker as ft

    version = ft.managing_version(position, active)
    if version == active:
        return None
    shown = sc.registered_name(version) or version
    return (f"Bu pozisyon giriş anındaki strateji sürümüyle ({shown}) yönetilir; aktif sürüm farklı olsa "
            "da çıkış kuralları değişmez.")


def entry_line(entry: dict) -> str:
    text = f"{entry['name']} — {entry['market']}: {STATUS_TEXT[entry['status']]}"
    if entry["reason"]:
        text += f" ({entry['reason']})"
    effect = entry.get("filter_effect")
    if effect == "COK":
        text += " · birden çok ayar değişti: tek filtre etkisi olarak sunulmaz"
    elif effect:
        text += f" · tek filtre etkisi: {effect}"
    return text


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
        lines.append(entry_line(entry))
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
    trial = {}
    a, b, c = st.columns(3)
    trial["lookback"] = a.text_input("Deneme: kırılım gün sayısı", key="cand_try_lookback")
    trial["ema"] = b.text_input("Deneme: EMA", key="cand_try_ema")
    trial["tolerance_pct"] = c.text_input("Deneme: tolerans (%)", key="cand_try_tol")
    if st.button("Denemeyi değerlendir", key=f"cand_try:{source['symbol']}"):
        outcome = evaluate_candidates(
            source["symbol"], source["frame"], source["decisions"], notional=source["notional"],
            capital=Decimal(source["backtest"]["initial_cash"]),
            quantity_step=source["quantity_step"], costs=source["costs"],
            now=datetime.now(timezone.utc), trial_settings=trial)
        if not outcome.ok:
            st.warning(outcome.reason)
    for line in build_candidate_view(source["frame"]).lines:
        st.caption(line)
    lookup_id = st.text_input("Kayıt no (ör. DK-0001)", key=f"cand_lookup:{source['symbol']}")
    if st.button("Kaydı göster", key=f"cand_lookup_btn:{source['symbol']}"):
        found = sc.lookup_evaluation(lookup_id)
        st.caption(found.reason if found.report is None else entry_line(found.report))
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
