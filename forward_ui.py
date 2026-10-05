"""İleri dönem sanal takip paneli (spec 0005, Adım 7).

Kayıtlar `forward_tracker` tablolarından okunur; burası yalnız gösterim satırlarını
üretir. Koşucunun son başarılı çalışması görünür; uzun süre çalışmadıysa uyarılır
(GitHub zamanlanmış görevleri 60 gün değişmeyen depoda kendiliğinden durur).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import forward_tracker as ft

STALE_AFTER = timedelta(days=2)
RECENT = 5
#: Ayrıntısı gösterilen en çok varlık-sürüm çifti (her çift birkaç sorgu; uzak veritabanında saniyeler tutar).
MAX_PAIRS = 6


@dataclass(frozen=True)
class ForwardView:
    lines: list[str]


def build_forward_status(now: datetime) -> list[str]:
    """Koşucunun sağlığı: yalnız iki ucuz sorgu (son çalışma, son başarılı çalışma).

    Ağır panel kapalıyken de kesinti görünür kalır (spec 0008 AC04; 0005 AC38/AC99)."""
    lines = []
    runs = ft.recent_runs(1)
    last = ft.last_successful_run()
    if last is None:
        lines.append("İleri takip koşucusu henüz çalışmadı.")
    else:
        lines.append(f"Son başarılı takip çalışması: {last.isoformat()}")
        if now - last > STALE_AFTER:
            lines.append("Uyarı: koşucu 2 günden uzun süredir çalışmadı; takip durmuş olabilir.")
    if runs:
        latest = runs[0]
        if not latest["ok"]:
            lines.append(f"Uyarı: Son çalışma başarısız ({latest['run_at'].isoformat()}): {latest['note']}")
        elif latest["note"] and latest["note"] != "tamam":
            lines.append(f"Son çalışma notu ({latest['run_at'].isoformat()}): {latest['note']}")
    return lines


def render_forward_status(now: datetime) -> None:
    import streamlit as st

    from storage import StorageAccessError

    try:
        lines = build_forward_status(now)
    except StorageAccessError:
        st.warning("İleri takip kayıtları okunamadı; kayıt deposu erişimini kontrol edin.")
        return
    for line in lines:
        (st.warning if line.startswith("Uyarı") else st.caption)(line)


def build_forward_view(now: datetime) -> ForwardView:
    lines = build_forward_status(now)
    pairs = ft.tracked_pairs()
    for asset, version in pairs[:MAX_PAIRS]:
        verdict = ft.assess(asset, version)
        status = "yeterli kanıt" if verdict.sufficient else "yetersiz kanıt"
        detail = "" if verdict.sufficient else " (" + ", ".join(verdict.missing) + ")"
        lines.append(f"{asset} · {version}: {status}{detail}")
        gaps = ft.missing_days(asset, version)
        if gaps:
            lines.append(f"{asset} · {version}: eksik görünen gün: {len(gaps)} "
                         f"(ilk: {gaps[0]}; resmi tatiller de eksik görünebilir)")
        recent = ft.decisions(asset, version)[-RECENT:]
        for decision in recent:
            lines.append(f"{decision.candle_day} · {decision.decision} · {version} · kaynak "
                         f"{decision.source} · karar {decision.evaluated_at.isoformat()} · "
                         f"{ft.status_text(decision)}")
        if recent and recent[-1].assumptions:
            shown = "; ".join(f"{key}: {value}" for key, value in recent[-1].assumptions.items())
            lines.append(f"Son kararın varsayımları — {shown}")
        for revision in ft.revisions(asset, version):
            lines.append(f"{revision['candle_day']}: mum sağlayıcıda revize edildi; karar değişmedi.")
    if len(pairs) > MAX_PAIRS:
        lines.append(f"… {len(pairs) - MAX_PAIRS} çift daha izleniyor; ayrıntı yalnız ilk {MAX_PAIRS} çift için gösterilir.")
    lines.append("Sanal takip gerçek işlem değildir; geçmiş sonuç gelecekteki kazanç olasılığı değildir.")
    return ForwardView(lines)


def render_forward_panel(now: datetime) -> None:
    import streamlit as st

    from storage import StorageAccessError

    try:
        view = build_forward_view(now)
    except StorageAccessError:
        st.warning("İleri takip kayıtları okunamadı; kayıt deposu erişimini kontrol edin.")
        return
    for line in view.lines:
        (st.warning if line.startswith("Uyarı") else st.caption)(line)
