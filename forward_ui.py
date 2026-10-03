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


@dataclass(frozen=True)
class ForwardView:
    lines: list[str]


def _tracked_pairs() -> list[tuple[str, str]]:
    from sqlalchemy import text

    with ft._engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT asset, strategy_version FROM forward_decisions "
                                 "ORDER BY asset, strategy_version")).fetchall()
    return [(row.asset, row.strategy_version) for row in rows]


def build_forward_view(now: datetime) -> ForwardView:
    lines = []
    last = ft.last_successful_run()
    if last is None:
        lines.append("İleri takip koşucusu henüz çalışmadı.")
    else:
        lines.append(f"Son başarılı takip çalışması: {last.isoformat()}")
        if now - last > STALE_AFTER:
            lines.append("Uyarı: koşucu 2 günden uzun süredir çalışmadı; takip durmuş olabilir.")
    for asset, version in _tracked_pairs():
        verdict = ft.assess(asset, version)
        status = "yeterli kanıt" if verdict.sufficient else "yetersiz kanıt"
        detail = "" if verdict.sufficient else " (" + ", ".join(verdict.missing) + ")"
        lines.append(f"{asset} · {version}: {status}{detail}")
        for decision in ft.decisions(asset, version)[-RECENT:]:
            lines.append(f"{decision.candle_day} · {decision.decision} · {version} · kaynak "
                         f"{decision.source} · karar {decision.evaluated_at.isoformat()} · "
                         f"{ft.status_text(decision)}")
        for revision in ft.revisions(asset, version):
            lines.append(f"{revision['candle_day']}: mum sağlayıcıda revize edildi; karar değişmedi.")
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
