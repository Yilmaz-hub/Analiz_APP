"""İleri dönem sanal takip paneli (spec 0005, Adım 7).

Kayıtlar `forward_tracker` tablolarından okunur; burası yalnız gösterim satırlarını
üretir. Koşucunun son başarılı çalışması görünür; uzun süre çalışmadıysa uyarılır
(GitHub zamanlanmış görevleri 60 gün değişmeyen depoda kendiliğinden durur).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import forward_tracker as ft
from config import ForwardConfig

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


ISTANBUL = ZoneInfo("Europe/Istanbul")
_MONTHS = ("Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara")
_MISSING_SETTINGS = "miktar adımı/varsayımlar eksik"
_STEP_TOO_BIG = "adet adımı × fiyat işlem tutarını aşıyor"
EXPLAIN = ("Takip bu ekrandan açılıp kapanmaz: GitHub görevi günde 3 kez (kripto 00:17, BIST 15:17, ABD ve altın "
           "21:17 UTC) kendiliğinden çalışır ve her kapanan mumda stratejinin kararını (AL / BEKLE / SAT) kaydeder. "
           "Yeterince gün ve işlem birikince sanal sonuç değerlendirilir. Aşağıdaki düğme yalnız ayrıntıyı gösterir.")
DISCLAIMER = "Sanal takip gerçek işlem değildir; geçmiş sonuç gelecekteki kazanç olasılığı değildir."


def _aware(moment: datetime) -> datetime:
    """Saat dilimi olmayan zaman UTC sayılır (depo UTC yazar)."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def friendly_time(moment: datetime, now: datetime) -> str:
    """`5 Eki 08:52 · 3 saat önce` biçimi; İstanbul saati (spec 0012). İç içe parantez üretmez."""
    moment, now = _aware(moment), _aware(now)
    local = moment.astimezone(ISTANBUL)
    age = now - moment
    if age < timedelta(hours=1):
        ago = "az önce"
    elif age < timedelta(days=1):
        ago = f"{int(age.total_seconds() // 3600)} saat önce"
    else:
        ago = f"{age.days} gün önce"
    return f"{local.day} {_MONTHS[local.month - 1]} {local:%H:%M} · {ago}"


@dataclass(frozen=True)
class StatusView:
    """Üstte görünen kısa durum: her satır (düzey, metin); düzey `ok`/`warn`/`info`."""
    headline: list[tuple[str, str]]
    notice: str            # ör. "9 varlıkta işlem varsayımı eksik …"; yoksa boş
    details: list[str]     # "Ayrıntı" bölümünde listelenenler


def _split_note(note: str) -> tuple[list[str], list[str], list[str]]:
    """Koşucu notunu (`;` ile birleşik) iki gruba ayırır: eksik varsayım olan semboller ve diğer notlar."""
    missing, others, blocked = [], [], []
    # Not "SEMBOL: ...; ..." biçiminde birleştirilmiş cümlelerdir; yalnız yeni "SEMBOL:" başında bölünür,
    # böylece bir notun kendi içindeki ";" (… eksik; sanal işlem hesaplanmadı.) parçalanmaz.
    for part in (p.strip() for p in re.split(r";\s+(?=[A-Za-z0-9_.=/\-]+:\s)", note)):
        if not part:
            continue
        if _MISSING_SETTINGS in part:
            missing.append(part.split(":")[0].strip())
        elif _STEP_TOO_BIG in part:
            blocked.append(part.split(":")[0].strip())
        else:
            others.append(part.rstrip("."))
    return missing, others, blocked


def build_status_view(now: datetime) -> StatusView:
    runs = ft.recent_runs(1)
    last = ft.last_successful_run()
    headline: list[tuple[str, str]] = []
    if last is None:
        headline.append(("info", "Takip henüz çalışmadı. İlk çalışma zamanlanmış görevle gerçekleşir."))
    elif now - last > STALE_AFTER:
        headline.append(("warn", f"Takip 2 günden uzun süredir çalışmadı (son: {friendly_time(last, now)}); "
                                 "durmuş olabilir."))
    else:
        headline.append(("ok", f"Takip çalışıyor · son başarılı çalışma: {friendly_time(last, now)}"))
    notice, details = "", []
    if runs:
        latest = runs[0]
        missing, others, blocked = _split_note(latest["note"]) if latest["note"] != "tamam" else ([], [], [])
        if not latest["ok"]:
            when = friendly_time(latest["run_at"], now)
            headline.append(("warn", f"Son çalışma başarısız ({when}): {'; '.join(others) or latest['note']}"))
            others = []
        if missing:
            notice = (f"{len(missing)} varlıkta işlem varsayımı (miktar adımı) girilmemiş: bu varlıklar için "
                      "sanal kâr/zarar hesaplanmıyor; kararlar yine kaydediliyor.")
            details.append("Varsayımı eksik: " + ", ".join(missing))
        if blocked:
            notice = (notice + " " if notice else "") + (
                f"{len(blocked)} varlıkta adet/lot adımı işlem tutarını aşıyor ({', '.join(blocked)}): hiç alım "
                "yapılamaz; adım en küçük alınabilir miktar olmalı (ör. ETH 0.001).")
        details.extend(others)
    return StatusView(headline, notice, details)


def render_forward_status(now: datetime) -> None:
    import streamlit as st

    from storage import StorageAccessError

    try:
        view = build_status_view(now)
    except StorageAccessError:
        st.warning("İleri takip kayıtları okunamadı; kayıt deposu erişimini kontrol edin.")
        return
    for level, text in view.headline:
        {"ok": st.success, "warn": st.warning}.get(level, st.info)(text)
    st.caption(EXPLAIN)
    if view.notice:
        st.caption("ℹ️ " + view.notice)
    if view.details:
        with st.expander("Son çalışma ayrıntısı"):
            for line in view.details:
                st.caption(line)


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


_DECISION_ICON = {"AL": "🟢 AL", "SAT": "🔴 SAT", "BEKLE": "⚪ BEKLE"}


def _short_day(day) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]}"


def build_summary_rows(pairs=None) -> list[dict]:
    """Varlık başına tek satır: son karar, birikim sayaçları ve durum (spec 0012)."""
    rows = []
    shown = (pairs if pairs is not None else ft.tracked_pairs())[:MAX_PAIRS]
    for asset, version in shown:
        decided = ft.decisions(asset, version)
        verdict = ft.assess(asset, version)
        counted = [d for d in decided if d.on_time and d.real_clock]
        regimes = {d.regime for d in counted if d.regime}
        last = decided[-1] if decided else None
        rows.append({
            "Varlık": asset if sum(a == asset for a, _ in shown) == 1 else f"{asset} ({version})",
            "Son karar": f"{_DECISION_ICON.get(last.decision, last.decision)} · {_short_day(last.candle_day)}"
                         if last else "—",
            "Gün": f"{len(counted)} / {ForwardConfig.MIN_TRACKED_DAYS}",
            "İşlem": f"{ft.closed_trades(asset, version)} / {ForwardConfig.MIN_CLOSED_TRADES}",
            "Koşul": f"{len(regimes)} / 3",
            "Durum": "✅ Yeterli kanıt" if verdict.sufficient else "⏳ Birikiyor",
            "_asset": asset, "_version": version,
        })
    return rows


def build_detail(asset: str, version: str) -> dict:
    """Seçilen varlığın ayrıntısı: son kararlar, kayıtlı varsayımlar, uyarılar."""
    decided = ft.decisions(asset, version)
    recent = decided[-RECENT:]
    decisions_table = [{"Gün": d.candle_day.isoformat(), "Karar": _DECISION_ICON.get(d.decision, d.decision),
                        "Kayıt": ft.status_text(d), "Kaynak": d.source} for d in reversed(recent)]
    assumptions = []
    if recent and recent[-1].assumptions:
        assumptions = [{"Varsayım": key, "Değer": ("girilmedi" if value == "bilinmiyor" else value)}
                       for key, value in recent[-1].assumptions.items()]
    warnings = []
    gaps = ft.missing_days(asset, version)
    if gaps:
        warnings.append(f"Eksik görünen gün: {len(gaps)} (ilki {gaps[0]}; resmi tatiller de eksik görünebilir).")
    for revision in ft.revisions(asset, version):
        warnings.append(f"{revision['candle_day']}: mum sağlayıcıda revize edildi; karar değişmedi.")
    verdict = ft.assess(asset, version)
    return {"decisions": decisions_table, "assumptions": assumptions, "warnings": warnings,
            "missing": [] if verdict.sufficient else list(verdict.missing)}


def render_forward_panel(now: datetime) -> None:
    import pandas as pd
    import streamlit as st

    from storage import StorageAccessError

    try:
        pairs = ft.tracked_pairs()
        rows = build_summary_rows(pairs)
        if not rows:
            st.info("Henüz kayıtlı karar yok. İlk karar, zamanlanmış görev çalışınca burada görünür.")
            st.caption(DISCLAIMER)
            return
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]),
                     width="stretch", hide_index=True)
        st.caption("Gün: izlenen gün · İşlem: kapanmış sanal işlem · Koşul: görülen piyasa koşulu "
                   "(yükselen / düşen / yatay). Soldaki sayı birikeni, sağdaki hedefi gösterir.")
        if len(pairs) > MAX_PAIRS:
            st.caption(f"{len(pairs) - MAX_PAIRS} varlık daha izleniyor; tabloda ilk {MAX_PAIRS} gösterilir.")
        names = [r["Varlık"] for r in rows]
        picked = st.selectbox("Ayrıntı için varlık seçin", names, key="forward_detail_asset")
        chosen = next(r for r in rows if r["Varlık"] == picked)
        asset, version = chosen["_asset"], chosen["_version"]
        detail = build_detail(asset, version)
    except StorageAccessError:
        st.warning("İleri takip kayıtları okunamadı; kayıt deposu erişimini kontrol edin.")
        return
    st.markdown(f"**{picked} · son {RECENT} karar**")
    st.dataframe(pd.DataFrame(detail["decisions"]), width="stretch", hide_index=True)
    if detail["missing"]:
        st.caption("Değerlendirme için eksik: " + ", ".join(detail["missing"]) + ".")
    for line in detail["warnings"]:
        st.caption("⚠️ " + line)
    if detail["assumptions"]:
        with st.expander("Son kararın varsayımları"):
            st.dataframe(pd.DataFrame(detail["assumptions"]), width="stretch", hide_index=True)
    st.caption(DISCLAIMER)
