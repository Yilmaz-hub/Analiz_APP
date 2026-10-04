"""Spec 0005 Adım 7 — ileri takip paneli (AC36, AC38, Q06a son çalışma görünürlüğü)."""
from datetime import date, datetime, timedelta, timezone

import forward_tracker as ft
import forward_ui

D0 = date(2026, 6, 1)


def _record(day, evaluated_at, decision="BEKLE"):
    ft.record(ft.ForwardDecision(
        asset="BTC-USD", strategy_version="V1", candle_day=day, decision=decision,
        source="Binance", evaluated_at=evaluated_at, candle={"close": "100"},
        on_time=ft.is_on_time("KRIPTO", day, evaluated_at), real_clock=True,
        regime="YUKSELEN", regime_version="REJIM-1"))


def _text(now):
    return " | ".join(forward_ui.build_forward_view(now).lines)


def test_ac36_panel_shows_source_times_and_version_of_each_decision():
    """AC36 — Panelde ileri dönem kararının kaynağı, karar zamanı, mum günü ve sürümü görünür."""
    at = ft.available_at("KRIPTO", D0) + timedelta(minutes=20)
    _record(D0, at, decision="AL")
    text = _text(at)
    assert f"{D0} · AL · V1 · kaynak Binance · karar {at.isoformat()} · zamanında" in text


def test_ac38_backfilled_decision_is_labelled_in_panel():
    """AC38 — Sonradan tamamlanan karar panelde "sonradan oluşturuldu" diye görünür."""
    late = ft.available_at("KRIPTO", D0) + timedelta(days=3)
    _record(D0, late)
    assert "sonradan oluşturuldu" in _text(late)


def test_panel_shows_last_successful_run_and_warns_when_stale():
    """Q06a — Son başarılı çalışma zamanı görünür; 2 günden eskiyse takibin durduğu uyarılır."""
    assert "İleri takip koşucusu henüz çalışmadı" in _text(datetime(2026, 6, 1, tzinfo=timezone.utc))
    run_at = datetime(2026, 6, 2, 0, 30, tzinfo=timezone.utc)
    ft.record_run(run_at, True, "tamam")
    assert f"Son başarılı takip çalışması: {run_at.isoformat()}" in _text(run_at + timedelta(hours=3))
    assert "takip durmuş olabilir" in _text(run_at + timedelta(days=3))


def test_panel_shows_sufficiency_progress():
    """AC39–AC41 — Panel yeterlilik ilerlemesini ve eksik koşulları gösterir."""
    at = ft.available_at("KRIPTO", D0) + timedelta(minutes=20)
    _record(D0, at)
    text = _text(at)
    assert "BTC-USD · V1: yetersiz kanıt" in text and "izlenen gün 1/90" in text
