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


# ---- Spec 0012: sadeleştirilmiş ekran ------------------------------------------------------------
USER_NOTE = "; ".join(
    [f"{s}: miktar adımı/varsayımlar eksik; sanal işlem hesaplanmadı."
     for s in ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "PEPE-USD", "XAU_GOLD",
               "GRAM_TRY", "THYAO.IS", "PGSUS.IS")] + ["EURUSD=X: piyasası tanınmıyor, atlandı."])


def test_ac01_friendly_time_is_istanbul_local_and_relative():
    """AC01 — `friendly_time` İstanbul saatiyle "5 Eki 08:52 (3 saat önce)" üretir."""
    moment = datetime(2026, 10, 5, 5, 52, tzinfo=timezone.utc)            # İstanbul +3 → 08:52
    assert forward_ui.friendly_time(moment, moment + timedelta(hours=3)) == "5 Eki 08:52 (3 saat önce)"
    assert forward_ui.friendly_time(moment, moment + timedelta(minutes=10)).endswith("(az önce)")
    assert forward_ui.friendly_time(moment, moment + timedelta(days=2)).endswith("(2 gün önce)")


def test_ac02_status_view_covers_never_ok_stale_and_failed():
    """AC02 — Durum satırı: hiç çalışmadı → bilgi; sağlıklı → yeşil; 2 günden eski → uyarı; başarısız son çalışma ayrı uyarı."""
    now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    assert forward_ui.build_status_view(now).headline[0][0] == "info"
    ft.record_run(now - timedelta(hours=3), True, "tamam")
    head = forward_ui.build_status_view(now).headline
    assert head[0][0] == "ok" and "Takip çalışıyor" in head[0][1] and "3 saat önce" in head[0][1]
    assert forward_ui.build_status_view(now + timedelta(days=3)).headline[0][0] == "warn"
    ft.record_run(now, False, "BTC-USD: veri alınamadı (Binance).")
    levels = [level for level, _ in forward_ui.build_status_view(now).headline]
    assert "warn" in levels[1:]


def test_ac03_missing_settings_notes_collapse_into_one_summary():
    """AC03 — 11 varlığın eksik-varsayım notu tek özet cümle ve tek ayrıntı satırı olur; diğer notlar korunur."""
    now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    ft.record_run(now - timedelta(hours=1), True, USER_NOTE)
    view = forward_ui.build_status_view(now)
    assert view.notice.startswith("11 varlıkta işlem varsayımı")
    assert sum("Varsayımı eksik" in d for d in view.details) == 1
    assert any("EURUSD=X: piyasası tanınmıyor" in d for d in view.details)
    assert "miktar adımı/varsayımlar eksik" not in " ".join(view.details)


def test_ac04_summary_has_one_row_per_pair_with_counters():
    """AC04 — Her varlık-sürüm çifti için bir satır; sayaçlar doğru."""
    at = ft.available_at("KRIPTO", D0) + timedelta(minutes=20)
    _record(D0, at, decision="AL")
    rows = forward_ui.build_summary_rows()
    assert len(rows) == 1 and rows[0]["Varlık"] == "BTC-USD"
    assert rows[0]["Son karar"] == "🟢 AL · 1 Haz"
    assert rows[0]["Gün"] == "1 / 90" and rows[0]["İşlem"] == "0 / 30" and rows[0]["Koşul"] == "1 / 3"
    assert rows[0]["Durum"] == "⏳ Birikiyor"


def test_ac05_detail_shows_recent_decisions_and_readable_assumptions():
    """AC05 — Seçili varlığın son kararları okunur; `bilinmiyor` varsayımı `girilmedi` yazılır."""
    at = ft.available_at("KRIPTO", D0) + timedelta(minutes=20)
    ft.record(ft.ForwardDecision(
        asset="BTC-USD", strategy_version="V1", candle_day=D0, decision="SAT", source="Binance",
        evaluated_at=at, candle={"close": "100"}, assumptions={"komisyon": "bilinmiyor", "dolum": "ertesi gün açılışı"},
        on_time=True, real_clock=True, regime="YUKSELEN", regime_version="REJIM-1"))
    detail = forward_ui.build_detail("BTC-USD", "V1")
    assert detail["decisions"][0] == {"Gün": D0.isoformat(), "Karar": "🔴 SAT", "Kayıt": "zamanında", "Kaynak": "Binance"}
    assert {"Varsayım": "komisyon", "Değer": "girilmedi"} in detail["assumptions"]
    assert any("izlenen gün 1/90" in m for m in detail["missing"])


def test_ac06_screen_is_readable_not_a_wall_of_text(store, monkeypatch, processed_df):
    """AC06 — Ekranda ham ISO zamanı ve tekrar eden uzun not yok; açıklama, özet cümle ve tablo var."""
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    now = datetime.now(timezone.utc)
    ft.record_run(now - timedelta(hours=2), True, USER_NOTE)
    at = ft.available_at("KRIPTO", D0) + timedelta(minutes=20)
    _record(D0, at, decision="AL")
    app = make_app(monkeypatch, processed_df).run()
    assert not app.exception
    shown = texts(app)
    assert "Takip çalışıyor" in shown and "11 varlıkta işlem varsayımı" in shown
    assert forward_ui.EXPLAIN in shown
    assert "T05:" not in shown and "+00:00" not in shown                       # ham ISO yok
    assert shown.count("miktar adımı/varsayımlar eksik") == 0
    app = next(t for t in app.toggle if t.label == "Ayrıntıları göster (takibi açıp kapatmaz)").set_value(True).run()
    table = next(f.value for f in app.dataframe if "Son karar" in f.value.columns)
    assert list(table["Varlık"]) == ["BTC-USD"]
