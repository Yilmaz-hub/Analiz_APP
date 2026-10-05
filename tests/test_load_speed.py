"""Spec 0008 — yayında sayfa yükleme hızı (kayıt deposu gidiş-dönüşleri, uyuyan bağlantı)."""
import pytest
from sqlalchemy import event

import forward_tracker as ft
import storage


def _record_statements(engine):
    seen = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(conn, cursor, statement, parameters, context, executemany):
        seen.append(statement)

    return seen


def test_ac01_forward_tables_are_created_once_per_process():
    """AC01 — İleri takip işlevleri art arda çağrıldığında ilk çağrıdan sonra CREATE TABLE sorgusu çalışmaz."""
    engine = storage.get_engine()
    seen = _record_statements(engine)
    ft.tracked_pairs()
    first = [s for s in seen if s.lstrip().upper().startswith("CREATE")]
    assert first                                     # ilk çağrıda tablolar oluşur
    seen.clear()
    ft.tracked_pairs()
    ft.recent_runs() if hasattr(ft, "recent_runs") else None
    ft.tracked_pairs()
    assert not [s for s in seen if s.lstrip().upper().startswith("CREATE")]


def test_ac03_connection_pool_checks_and_recycles_connections():
    """AC03 — Havuz bağlantıyı kullanmadan önce sınar ve en çok 5 dakikada bir yeniler."""
    pool = storage.get_engine().pool
    assert pool._pre_ping is True
    assert 0 < pool._recycle <= 300


def _forward_selects(seen):
    return [s for s in seen if s.lstrip().upper().startswith("SELECT") and "forward_" in s]


def _app_with_assets(store, monkeypatch, processed_df):
    from app_helpers import make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    return make_app(monkeypatch, processed_df)


def test_ac02_app_does_not_query_forward_details_until_asked(store, monkeypatch, processed_df):
    """AC02 — Varsayılan açılışta ileri takip ayrıntı tablolarına sorgu gitmez; "göster" açılınca panel içeriği görünür."""
    from app_helpers import texts

    app = _app_with_assets(store, monkeypatch, processed_df)
    seen = _record_statements(storage.get_engine())
    at = app.run()
    assert not at.exception
    assert not [s for s in _forward_selects(seen) if "forward_decisions" in s or "forward_trades" in s]
    assert "Sanal takip gerçek işlem değildir" not in texts(at)
    seen.clear()
    at = next(t for t in at.toggle if t.label == "Ayrıntıları göster (takibi açıp kapatmaz)").set_value(True).run()
    assert not at.exception
    assert "Sanal takip gerçek işlem değildir" in texts(at)             # panel içeriği görünür
    assert [s for s in seen if "forward_decisions" in s]


def test_ac04_failed_last_run_is_visible_without_opening_the_panel(store, monkeypatch, processed_df):
    """AC04 — Başarısız son çalışma uyarısı toggle açılmadan, en çok 2 ucuz sorguyla görünür."""
    from datetime import datetime, timezone

    from app_helpers import texts

    ft.record_run(datetime.now(timezone.utc), False, "sağlayıcı hatası")
    app = _app_with_assets(store, monkeypatch, processed_df)
    seen = _record_statements(storage.get_engine())
    at = app.run()
    assert not at.exception
    assert "Son çalışma başarısız" in texts(at) and "sağlayıcı hatası" in texts(at)
    assert len(_forward_selects(seen)) <= 2


def test_ac04_stale_runner_is_visible_without_opening_the_panel(store, monkeypatch, processed_df):
    """AC04 — Koşucu 2 günden uzun süredir çalışmadıysa uyarı toggle açılmadan görünür."""
    from datetime import datetime, timedelta, timezone

    from app_helpers import texts

    ft.record_run(datetime.now(timezone.utc) - timedelta(days=5), True, "tamam")
    at = _app_with_assets(store, monkeypatch, processed_df).run()
    assert "2 günden uzun süredir çalışmadı" in texts(at)


def test_ac04_healthy_runner_shows_last_success_and_no_warning(store, monkeypatch, processed_df):
    """AC04 — Koşucu düzgün çalışıyorsa son başarılı çalışma görünür, uyarı görünmez."""
    from datetime import datetime, timezone

    from app_helpers import texts

    ft.record_run(datetime.now(timezone.utc), True, "tamam")
    at = _app_with_assets(store, monkeypatch, processed_df).run()
    shown = texts(at)
    assert "Takip çalışıyor · son başarılı çalışma" in shown and "Son çalışma başarısız" not in shown
    assert "süredir çalışmadı" not in shown
