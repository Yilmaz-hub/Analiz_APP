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


def test_ac02_app_does_not_query_forward_tables_until_asked(store, monkeypatch, processed_df):
    """AC02 — Ana ekran varsayılan açılışta ileri takip tablolarına sorgu yapmaz; "göster" açılınca içerik görünür."""
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    engine = storage.get_engine()
    seen = _record_statements(engine)
    at = make_app(monkeypatch, processed_df).run()
    assert not at.exception
    assert not [s for s in seen if "forward_" in s]
    toggle = next(t for t in at.toggle if t.label == "Sanal takip durumunu göster")
    seen.clear()
    at = toggle.set_value(True).run()
    assert not at.exception
    assert [s for s in seen if "forward_" in s]
