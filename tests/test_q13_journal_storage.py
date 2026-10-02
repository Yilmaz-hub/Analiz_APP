"""Q13 (spec 0003): işlem günlüğü kalıcı depoda; okunamazsa boş günlükle devam yok."""
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

import storage
from app_helpers import break_storage, break_writes_only, make_app, texts
from position_journal import PositionJournal

UTC = timezone.utc
AT = datetime(2026, 9, 9, 13, 30, tzinfo=UTC)
REC = datetime(2026, 9, 10, 8, tzinfo=UTC)


def _buy(journal, event_id="buy-1", symbol="BTC-USD"):
    return journal.confirm_trade(event_id, "BUY", Decimal("2"), Decimal("100"), AT, REC, symbol=symbol)


def test_q13_journal_is_written_to_the_record_store_not_a_local_file(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _buy(PositionJournal())
    assert storage.read_doc(storage.JOURNAL_KEY)["trades"][0]["event_id"] == "buy-1"
    assert not (tmp_path / "position_journal.json").exists()


def test_q13_journal_survives_restart_through_the_store(store):
    _buy(PositionJournal())
    assert PositionJournal().positions["BTC-USD"].quantity == Decimal("2")


def test_q13_local_journal_file_is_imported_once(store, legacy_dir):
    (legacy_dir / "position_journal.json").write_text(json.dumps({
        "trades": [{"event_id": "old-1", "side": "BUY", "quantity": "1", "price": "50",
                    "executed_at": AT.isoformat(), "recorded_at": REC.isoformat(),
                    "fee": None, "symbol": "ETH-USD"}],
        "cooldown_bars": 0}), encoding="utf-8")

    assert storage.JOURNAL_KEY in storage.import_legacy_documents()
    assert PositionJournal().positions["ETH-USD"].entry_price == Decimal("50")

    # İkinci açılışta depodaki kayıt üzerine yazılmaz.
    journal = PositionJournal()
    journal.confirm_trade("new-1", "SELL", Decimal("1"), Decimal("60"), AT, REC, symbol="ETH-USD")
    assert storage.JOURNAL_KEY not in storage.import_legacy_documents()
    assert [t.event_id for t in PositionJournal().trades] == ["old-1", "new-1"]


def test_q13_unreadable_store_does_not_yield_an_empty_journal(store):
    _buy(PositionJournal())
    geri_al = break_storage(store)
    try:
        with pytest.raises(storage.StorageAccessError):
            PositionJournal()
    finally:
        geri_al()


def test_q13_corrupt_journal_document_is_not_replaced_by_an_empty_one(store):
    store.write_doc(storage.JOURNAL_KEY, {"trades": [{"event_id": "x"}]})
    with pytest.raises(storage.StorageAccessError):
        PositionJournal()
    assert store.read_doc(storage.JOURNAL_KEY) == {"trades": [{"event_id": "x"}]}


def test_q13_failed_write_rolls_back_memory_and_raises(store):
    journal = PositionJournal()
    geri_al = break_writes_only(store)
    try:
        with pytest.raises(storage.StorageAccessError):
            _buy(journal)
    finally:
        geri_al()
    assert journal.trades == [] and journal.positions == {}
    assert PositionJournal().trades == []


def test_q13_app_with_corrupt_journal_disables_trade_entry_and_keeps_records(
        store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    bozuk = {"trades": [{"event_id": "x"}]}
    store.write_doc(storage.JOURNAL_KEY, bozuk)

    at = make_app(monkeypatch, processed_df).run()

    assert not at.exception
    assert any("günlüğü okunamadı" in e.value for e in at.sidebar.error)
    assert all(b.disabled for b in at.button if b.label == "➕ Emri Gir / Ekle")
    assert store.read_doc(storage.JOURNAL_KEY) == bozuk
