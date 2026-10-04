"""Spec 0005 Rev 5 — B13: "geç kayıtla güncellendi" işareti ekranda (AC76, Q08)."""
from datetime import date, datetime, timezone
from decimal import Decimal

import performance_ui
from position_journal import PositionJournal


def _journal_with_late_trade(symbol="BTC-USD"):
    journal = PositionJournal()
    executed = datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc)
    recorded = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    journal.confirm_trade("late-1", "BUY", Decimal("1"), Decimal("100"), executed, recorded, symbol=symbol)
    return journal


def test_ac76_period_with_retroactive_real_trade_is_flagged(store):
    """AC76 — Kapanmış döneme geriye dönük kaydedilmiş gerçek işlem varsa rapor "geç kayıtla güncellendi" notu taşır."""
    journal = _journal_with_late_trade()
    notes = performance_ui.late_record_notes(journal.trades, "BTC-USD", date(2026, 9, 1), date(2026, 9, 20))
    assert notes == ["Bu dönem geç kayıtla güncellendi: dönem kapandıktan sonra geriye dönük gerçek işlem "
                     "kaydedildi; ileri dönem yeterlilik sayacı bundan etkilenmez."]


def test_ac76_period_without_retroactive_trade_or_other_asset_has_no_flag(store):
    """AC76 — İşlem dönem içinde kaydedildiyse ya da başka varlığa aitse işaret yoktur."""
    journal = _journal_with_late_trade()
    assert performance_ui.late_record_notes(journal.trades, "BTC-USD", date(2026, 9, 1), date(2026, 9, 30)) == []
    assert performance_ui.late_record_notes(journal.trades, "ETH-USD", date(2026, 9, 1), date(2026, 9, 20)) == []


def test_ac76_flag_does_not_change_the_forward_counter(store):
    """AC76 — Geç kayıt ileri dönem yeterlilik sayacını değiştirmez."""
    import forward_tracker as ft

    before = ft.tracked_days("BTC-USD", "V1")
    _journal_with_late_trade()
    assert ft.tracked_days("BTC-USD", "V1") == before == 0
