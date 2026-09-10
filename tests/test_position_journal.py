from datetime import datetime, timedelta, timezone
from decimal import Decimal

from position_journal import PositionJournal


UTC = timezone.utc
TRADE_AT = datetime(2026, 9, 9, 13, 30, tzinfo=UTC)
RECORDED_AT = datetime(2026, 9, 10, 8, tzinfo=UTC)


def _journal(tmp_path):
    return PositionJournal(tmp_path / "journal.json")


def _buy(journal, event_id="buy-1", fee=None):
    return journal.confirm_trade(event_id, "BUY", Decimal("2"), Decimal("100"), TRADE_AT, RECORDED_AT, fee=fee)


def test_requirement_signal_does_not_open_real_position(tmp_path):
    journal = _journal(tmp_path); journal.apply_signal("AL")
    assert journal.position is None


def test_ac40_signal_does_not_close_real_position(tmp_path):
    journal = _journal(tmp_path); _buy(journal); journal.apply_signal("SAT")
    assert journal.position.quantity == Decimal("2")


def test_ac41_duplicate_confirmation(tmp_path):
    journal = _journal(tmp_path)
    assert _buy(journal) is True and _buy(journal) is False and len(journal.trades) == 1


def test_ac42_correction_keeps_execution(tmp_path):
    journal = _journal(tmp_path); _buy(journal); journal.correct_trade("buy-1", price=Decimal("101"))
    assert len(journal.trades) == 1 and journal.trades[0].price == Decimal("101")


def test_ac43_paper_real_isolation(tmp_path):
    journal = _journal(tmp_path); journal.record_paper("BUY", Decimal("4"), Decimal("90"))
    assert journal.position is None


def test_ac77_ignored_buy_no_real_trade(tmp_path):
    journal = _journal(tmp_path); journal.ignore_signal("AL")
    assert journal.trades == []


def test_ac78_late_entry_preserves_trade_time(tmp_path):
    journal = _journal(tmp_path); _buy(journal)
    assert journal.trades[0].executed_at == TRADE_AT and journal.trades[0].recorded_at == RECORDED_AT


def test_ac81_late_fee_preserves_trade(tmp_path):
    journal = _journal(tmp_path); _buy(journal); journal.add_fee("buy-1", Decimal("2"))
    trade = journal.trades[0]
    assert (trade.quantity, trade.price, trade.executed_at, trade.fee) == (Decimal("2"), Decimal("100"), TRADE_AT, Decimal("2"))


def test_ac88_late_fee_no_new_cooldown(tmp_path):
    journal = _journal(tmp_path); _buy(journal); journal.add_fee("buy-1", Decimal("999"))
    assert journal.cooldown_bars == 0


def test_ac89_late_fee_preserves_cooldown(tmp_path):
    journal = _journal(tmp_path); journal.set_cooldown(1); _buy(journal); journal.add_fee("buy-1", Decimal("2"))
    assert journal.cooldown_bars == 1


def test_ac103_real_quantity_unchanged(tmp_path):
    journal = _journal(tmp_path); _buy(journal)
    journal.set_paper_quantity_step(Decimal("1"))
    assert journal.trades[0].quantity == Decimal("2")


def test_ac114_prospective_cost(tmp_path):
    journal = _journal(tmp_path); journal.record_paper("BUY", Decimal("1"), Decimal("100"), spread_bps=Decimal("10"))
    journal.set_paper_costs(spread_bps=Decimal("20")); journal.record_paper("BUY", Decimal("1"), Decimal("100"))
    assert [item.spread_bps for item in journal.paper_trades] == [Decimal("10"), Decimal("20")]


def test_ac122_legacy_record(tmp_path):
    journal = _journal(tmp_path)
    journal.import_legacy([{"id": "old-1", "auto_closed": True, "pnl": "25"}])
    assert journal.verified_real_pnl == Decimal("0") and journal.legacy_records[0].status == "ESKI_TEYITSIZ"


def test_invalid_confirmation_is_rejected(tmp_path):
    journal = _journal(tmp_path)
    assert journal.confirm_trade("bad", "BUY", Decimal("0"), Decimal("100"), TRADE_AT, RECORDED_AT) is False


def test_journal_survives_restart(tmp_path):
    path = tmp_path / "journal.json"; first = PositionJournal(path); _buy(first)
    second = PositionJournal(path)
    assert second.position.quantity == Decimal("2") and second.trades[0].executed_at == TRADE_AT


def test_real_positions_are_kept_separate_by_asset(tmp_path):
    journal = _journal(tmp_path)
    journal.confirm_trade("eth-buy", "BUY", Decimal("2"), Decimal("100"), TRADE_AT, RECORDED_AT, symbol="ETH/USD")
    journal.confirm_trade("xau-buy", "BUY", Decimal("1"), Decimal("2000"), TRADE_AT, RECORDED_AT, symbol="XAU_GOLD")
    journal.confirm_trade("eth-sell", "SELL", Decimal("2"), Decimal("110"), TRADE_AT, RECORDED_AT, symbol="ETH/USD")

    restarted = PositionJournal(journal.path)

    assert "ETH/USD" not in restarted.positions
    assert restarted.positions["XAU_GOLD"].entry_price == Decimal("2000")
