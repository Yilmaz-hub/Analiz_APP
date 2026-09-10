from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trade_decisions import (
    Action,
    Position,
    PositionState,
    Protection,
    Signal,
    classify_exit,
    decide_action,
    initial_stop,
    validate_position,
)


NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


def _open_position(stop=Decimal("90")):
    return Position(PositionState.OPEN, Decimal("2"), Decimal("100"), NOW, stop)


def test_requirement_flat_buy_action():
    assert decide_action(Signal.BUY, Position.flat()).action is Action.BUY


def test_requirement_open_buy_means_hold():
    assert decide_action(Signal.BUY, _open_position()).action is Action.HOLD


def test_requirement_wait_preserves_position():
    assert decide_action(Signal.WAIT, _open_position()).action is Action.HOLD


def test_requirement_flat_wait_no_entry():
    assert decide_action(Signal.WAIT, Position.flat()).action is Action.NONE


def test_requirement_flat_sell_no_sale():
    assert decide_action(Signal.SELL, Position.flat()).action is Action.NONE


def test_requirement_stop_overrides_buy():
    result = decide_action(Signal.BUY, _open_position(), stop_touched=True)
    assert (result.action, result.reason) == (Action.SELL, "STOP")


def test_ac39_missing_entry_no_stop():
    invalid = Position(PositionState.OPEN, Decimal("2"), None, NOW, None)
    assert validate_position(invalid).protection is None


def test_ac44_single_exit_for_collision():
    result = decide_action(Signal.SELL, _open_position(), stop_touched=True)
    assert (result.action, result.exit_count, result.reason) == (Action.SELL, 1, "STOP")


def test_ac45_confirmed_stop_above_entry_valid():
    protection = Protection.updated(Decimal("105"), NOW)
    assert protection.validate(Decimal("100")).is_valid


def test_ac46_disabled_target_keeps_stop():
    result = decide_action(
        Signal.WAIT, _open_position(), target_enabled=False, stop_touched=False
    )
    assert result.active_stop == Decimal("90")


@pytest.mark.parametrize("stop", [Decimal("100"), Decimal("101")])
def test_ac47_invalid_initial_stop(stop):
    assert not Protection.initial(stop).validate(Decimal("100")).is_valid


def test_ac48_loss_cooldown_before_two():
    assert decide_action(Signal.BUY, Position.flat(), cooldown_bars=1).action is Action.NONE


def test_ac49_loss_cooldown_exact_two():
    assert decide_action(Signal.BUY, Position.flat(), cooldown_bars=2).action is Action.BUY


def test_ac50_profit_no_loss_cooldown():
    assert classify_exit(Decimal("5"), fee_known=True).cooldown_required is False


def test_ac74_stop_fixed_until_confirmation():
    position = _open_position()
    result = decide_action(Signal.BUY, position, suggested_stop=Decimal("98"))
    assert result.active_stop == Decimal("90")


def test_ac85_net_loss_after_known_fee():
    result = classify_exit(Decimal("1"), fee_known=True, fees=Decimal("2"))
    assert (result.net, result.cooldown_required) == (Decimal("-1"), True)


def test_ac87_breakeven_no_cooldown():
    assert classify_exit(Decimal("0"), fee_known=True).cooldown_required is False
