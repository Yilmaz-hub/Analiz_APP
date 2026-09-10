from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trade_execution import (
    Bar,
    CostAssumptions,
    adjusted_price,
    execute_next_open,
    execute_purchase,
    evaluate_stop,
    round_stop_up,
    validate_costs,
)


UTC = timezone.utc
KNOWN = datetime(2026, 9, 9, 21, tzinfo=UTC)
NEXT = datetime(2026, 9, 10, 13, 30, tzinfo=UTC)


def _bar(open="100", high="110", low="90", close="105", at=NEXT):
    return Bar(at, Decimal(open), Decimal(high), Decimal(low), Decimal(close))


def test_ac28_no_fill_before_signal_known():
    with pytest.raises(ValueError, match="SINYALDEN_ONCE"):
        execute_next_open(KNOWN, _bar(at=KNOWN - timedelta(seconds=1)), "BUY")


def test_ac29_next_daily_open():
    fill = execute_next_open(KNOWN, _bar(open="103"), "BUY")
    assert (fill.at, fill.base_price) == (NEXT, Decimal("103"))


def test_ac30_intrabar_stop_trigger():
    fill = evaluate_stop(_bar(open="105", low="89", close="108"), Decimal("90"), KNOWN)
    assert fill.reason == "STOP" and fill.base_price == Decimal("90")


def test_ac31_stop_target_collision():
    fill = evaluate_stop(_bar(high="120", low="89"), Decimal("90"), KNOWN, info_target=Decimal("115"))
    assert fill.reason == "STOP"


def test_ac32_stop_gap_fill():
    assert evaluate_stop(_bar(open="85", low="80"), Decimal("90"), KNOWN).base_price == Decimal("85")


def test_ac51_exact_stop_touch():
    assert evaluate_stop(_bar(low="90"), Decimal("90"), KNOWN).reason == "STOP"


def test_ac52_target_touch_no_sale():
    assert evaluate_stop(_bar(low="95", high="115"), Decimal("90"), KNOWN, info_target=Decimal("115")) is None


def test_ac53_target_gap_no_sale():
    assert evaluate_stop(_bar(open="120", low="110", high="125"), Decimal("90"), KNOWN, info_target=Decimal("115")) is None


def test_ac55_preentry_touch_no_exit():
    assert evaluate_stop(_bar(low="80"), Decimal("90"), NEXT + timedelta(seconds=1)) is None


def test_ac56_explicit_zero_fee():
    costs = CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0"))
    assert validate_costs(costs).is_complete


def test_ac57_no_double_spread():
    costs = CostAssumptions(Decimal("100"), Decimal("25"), Decimal("0"))
    assert adjusted_price(Decimal("100.75"), "BUY", costs, costs_included=True) == Decimal("100.75")


def test_ac59_cash_including_fees():
    result = execute_purchase(Decimal("1000"), Decimal("1000"), Decimal("100"), fee=Decimal("1"), quantity_step=Decimal("1"))
    assert not result.executed and result.reason == "YETERSIZ_NAKIT"


def test_ac75_stop_raise_effective_time():
    assert evaluate_stop(_bar(low="95"), Decimal("98"), NEXT + timedelta(hours=1)) is None


def test_ac94_fee_added_to_notional():
    result = execute_purchase(Decimal("10000"), Decimal("1000"), Decimal("100"), fee=Decimal("2"), quantity_step=Decimal("1"))
    assert (result.spent, result.cash_after) == (Decimal("1000"), Decimal("8998"))


def test_ac95_exact_cash_with_fee():
    exact = execute_purchase(Decimal("1002"), Decimal("1000"), Decimal("100"), fee=Decimal("2"), quantity_step=Decimal("1"))
    short = execute_purchase(Decimal("1001.99"), Decimal("1000"), Decimal("100"), fee=Decimal("2"), quantity_step=Decimal("1"))
    assert exact.executed and not short.executed


def test_ac99_real_fill_no_double_slippage():
    costs = CostAssumptions(Decimal("100"), Decimal("25"), Decimal("0"))
    assert adjusted_price(Decimal("97.42"), "SELL", costs, confirmed_real=True) == Decimal("97.42")


def test_ac100_quantity_round_down():
    result = execute_purchase(Decimal("1000"), Decimal("1000"), Decimal("300"), fee=Decimal("0"), quantity_step=Decimal("1"))
    assert (result.quantity, result.spent, result.cash_after) == (Decimal("3"), Decimal("900"), Decimal("100"))


def test_ac108_open_stop_priority():
    fill = evaluate_stop(_bar(open="90", low="85"), Decimal("90"), KNOWN, pending_sell=True)
    assert (fill.reason, fill.base_price, fill.exit_count) == ("STOP", Decimal("90"), 1)


def test_ac109_entry_bar_stop():
    fill = evaluate_stop(_bar(open="100", low="90"), Decimal("90"), NEXT, entered_at=NEXT)
    assert fill.reason == "STOP" and fill.base_price == Decimal("90")


def test_ac110_invalid_stop():
    result = execute_purchase(Decimal("10000"), Decimal("1000"), Decimal("100"), fee=Decimal("0"), quantity_step=Decimal("1"), initial_stop=Decimal("0"))
    assert not result.executed and result.reason == "GECERSIZ_STOP"


def test_ac111_cost_formula():
    costs = CostAssumptions(Decimal("100"), Decimal("25"), Decimal("0"))
    assert adjusted_price(Decimal("100"), "BUY", costs) == Decimal("100.7500")
    assert adjusted_price(Decimal("100"), "SELL", costs) == Decimal("99.2500")


def test_ac112_cost_boundary():
    assert not validate_costs(CostAssumptions(Decimal("10000"), Decimal("5000"), Decimal("0"))).is_valid
    assert validate_costs(CostAssumptions(Decimal("10000"), Decimal("4999"), Decimal("0"))).is_valid


def test_ac115_stop_rounding():
    assert round_stop_up(Decimal("90.03"), Decimal("0.05"), Decimal("100")) == Decimal("90.05")
    assert round_stop_up(Decimal("99.99"), Decimal("1"), Decimal("100")) is None


@pytest.mark.parametrize("spread,slippage", [(Decimal("-1"), Decimal("0")), (Decimal("0"), Decimal("-1"))])
def test_requirement_negative_market_cost_rejected(spread, slippage):
    assert not validate_costs(CostAssumptions(spread, slippage, Decimal("0"))).is_valid
