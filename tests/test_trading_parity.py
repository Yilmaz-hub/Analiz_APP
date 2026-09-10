from datetime import datetime, timedelta, timezone
from decimal import Decimal

from market_validation import MarketValidation
from trade_decisions import Position, Signal, initial_stop
from trading_service import DecisionLedger, PaperAccounts, TradingContext, TradingSettings, evaluate_all_modes, point_in_time_value


UTC = timezone.utc
NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _context(**changes):
    values = dict(signal=Signal.BUY, position=Position.flat(), data=MarketValidation(None, True, status="GECERLI", allow_new_action=True),
                  components={name: "READY" for name in ("trend", "momentum", "volume", "pattern", "advanced", "ml", "regime")},
                  cutoff=NOW, settings_version="v1")
    values.update(changes)
    return TradingContext(**values)


def test_requirement_decision_parity():
    results = evaluate_all_modes(_context())
    assert [result.action.value for result in results.values()] == ["AL", "AL", "AL"]


def test_requirement_stop_parity():
    results = evaluate_all_modes(_context(), entry=Decimal("100"), atr=Decimal("4"))
    assert {result.stop for result in results.values()} == {Decimal("90.0")}


def test_requirement_target_parity():
    results = evaluate_all_modes(_context())
    assert {result.target_enabled for result in results.values()} == {False}


def test_ac14_net_parity():
    results = evaluate_all_modes(_context(), gross=Decimal("12"), fees=Decimal("2"))
    assert {result.net for result in results.values()} == {Decimal("10")}


def test_ac66_no_parity_success_missing_component():
    components = _context().components | {"ml": "ERROR"}
    results = evaluate_all_modes(_context(components=components))
    assert not any(result.comparable for result in results.values())


def test_ac67_internal_net_precision():
    results = evaluate_all_modes(_context(), gross=Decimal("1.004"), fees=Decimal("0"))
    assert results["screen"].net != Decimal("1.00")


def test_ac68_timezone_same_instant():
    local = NOW.astimezone(timezone(timedelta(hours=3)))
    assert _context(cutoff=NOW).identity == _context(cutoff=local).identity


def test_ac69_repeat_paper_decision_once():
    ledger = DecisionLedger()
    assert ledger.record(_context().identity) is True
    assert ledger.record(_context().identity) is False


def test_ac83_initial_stop_atr_parity():
    results = evaluate_all_modes(_context(), entry=Decimal("100"), atr=Decimal("4"))
    assert {result.stop for result in results.values()} == {Decimal("90.0")}


def test_ac84_decision_candle_atr_frozen():
    assert initial_stop(Decimal("100"), Decimal("4"), later_atr=Decimal("6")) == Decimal("90.0")


def test_ac91_custom_paper_amounts():
    settings = TradingSettings(capital=Decimal("25000"), trade_notional=Decimal("2500"))
    assert (settings.capital, settings.trade_notional) == (Decimal("25000"), Decimal("2500"))


def test_ac92_asset_balances_independent():
    accounts = PaperAccounts(default_capital=Decimal("10000"))
    accounts.debit("ETH/USD", Decimal("1000"))
    assert accounts.balance("AAPL/USD") == Decimal("10000")


def test_ac98_market_cost_assumptions():
    settings = TradingSettings(costs={"CRYPTO": Decimal("25"), "BIST": Decimal("10")})
    assert settings.cost_for("CRYPTO") == Decimal("25") and settings.cost_for("BIST") == Decimal("10")


def test_ac121_ml_time_boundary():
    observations = [(NOW - timedelta(days=1), Decimal("1")), (NOW + timedelta(days=1), Decimal("999"))]
    assert point_in_time_value(observations, NOW) == Decimal("1")
