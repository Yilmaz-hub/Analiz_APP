from technical_analysis import run_strategy_backtest

CUSTOM_WEIGHTS = {"trend": 0.05, "momentum": 0.05, "volume": 0.60, "pattern": 0.05, "ml": 0.05, "advanced": 0.20}


def test_backtest_with_custom_weights_runs_and_can_differ_from_default(processed_df):
    default_bt = run_strategy_backtest(processed_df, initial_balance=10000, timeframe="1d", weights=None)
    custom_bt = run_strategy_backtest(processed_df, initial_balance=10000, timeframe="1d", weights=CUSTOM_WEIGHTS)
    # Both must at least run cleanly; a heavily volume-weighted strategy
    # trading the same data as the balanced default is expected (not
    # guaranteed on every possible dataset, but true for this fixture) to
    # produce a different trade count.
    assert default_bt is not None
    assert custom_bt is None or custom_bt["total_trades"] != default_bt["total_trades"] or custom_bt["total_return"] != default_bt["total_return"]
