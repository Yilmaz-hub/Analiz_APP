from signal_engine import generate_stable_signal, _compute_bar_score, SignalStateMachine
from technical_analysis import _check_exit, run_strategy_backtest


def test_process_data_has_required_columns(processed_df):
    for c in ["RSI", "ADX", "ATR", "EMA_20", "EMA_50", "MACD", "MACD_Signal"]:
        assert c in processed_df.columns


def test_process_data_does_not_backfill_indicator_warmup():
    """ADX needs 13 bars of history and must not be filled from future bars."""
    from conftest import make_ohlcv
    from data_fetchers import process_data

    df, _ = process_data(make_ohlcv(segments=[(80, 0.001)]), "test")
    assert df["ADX"].iloc[:13].isna().all()


def test_stable_signal_is_deterministic(processed_df):
    s1 = generate_stable_signal(processed_df, "1d", include_ml=False)
    s2 = generate_stable_signal(processed_df, "1d", include_ml=False)
    assert s1.verdict == s2.verdict
    assert s1.final_score == s2.final_score
    assert s1.verdict in ("GÜÇLÜ AL", "AL", "BEKLE", "SAT", "GÜÇLÜ SAT")


def test_state_machine_reduces_whipsaw(processed_df):
    """The confirmed (stable) signal must flip direction less often than
    the raw per-bar direction — that's the entire point of the anti-
    whipsaw state machine."""
    raw_flips, stable_flips = 0, 0
    prev_raw, prev_stable = None, None
    machine = SignalStateMachine()
    for i in range(150, len(processed_df)):
        sl = processed_df.iloc[:i]
        b = _compute_bar_score(sl, "1d", include_ml=False)
        if b is None:
            continue
        state, raw = machine.update(b["score"], b["confidence"], b["rsi"], b["adx"])
        if prev_raw is not None and raw != prev_raw:
            raw_flips += 1
        if prev_stable is not None and state != prev_stable:
            stable_flips += 1
        prev_raw, prev_stable = raw, state
    assert stable_flips < raw_flips


def test_state_machine_does_not_arm_with_unknown_enabled_regime():
    machine = SignalStateMachine()
    for _ in range(2):
        machine.update(score=40, confidence=90, rsi=50, adx=30, regime=0)
    assert machine.state == 1
    assert machine.signal == 0

    machine.update(score=40, confidence=90, rsi=50, adx=30, regime=1)
    assert machine.signal == 1


def test_backtest_uses_conservative_intrabar_stop_fill():
    position = {"sl": 95.0, "tp": 105.0}
    # Both levels trade during this candle; OHLC cannot know their order.
    reason, fill = _check_exit(
        position,
        {"open": 100.0, "high": 106.0, "low": 94.0, "close": 101.0},
        state=1,
        direction=1,
    )
    assert (reason, fill) == ("SL", 95.0)


def test_short_data_guard(processed_df):
    short = generate_stable_signal(processed_df.iloc[:40], "1d")
    assert short.verdict == "BEKLE"
    assert "Yetersiz" in short.reasons[0]


def test_unclosed_candle_immunity(processed_df):
    """A price pump on the still-forming (last, unclosed) candle must not
    change the published verdict — DROP_UNCLOSED_CANDLE exists precisely
    to prevent intraday flip-flopping."""
    df_mut = processed_df.copy()
    df_mut.iloc[-1, df_mut.columns.get_loc("Close")] *= 1.10
    s1 = generate_stable_signal(processed_df, "1d", include_ml=False)
    s_mut = generate_stable_signal(df_mut, "1d", include_ml=False)
    assert s1.verdict == s_mut.verdict


def test_backtest_runs_and_is_deterministic(processed_df):
    bt1 = run_strategy_backtest(processed_df, initial_balance=10000, timeframe="1d")
    assert bt1 is not None
    assert bt1["total_trades"] > 0
    bt2 = run_strategy_backtest(processed_df, initial_balance=10000, timeframe="1d")
    assert bt2["total_return"] == bt1["total_return"]


def test_backtest_short_direction_runs(processed_df):
    """Covers the direction='short' path (mirrors the long path since the
    long/short trailing-stop and exit logic was deduplicated into shared
    helpers) — must execute cleanly and only produce known exit reasons."""
    bt = run_strategy_backtest(
        processed_df, initial_balance=10000, timeframe="1d", direction="short"
    )
    if bt is not None:
        assert bt["total_trades"] >= 0
        assert set(t["reason"] for t in bt["trades"]) <= {"SL", "TP", "Sinyal", "Final"}


def test_weekly_and_4h_backtests_do_not_crash(processed_df):
    for tf in ("1wk", "4h"):
        run_strategy_backtest(processed_df, timeframe=tf)  # None is a valid result
