import pandas as pd

from ml_models import calculate_ml_direction_signal, calculate_smart_prediction_FIXED


def test_ml_direction_signal_schema(processed_df):
    result = calculate_ml_direction_signal(processed_df)
    assert result is not None
    assert result["direction"] in ("BULLISH", "BEARISH", "NEUTRAL")
    assert 0 <= result["confidence"] <= 100


def test_ai_forecast_anchored_to_true_latest_bar(processed_df):
    """Regression test for the off-by-one bug: dropna() run after attaching
    a shift(-1) target used to silently discard the newest bar, so the
    forecast started two bars after the input's true last index instead of
    one. Pin the correct alignment so this can't regress silently."""
    df = processed_df.iloc[:300]
    future_dates, predictions, accuracy = calculate_smart_prediction_FIXED(df, periods=5)
    assert future_dates, "expected a non-empty forecast"
    one_bar = df.index[-1] - df.index[-2]
    assert future_dates[0] == df.index[-1] + one_bar
    assert len(predictions) == len(future_dates) == 5
    assert 0 <= accuracy <= 100


def test_ml_direction_signal_tracks_the_newest_available_bar(processed_df):
    """Same regression family as above, for the classifier: predicting on
    df[:300] vs df[:301] must be based on genuinely different (newer)
    feature rows, not the same stale one twice."""
    df_a = processed_df.iloc[:300]
    df_b = processed_df.iloc[:301]
    res_a = calculate_ml_direction_signal(df_a)
    res_b = calculate_ml_direction_signal(df_b)
    assert res_a is not None and res_b is not None
    # predicted_change_pct is derived from a regression fit on the labeled
    # history up to (but excluding) the newest row, then evaluated on that
    # newest row's own features — those features differ between the two
    # slices, so an exact-equal result would indicate the newest row isn't
    # actually being used (the bug's signature).
    assert res_a["predicted_change_pct"] != res_b["predicted_change_pct"]


def test_insufficient_data_returns_none_or_empty():
    short_df = pd.DataFrame(
        {"Open": [1] * 10, "High": [1] * 10, "Low": [1] * 10, "Close": [1] * 10, "Volume": [1] * 10}
    )
    assert calculate_ml_direction_signal(short_df) is None
    fd, pr, acc = calculate_smart_prediction_FIXED(short_df)
    assert fd == [] and pr == [] and acc == 0
