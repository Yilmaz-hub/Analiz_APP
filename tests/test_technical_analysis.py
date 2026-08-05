from technical_analysis import (
    calculate_sr_advanced,
    calculate_regime_score,
    calculate_trend_strength,
    detect_patterns,
    detect_advanced_patterns,
    calculate_trade_setup,
    detect_rsi_divergence,
)


def test_regime_score_bounds_and_schema(processed_df):
    result = calculate_regime_score(processed_df)
    assert result is not None
    assert 0 <= result["score"] <= 100
    assert result["label"]
    assert set(result["components"]) == {"persistence", "slope", "adx", "alignment"}


def test_regime_score_high_for_clean_uptrend(trending_df):
    result = calculate_regime_score(trending_df)
    assert result is not None
    assert result["score"] >= 50, "a clean, persistent uptrend should score as trending"


def test_regime_score_none_on_insufficient_data():
    import pandas as pd

    assert calculate_regime_score(pd.DataFrame({"Close": [1, 2, 3]})) is None
    assert calculate_regime_score(None) is None


def test_trend_strength_bounds(processed_df):
    score = calculate_trend_strength(processed_df)
    assert -100 <= score <= 100


def test_trend_strength_positive_for_uptrend(trending_df):
    assert calculate_trend_strength(trending_df) > 0


def test_detect_patterns_returns_well_formed_list(processed_df):
    patterns = detect_patterns(processed_df)
    assert isinstance(patterns, list)
    for p in patterns:
        assert "type" in p and "name" in p


def test_detect_advanced_patterns_returns_list(processed_df):
    patterns = detect_advanced_patterns(processed_df)
    assert isinstance(patterns, list)


def test_trade_setup_long_and_short(processed_df):
    long_setup = calculate_trade_setup(processed_df, "AL")
    assert long_setup["direction"] == "LONG"
    assert long_setup["sl"] < long_setup["entry"] < long_setup["tp"]

    short_setup = calculate_trade_setup(processed_df, "SAT")
    assert short_setup["direction"] == "SHORT"
    assert short_setup["tp"] < short_setup["entry"] < short_setup["sl"]

    assert calculate_trade_setup(processed_df, "BEKLE") is None


def test_rsi_divergence_returns_valid_value(processed_df):
    result = detect_rsi_divergence(processed_df)
    assert result in (None, "BULLISH", "BEARISH")


def test_sr_advanced_returns_sorted_unique_levels(processed_df):
    supports, resistances = calculate_sr_advanced(processed_df, "1d")
    assert supports == sorted(supports)
    assert resistances == sorted(resistances)
    assert len(supports) == len(set(supports))
    assert len(resistances) == len(set(resistances))
