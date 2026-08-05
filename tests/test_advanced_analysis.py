from advanced_analysis import (
    detect_elliott_wave,
    analyze_ichimoku,
    detect_wyckoff_phase,
    analyze_market_structure,
    calculate_advanced_score,
)


def test_elliott_wave_schema(processed_df):
    result = detect_elliott_wave(processed_df)
    assert "detected" in result
    assert result["direction"] in ("NEUTRAL", "BULLISH", "BEARISH")


def test_ichimoku_schema_and_bounds(processed_df):
    result = analyze_ichimoku(processed_df)
    assert -100 <= result["score"] <= 100
    assert result["signal"] in ("NEUTRAL", "BULLISH", "BEARISH")


def test_wyckoff_schema_and_bounds(processed_df):
    result = detect_wyckoff_phase(processed_df)
    assert -100 <= result["score"] <= 100


def test_market_structure_schema_and_bounds(processed_df):
    result = analyze_market_structure(processed_df)
    assert -100 <= result["score"] <= 100
    assert isinstance(result["bos"], bool)
    assert isinstance(result["choch"], bool)


def test_combined_advanced_score_bounds(processed_df):
    score, reasons = calculate_advanced_score(processed_df, "1d")
    assert -100 <= score <= 100
    assert isinstance(reasons, list)


def test_short_or_missing_data_returns_safe_defaults():
    assert detect_elliott_wave(None)["detected"] is False
    assert analyze_ichimoku(None)["score"] == 0
    assert detect_wyckoff_phase(None)["score"] == 0
    assert analyze_market_structure(None)["score"] == 0
