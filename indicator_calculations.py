"""Pandas-native indicators used by the decision and ML paths.

They avoid the pandas_ta/numba import cost while preserving standard Wilder
and exponential definitions.  These values are analytical intermediates;
financial records continue to use Decimal.
"""
import numpy as np
import pandas as pd


def _rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_core_indicators(frame, config):
    data = frame.copy()
    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = _rma(gain, config.RSI_LENGTH)
    average_loss = _rma(loss, config.RSI_LENGTH)
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    data["RSI"] = (100 - 100 / (1 + relative_strength)).where(average_loss.ne(0), 100)

    data["EMA_20"] = close.ewm(span=config.EMA_SHORT, adjust=False, min_periods=config.EMA_SHORT).mean()
    data["EMA_50"] = close.ewm(span=config.EMA_LONG, adjust=False, min_periods=config.EMA_LONG).mean()
    fast = close.ewm(span=config.MACD_FAST, adjust=False, min_periods=config.MACD_FAST).mean()
    slow = close.ewm(span=config.MACD_SLOW, adjust=False, min_periods=config.MACD_SLOW).mean()
    data["MACD"] = fast - slow
    data["MACD_Signal"] = data["MACD"].ewm(
        span=config.MACD_SIGNAL, adjust=False, min_periods=config.MACD_SIGNAL
    ).mean()

    middle = close.rolling(config.BOLLINGER_LENGTH, min_periods=config.BOLLINGER_LENGTH).mean()
    deviation = close.rolling(config.BOLLINGER_LENGTH, min_periods=config.BOLLINGER_LENGTH).std(ddof=0)
    data["BB_Lower"] = middle - config.BOLLINGER_STD * deviation
    data["BB_Upper"] = middle + config.BOLLINGER_STD * deviation

    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - previous_close).abs(), (low - previous_close).abs()
    ], axis=1).max(axis=1)
    data["ATR"] = _rma(true_range, config.ATR_LENGTH)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _rma(true_range, 14)
    plus_di = 100 * _rma(plus_dm, 14) / atr
    minus_di = 100 * _rma(minus_dm, 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data["ADX"] = _rma(dx, 14)
    return data


def add_ml_features(frame, config):
    data = add_core_indicators(frame, config)
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    typical_mean = typical.rolling(config.CCI_LENGTH, min_periods=config.CCI_LENGTH).mean()
    mean_deviation = typical.rolling(config.CCI_LENGTH, min_periods=config.CCI_LENGTH).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))), raw=True
    )
    data["CCI"] = (typical - typical_mean) / (0.015 * mean_deviation.replace(0, np.nan))

    rsi_low = data["RSI"].rolling(14, min_periods=14).min()
    rsi_high = data["RSI"].rolling(14, min_periods=14).max()
    stochastic = 100 * (data["RSI"] - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    data["StochRSI_K"] = stochastic.rolling(3, min_periods=3).mean()

    volume = pd.to_numeric(data.get("Volume", 0), errors="coerce").fillna(0)
    data["OBV"] = (np.sign(data["Close"].diff()).fillna(0) * volume).cumsum()
    data["Volume_SMA"] = volume.rolling(20, min_periods=20).mean()
    data["Volume_Ratio"] = volume / data["Volume_SMA"].replace(0, np.nan)
    data["Price_SMA50"] = data["Close"].rolling(50, min_periods=50).mean()
    data["Price_Distance"] = (data["Close"] - data["Price_SMA50"]) / data["Price_SMA50"] * 100
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["Close"] * 100
    data["Return"] = data["Close"].pct_change()
    data["Return_5"] = data["Close"].pct_change(5)
    data["Volatility"] = data["Return"].rolling(20, min_periods=20).std()
    return data
