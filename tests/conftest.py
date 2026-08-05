import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(seed=42, segments=None):
    """Synthetic OHLCV with configurable trend regimes (no network needed).
    segments: list of (bar_count, daily_drift) tuples."""
    if segments is None:
        segments = [(120, 0.003), (80, -0.004), (100, 0.0), (120, 0.005), (80, -0.002)]
    rng = np.random.default_rng(seed)
    regime = np.concatenate([np.full(count, drift) for count, drift in segments])
    n = len(regime)
    rets = regime + rng.normal(0, 0.02, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(1e6, 5e6, n) * (1 + np.abs(regime) * 100)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


@pytest.fixture(scope="session")
def processed_df():
    """500-bar synthetic OHLCV (mixed trend/chop regimes) through the real
    indicator pipeline — the same shape used across most signal/backtest
    tests."""
    from data_fetchers import process_data

    df = make_ohlcv()
    df, _ = process_data(df, "test")
    return df


@pytest.fixture(scope="session")
def trending_df():
    """A cleaner, persistently-uptrending series for tests that need an
    unambiguous trend (regime score, trend strength direction, etc.)."""
    from data_fetchers import process_data

    df = make_ohlcv(seed=7, segments=[(300, 0.004)])
    df, _ = process_data(df, "test")
    return df
