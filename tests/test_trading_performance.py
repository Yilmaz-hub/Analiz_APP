import statistics
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from decimal import Decimal

from config import TradingV1Config
from trade_decisions import Position, Signal
from trading_ui import PanelInput, build_decision_panel


def _compute_once():
    return build_decision_panel(PanelInput(
        signal=Signal.BUY, position=Position.flat(), data_status="GECERLI",
        fee=Decimal("0"), spread_bps=Decimal("0"), slippage_bps=Decimal("0"),
        confidence=Decimal("60"), decision_at=datetime.now(timezone.utc),
    ))


def _signal_frame():
    from data_fetchers import process_data

    count = 500
    axis = np.arange(count)
    close = 100 + axis * 0.1 + np.sin(axis / 10)
    raw = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1000 + axis,
    }, index=pd.date_range("2024-01-01", periods=count, freq="D"))
    return process_data(raw, "PERF")[0]


def test_ac36_compute_budget():
    from signal_engine import generate_stable_signal

    frame = _signal_frame()
    started = time.perf_counter()
    signal = generate_stable_signal(frame, "1d", include_ml=True, data_is_closed=True)
    first_signal_elapsed = time.perf_counter() - started
    assert signal is not None
    assert first_signal_elapsed <= TradingV1Config.FIRST_SIGNAL_COMPUTE_BUDGET_SECONDS
    for _ in range(3):
        _compute_once()
    samples = []
    for _ in range(20):
        started = time.perf_counter(); _compute_once(); samples.append(time.perf_counter() - started)
    assert statistics.median(samples) <= TradingV1Config.SCREEN_COMPUTE_BUDGET_SECONDS
    assert max(samples) <= TradingV1Config.SCREEN_COMPUTE_MAX_SECONDS


def test_ac73_total_wait_budget():
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        subprocess.run([sys.executable, "-c", "import data_fetchers"], check=True,
                       capture_output=True, timeout=TradingV1Config.COLD_START_BUDGET_SECONDS)
        samples.append(time.perf_counter() - started)
    assert max(samples) <= TradingV1Config.COLD_START_BUDGET_SECONDS


def test_500_bar_indicator_startup_avoids_pandas_ta_and_stays_in_budget():
    script = (
        "import sys,time,numpy as np,pandas as pd; "
        "from data_fetchers import process_data; "
        "n=500; c=100+np.arange(n)*.1; "
        "d=pd.DataFrame({'Open':c,'High':c+1,'Low':c-1,'Close':c,'Volume':1000},"
        "index=pd.date_range('2024-01-01',periods=n)); "
        "t=time.perf_counter(); out,_=process_data(d,'test'); elapsed=time.perf_counter()-t; "
        "assert out is not None and 'pandas_ta' not in sys.modules and elapsed < 2.0"
    )
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        timeout=TradingV1Config.COLD_START_BUDGET_SECONDS,
    )
