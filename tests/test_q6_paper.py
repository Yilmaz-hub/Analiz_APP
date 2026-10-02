"""Q6: sanal takip, hesaplanamayan bileşende yeni AL/SAT yazmaz."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import paper_trading
from config import FileConfig
from conftest import make_ohlcv


def _daily_frame_ending_yesterday():
    from data_fetchers import process_data

    raw = make_ohlcv(seed=11, segments=[(250, 0.004)])
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    raw["Open"] = raw["Open"].clip(lower=raw["Low"], upper=raw["High"])  # geçerli OHLC
    raw.index = pd.date_range(end=pd.Timestamp(yesterday), periods=len(raw), freq="D", tz="UTC")
    frame, _ = process_data(raw, "test")
    return frame


@pytest.fixture
def paper_env(tmp_path, monkeypatch):
    monkeypatch.setattr(FileConfig, "PAPER_FILE", str(tmp_path / "paper.json"))
    import data_fetchers
    monkeypatch.setattr(data_fetchers, "get_market_data",
                        lambda *_a, **_k: (_daily_frame_ending_yesterday(), "TEST"))
    return paper_trading


def test_q6_paper_does_not_record_buy_or_sell_when_component_unavailable(paper_env, monkeypatch):
    import signal_engine
    from signal_engine import CompositeSignal

    def fake_signal(df, *_a, **_k):
        return CompositeSignal(timeframe="1d", verdict="AL", unavailable_components=("ml",))

    monkeypatch.setattr(signal_engine, "generate_stable_signal", fake_signal)
    result = paper_env.run_paper_update({"BTC": "BTC-USD"})
    state = paper_env._load_state()
    assert result["new_rows"] == 1 and not result["errors"]
    assert state["journal"][0]["verdict"] == "BILESEN_YOK"
    assert state["journal"][0]["unavailable_components"] == ["ml"]
    assert state["assets"]["BTC"]["pending"]["verdict"] == "BILESEN_YOK"


def test_q6_paper_reports_missing_components_by_name(paper_env, monkeypatch):
    import data_fetchers

    def no_rsi(*_a, **_k):
        return _daily_frame_ending_yesterday().drop(columns=["RSI"]), "TEST"

    monkeypatch.setattr(data_fetchers, "get_market_data", no_rsi)
    result = paper_env.run_paper_update({"BTC": "BTC-USD"})
    assert result["new_rows"] == 0
    assert result["errors"] == ["BTC: Karar bileşeni hazır değil: momentum"]
