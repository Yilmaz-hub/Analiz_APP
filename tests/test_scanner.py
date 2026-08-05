from scanner import vol_sizing_factor
from config import SizingConfig


def test_vol_sizing_neutral_when_missing_data():
    assert vol_sizing_factor(0, 0) == 1.0
    assert vol_sizing_factor(None, 0.1) == 1.0
    assert vol_sizing_factor(0.1, None) == 1.0


def test_vol_sizing_bounds():
    assert vol_sizing_factor(0.001, 0.1) == SizingConfig.ADJ_MAX
    assert vol_sizing_factor(10, 0.1) == SizingConfig.ADJ_MIN


def test_vol_sizing_monotonic_in_volatility():
    calm = vol_sizing_factor(0.01, 0.05)
    wild = vol_sizing_factor(0.2, 0.05)
    assert calm > wild
