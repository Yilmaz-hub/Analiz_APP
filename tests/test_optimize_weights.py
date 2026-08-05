import pandas as pd

from config import DecisionEngineConfig
from optimize_weights import _weights_from_vector, _asset_score, _split_train_test, MIN_TRADES, REJECT_SCORE


def test_weights_from_vector_fills_ml_and_derives_advanced():
    weights = _weights_from_vector([0.30, 0.15, 0.25, 0.05])
    assert weights is not None
    assert weights["trend"] == 0.30
    assert weights["momentum"] == 0.15
    assert weights["volume"] == 0.25
    assert weights["pattern"] == 0.05
    assert weights["ml"] == DecisionEngineConfig.ML_WEIGHT
    expected_advanced = 1.0 - DecisionEngineConfig.ML_WEIGHT - (0.30 + 0.15 + 0.25 + 0.05)
    assert abs(weights["advanced"] - expected_advanced) < 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_from_vector_rejects_infeasible_combination():
    # trend+momentum+volume+pattern alone already exceed what's left after
    # ML_WEIGHT is reserved -> advanced would be negative -> infeasible.
    assert _weights_from_vector([0.9, 0.9, 0.9, 0.9]) is None


def test_asset_score_rejects_thin_trade_count():
    bt = {"total_trades": MIN_TRADES - 1, "total_return": 50.0, "profit_factor": 3.0, "win_rate": 90.0}
    assert _asset_score(bt) == REJECT_SCORE


def test_asset_score_rejects_none_result():
    assert _asset_score(None) == REJECT_SCORE


def test_asset_score_positive_for_good_backtest():
    bt = {"total_trades": MIN_TRADES, "total_return": 20.0, "profit_factor": 2.0, "win_rate": 60.0}
    score = _asset_score(bt)
    assert score > 0
    # formula: 20.0 * (0.5 + 0.3*min(2,5)/5 + 0.2*60/100) = 20.0 * (0.5+0.12+0.12) = 20.0*0.74
    assert abs(score - 14.8) < 1e-9


def test_split_train_test_is_chronological_and_70_30():
    df = pd.DataFrame({"Close": range(100)})
    train, test = _split_train_test(df)
    assert len(train) == 70
    assert len(test) == 30
    assert train.iloc[-1]["Close"] < test.iloc[0]["Close"]  # train is strictly earlier
    assert list(train["Close"]) + list(test["Close"]) == list(df["Close"])  # no gaps/overlap
