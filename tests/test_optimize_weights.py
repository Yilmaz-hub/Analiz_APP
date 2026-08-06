import pandas as pd

from config import DecisionEngineConfig
from optimize_weights import _weights_from_vector, _pool_score, _split_train_test, MIN_TRADES, REJECT_SCORE


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


def test_pool_score_rejects_thin_pooled_trade_count():
    # 2 trades total across both assets, below MIN_TRADES=5.
    bt_results = [
        {"trades": [{"pnl": 10, "pnl_pct": 5.0}]},
        {"trades": [{"pnl": -5, "pnl_pct": -2.0}]},
    ]
    assert _pool_score(bt_results) == REJECT_SCORE


def test_pool_score_rejects_all_none_results():
    assert _pool_score([None, None]) == REJECT_SCORE


def test_pool_score_ignores_none_entries_in_the_list():
    """A None result (an asset with zero trades) must not change the pooled
    score versus simply omitting that asset."""
    bt_a = {"trades": [{"pnl": 10, "pnl_pct": 5.0} for _ in range(5)]}
    assert _pool_score([bt_a, None]) == _pool_score([bt_a])


def test_pool_score_pools_trades_across_assets():
    """Regression test for the real bug this replaces: 3 winning trades
    from one asset and 3 losing trades from another individually clear
    neither the old per-asset MIN_TRADES bar nor look great alone, but
    pooled (6 trades) they clear MIN_TRADES and produce a real score."""
    bt_a = {"trades": [{"pnl": 100, "pnl_pct": 10.0} for _ in range(3)]}
    bt_b = {"trades": [{"pnl": -50, "pnl_pct": -5.0} for _ in range(3)]}
    score = _pool_score([bt_a, bt_b])
    # win_rate=50%, avg_return_pct=(10*3-5*3)/6=2.5, profit_factor=300/150=2.0
    # score = 2.5 * (0.5 + 0.3*2/5 + 0.2*0.5) = 2.5 * 0.72 = 1.8
    assert abs(score - 1.8) < 1e-9


def test_pool_score_all_wins_gets_high_profit_factor_not_zero():
    """An all-winning pooled sample (no losses) must not be penalized with
    profit_factor=0 -- it should be treated as very favorable."""
    bt_a = {"trades": [{"pnl": 10, "pnl_pct": 5.0} for _ in range(5)]}
    score = _pool_score([bt_a])
    # win_rate=100, avg_return_pct=5.0, profit_factor capped at 5.0
    # score = 5.0 * (0.5 + 0.3*5/5 + 0.2*1.0) = 5.0 * 1.0 = 5.0
    assert abs(score - 5.0) < 1e-9


def test_split_train_test_is_chronological_and_70_30():
    df = pd.DataFrame({"Close": range(100)})
    train, test = _split_train_test(df)
    assert len(train) == 70
    assert len(test) == 30
    assert train.iloc[-1]["Close"] < test.iloc[0]["Close"]  # train is strictly earlier
    assert list(train["Close"]) + list(test["Close"]) == list(df["Close"])  # no gaps/overlap
