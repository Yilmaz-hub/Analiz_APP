from portfolio import validate_portfolio_risk
from config import RiskConfig


def test_position_within_limits_is_accepted():
    ok, msg = validate_portfolio_risk(100, 1000, [])
    assert ok is True


def test_position_exceeding_max_position_size_is_rejected():
    too_big = 1000 * RiskConfig.MAX_POSITION_SIZE + 1
    ok, msg = validate_portfolio_risk(too_big, 1000, [])
    assert ok is False
    assert str(int(RiskConfig.MAX_POSITION_SIZE * 100)) in msg


def test_total_exposure_exceeding_limit_is_rejected():
    # total_equity = balance(250) + active positions(750) = 1000.
    # new_investment(100) alone is well under the single-position cap
    # (MAX_POSITION_SIZE=0.4 -> 400), so this isolates the exposure check:
    # 750 + 100 = 850 > MAX_TOTAL_EXPOSURE(0.8) * 1000 = 800.
    existing = [
        {"Yatırım": 375, "Status": "ACTIVE"},
        {"Yatırım": 375, "Status": "ACTIVE"},
    ]
    ok, msg = validate_portfolio_risk(100, 250, existing)
    assert ok is False
    assert str(int(RiskConfig.MAX_TOTAL_EXPOSURE * 100)) in msg


def test_pending_positions_excluded_from_exposure():
    pending = [{"Yatırım": 900, "Status": "PENDING"}]
    ok, msg = validate_portfolio_risk(100, 1000, pending)
    assert ok is True
