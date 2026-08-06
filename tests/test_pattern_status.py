from technical_analysis import get_pattern_status


def _ascending_triangle(target=110.0):
    return {
        "type": "triangle", "name": "Yükselen Üçgen ▲", "color": "green",
        "lines": [
            {"x0": 0, "y0": 100.0, "x1": 10, "y1": 100.0},   # resistance, flat at 100
            {"x0": 0, "y0": 90.0, "x1": 10, "y1": 98.0},     # support, rising toward 98
        ],
        "direction": "BULLISH", "target": target, "confidence": 75,
    }


def _descending_triangle(target=80.0):
    return {
        "type": "triangle", "name": "Düşen Üçgen ▼", "color": "red",
        "lines": [
            {"x0": 0, "y0": 105.0, "x1": 10, "y1": 98.0},    # resistance, falling toward 98
            {"x0": 0, "y0": 90.0, "x1": 10, "y1": 90.0},     # support, flat at 90
        ],
        "direction": "BEARISH", "target": target, "confidence": 75,
    }


def _symmetric_triangle():
    return {
        "type": "triangle", "name": "Simetrik Üçgen ◇", "color": "yellow",
        "lines": [
            {"x0": 0, "y0": 110.0, "x1": 10, "y1": 102.0},   # resistance, falling
            {"x0": 0, "y0": 90.0, "x1": 10, "y1": 98.0},     # support, rising
        ],
        # target=100.0 mirrors the real detector's pre-breakout placeholder
        # (== current_price at detection time) -- must never be used once broken out.
        "direction": "NEUTRAL", "target": 100.0, "confidence": 60,
    }


def _reversal(neckline=95.0, target=80.0):
    return {
        "type": "reversal", "name": "Baş-Omuz 👤", "color": "red",
        "x0": 0, "y0": 110.0, "head_x": 5, "head_y": 120.0, "x1": 10, "y1": 110.0,
        "neckline": neckline, "direction": "BEARISH", "target": target, "confidence": 90,
    }


def _continuation(box_top=105.0, target=115.0):
    return {
        "type": "continuation", "name": "Boğa Bayrağı 🚩", "color": "lime",
        "x0": 0, "y0": 95.0, "x1": 10, "y1": box_top,
        "direction": "BULLISH", "target": target, "confidence": 70,
    }


def _harmonic():
    return {
        "type": "harmonic", "name": "ABCD Boğa 🦬", "color": "cyan",
        "points": [{"idx": 0, "price": 100.0, "type": "low"}],
        "direction": "BULLISH", "target": 90.0, "confidence": 80,
    }


def test_harmonic_pattern_returns_none():
    assert get_pattern_status(_harmonic(), current_price=95.0) is None


def test_ascending_triangle_forming_inside_boundary():
    status = get_pattern_status(_ascending_triangle(), current_price=99.0)
    assert status["stage"] == "forming"


def test_ascending_triangle_breaks_out_above_resistance():
    status = get_pattern_status(_ascending_triangle(target=110.0), current_price=105.0)
    assert status["stage"] == "broke_out"
    assert status["direction"] == "BULLISH"
    assert "yukarı kırıldı" in status["message"]
    assert "110" in status["message"]


def test_ascending_triangle_reaches_target():
    status = get_pattern_status(_ascending_triangle(target=110.0), current_price=112.0)
    assert status["stage"] == "target_reached"


def test_descending_triangle_forming_inside_boundary():
    status = get_pattern_status(_descending_triangle(), current_price=94.0)
    assert status["stage"] == "forming"


def test_descending_triangle_breaks_out_below_support():
    status = get_pattern_status(_descending_triangle(target=80.0), current_price=85.0)
    assert status["stage"] == "broke_out"
    assert status["direction"] == "BEARISH"
    assert "aşağı kırıldı" in status["message"]


def test_descending_triangle_reaches_target():
    status = get_pattern_status(_descending_triangle(target=80.0), current_price=79.0)
    assert status["stage"] == "target_reached"


def test_symmetric_triangle_forming_inside_boundary():
    status = get_pattern_status(_symmetric_triangle(), current_price=100.0)
    assert status["stage"] == "forming"


def test_symmetric_triangle_bullish_breakout_recomputes_real_target():
    """Regression test: the stored target (100.0, == current_price at
    detection time) must NOT be used once broken out -- a real
    measured-move target is computed instead."""
    status = get_pattern_status(_symmetric_triangle(), current_price=105.0)
    assert status["stage"] == "broke_out"
    assert status["direction"] == "BULLISH"
    assert status["target"] == 122.0  # resistance_at_last(102) + height(110-90=20)


def test_symmetric_triangle_bearish_breakout_recomputes_real_target():
    status = get_pattern_status(_symmetric_triangle(), current_price=95.0)
    assert status["stage"] == "broke_out"
    assert status["direction"] == "BEARISH"
    assert status["target"] == 78.0  # support_at_last(98) - height(20)


def test_symmetric_triangle_target_is_fixed_not_chasing_price():
    """Regression test for a real bug caught during planning: a target
    defined as current_price + height can never be reached, since it
    always sits ahead of whatever price triggered the check. The target
    must be anchored to the fixed breakout boundary instead, so a later,
    higher price can actually reach it."""
    broke_out = get_pattern_status(_symmetric_triangle(), current_price=105.0)
    reached = get_pattern_status(_symmetric_triangle(), current_price=125.0)
    assert broke_out["target"] == reached["target"] == 122.0
    assert reached["stage"] == "target_reached"


def test_reversal_forming_above_neckline():
    status = get_pattern_status(_reversal(neckline=95.0), current_price=97.0)
    assert status["stage"] == "forming"


def test_reversal_breaks_out_below_neckline():
    status = get_pattern_status(_reversal(neckline=95.0, target=80.0), current_price=90.0)
    assert status["stage"] == "broke_out"
    assert status["direction"] == "BEARISH"


def test_reversal_reaches_target():
    status = get_pattern_status(_reversal(neckline=95.0, target=80.0), current_price=79.0)
    assert status["stage"] == "target_reached"


def test_continuation_forming_inside_box():
    status = get_pattern_status(_continuation(box_top=105.0), current_price=104.0)
    assert status["stage"] == "forming"


def test_continuation_breaks_out_above_box():
    status = get_pattern_status(_continuation(box_top=105.0, target=115.0), current_price=108.0)
    assert status["stage"] == "broke_out"
    assert status["direction"] == "BULLISH"


def test_continuation_reaches_target():
    status = get_pattern_status(_continuation(box_top=105.0, target=115.0), current_price=116.0)
    assert status["stage"] == "target_reached"


def test_target_reached_takes_priority_over_broke_out():
    """When price has moved straight through the boundary and past the
    target (e.g. checked only after a big gap-up bar), the reported stage
    must be target_reached, not broke_out."""
    status = get_pattern_status(_ascending_triangle(target=110.0), current_price=150.0)
    assert status["stage"] == "target_reached"
