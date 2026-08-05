from technical_analysis import detect_advanced_patterns


def test_triangle_patterns_carry_both_boundary_lines(trending_df):
    """Regression test for the chart bug: a triangle has two boundary
    lines (resistance from peaks, support from troughs); the pattern dict
    must carry both so the renderer can draw the actual shape instead of
    a single edge."""
    patterns = detect_advanced_patterns(trending_df)
    for p in patterns:
        if p["type"] == "triangle":
            assert "lines" in p
            assert len(p["lines"]) == 2
            for line in p["lines"]:
                assert {"x0", "y0", "x1", "y1"} <= set(line)


def test_reversal_patterns_carry_head_coordinates(processed_df):
    """Regression test: a head-and-shoulders pattern must carry the head's
    coordinates so the renderer can draw left-shoulder -> head ->
    right-shoulder, not just the neckline."""
    patterns = detect_advanced_patterns(processed_df)
    for p in patterns:
        if p["type"] == "reversal":
            assert "head_x" in p and "head_y" in p


def test_continuation_patterns_carry_box_coordinates(processed_df):
    patterns = detect_advanced_patterns(processed_df)
    for p in patterns:
        if p["type"] == "continuation":
            assert {"x0", "y0", "x1", "y1"} <= set(p)
