from technical_analysis import detect_advanced_patterns

import streamlit as st

import ui_components
import theme
from ui_components import render_main_chart


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


def _ascending_triangle_broken_out():
    return {
        "type": "triangle", "name": "Yükselen Üçgen ▲", "color": "green",
        "lines": [
            {"x0": 0, "y0": 100.0, "x1": 10, "y1": 100.0},
            {"x0": 0, "y0": 90.0, "x1": 10, "y1": 98.0},
        ],
        "direction": "BULLISH", "target": 110.0, "confidence": 75,
    }


def _render_kwargs(df_view, curr, f_advanced=True, show_all_pats=True):
    return dict(
        df_view=df_view, view_tf="1d", curr=curr,
        f_dates=[], f_prices=[], ai_score=0,
        show_cloud=False, show_pred=False, show_ai=False, show_all_pats=show_all_pats,
        f_wm=False, f_candle=False, f_advanced=f_advanced,
        items_raw=[], lines=[],
    )


def test_render_main_chart_returns_pattern_statuses(monkeypatch, processed_df):
    curr = 105.0  # above resistance_at_last(100.0), below target(110.0) -> broke_out
    monkeypatch.setattr(ui_components, "detect_advanced_patterns", lambda df: [_ascending_triangle_broken_out()])
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **kwargs: None)

    result = render_main_chart(**_render_kwargs(processed_df, curr))

    assert len(result) == 3
    _, _, pattern_statuses = result
    assert len(pattern_statuses) == 1
    assert "yukarı kırıldı" in pattern_statuses[0]
    assert "110" in pattern_statuses[0]


def test_render_main_chart_target_line_uses_distinct_style(monkeypatch, processed_df):
    curr = 105.0
    monkeypatch.setattr(ui_components, "detect_advanced_patterns", lambda df: [_ascending_triangle_broken_out()])
    captured = {}
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **kwargs: captured.__setitem__("fig", fig))

    render_main_chart(**_render_kwargs(processed_df, curr))

    target_shapes = [s for s in captured["fig"].layout.shapes
                      if s.line.color == theme.PALETTE["accent"] and s.line.width == 3]
    assert len(target_shapes) == 1
    assert target_shapes[0].y0 == target_shapes[0].y1 == 110.0


def test_render_main_chart_pattern_statuses_empty_when_advanced_off(monkeypatch, processed_df):
    curr = float(processed_df['Close'].iloc[-1])
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **kwargs: None)

    _, _, pattern_statuses = render_main_chart(**_render_kwargs(processed_df, curr, f_advanced=False, show_all_pats=True))
    assert pattern_statuses == []
