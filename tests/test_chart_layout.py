import pandas as pd
import streamlit as st

import ui_components
from ui_components import render_main_chart


def _base_kwargs(df_view, **overrides):
    curr = df_view['Close'].iloc[-1]
    kwargs = dict(
        df_view=df_view, view_tf="1d", curr=curr,
        f_dates=[], f_prices=[], ai_score=0,
        show_cloud=False, show_pred=False, show_ai=False, show_all_pats=False,
        f_wm=False, f_candle=False, f_advanced=False,
        items_raw=[], lines=[],
    )
    kwargs.update(overrides)
    return kwargs


def test_chart_config_is_responsive_for_mobile(monkeypatch, processed_df):
    """Regression test: without config['responsive']=True, Plotly renders at
    a stale width on mobile and the chart gets cropped -- only part of the
    candles are visible without horizontal scrolling."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["config"] = kwargs.get("config")

    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    render_main_chart(**_base_kwargs(processed_df))

    assert captured["config"]["responsive"] is True


def test_chart_uses_thin_crosshair_not_unified_hover(monkeypatch, processed_df):
    """Regression test: hovermode='x unified' drew a thick vertical line
    across the whole chart, obscuring candles. Thin axis spikes +
    hovermode='x' give a TradingView-style crosshair with axis-edge
    price/date labels instead."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    render_main_chart(**_base_kwargs(processed_df))

    layout = captured["fig"].layout
    assert layout.hovermode == "x"
    assert layout.xaxis.showspikes is True
    assert layout.yaxis.showspikes is True
    assert layout.xaxis.spikethickness == 1
    assert layout.yaxis.spikethickness == 1
    assert layout.xaxis.spikemode == "across"
    assert layout.yaxis.spikemode == "across"


def test_crosshair_is_faint_and_dashed_like_tradingview(monkeypatch, processed_df):
    """Regression test: the crosshair was a solid line at 0.35 opacity --
    dashed and fainter reads closer to TradingView's default crosshair."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    render_main_chart(**_base_kwargs(processed_df))

    layout = captured["fig"].layout
    assert layout.xaxis.spikedash == "dash"
    assert layout.yaxis.spikedash == "dash"
    assert "0.22" in layout.xaxis.spikecolor
    assert "0.22" in layout.yaxis.spikecolor


def _prediction_kwargs(df_view, n_days=15):
    last_date = df_view.index[-1]
    delta = df_view.index[-1] - df_view.index[-2]
    f_dates = [last_date + delta * step for step in range(1, n_days + 1)]
    # Deliberately far outside the recent High/Low range, and the point of
    # the regression test: a real forecast can legitimately do this.
    far_price = float(df_view['High'].tail(80).max()) * 1.5
    f_prices = [far_price] * n_days
    return _base_kwargs(df_view, show_pred=True, f_dates=f_dates, f_prices=f_prices, ai_score=50)


def test_prediction_line_extends_visible_x_range(monkeypatch, processed_df):
    """Regression test: the default zoom window only extended 5 bars into
    the future (gap_multiplier), but the AI forecast runs 15 bars ahead --
    most of the prediction line fell outside the visible x-range entirely,
    so it could never be seen or hovered no matter where the cursor was."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    kwargs = _prediction_kwargs(processed_df)
    render_main_chart(**kwargs)

    x_range = captured["fig"].layout.xaxis.range
    assert pd.Timestamp(x_range[1]) >= kwargs["f_dates"][-1]


def test_prediction_line_extends_visible_y_range(monkeypatch, processed_df):
    """Regression test: the y-axis range was computed purely from recent
    candle High/Low, ignoring the predicted prices -- a forecast outside
    that historical range got clipped off-screen (invisible, un-hoverable)
    even when its x-position was within the visible window."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    kwargs = _prediction_kwargs(processed_df)
    render_main_chart(**kwargs)

    y_range = captured["fig"].layout.yaxis.range
    predicted_price = kwargs["f_prices"][0]
    assert y_range[1] >= predicted_price


def test_prediction_points_expose_future_price_on_hover(monkeypatch):
    """Forecast dates need real hover targets and an explicit price tooltip;
    a line alone can be difficult to hit in the empty future chart area."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["fig"] = fig

    index = pd.date_range("2026-01-01", periods=3, freq="D")
    df_view = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [101.0, 102.0, 103.0],
    }, index=index)
    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    monkeypatch.setattr(ui_components, "calculate_sr_advanced", lambda df, tf: ([], []))
    kwargs = _prediction_kwargs(df_view)
    render_main_chart(**kwargs)

    prediction = next(trace for trace in captured["fig"].data
                      if trace.name.startswith("AI Tahmini"))
    assert prediction.mode == "lines+markers"
    assert list(prediction.marker.size)[1:] == [5] * len(kwargs["f_dates"])
    assert "Tahmini Fiyat" in prediction.hovertemplate
    assert "%{y:,.2f}" in prediction.hovertemplate
