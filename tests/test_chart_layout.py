import streamlit as st

from ui_components import render_main_chart


def _base_kwargs(df_view):
    curr = df_view['Close'].iloc[-1]
    return dict(
        df_view=df_view, view_tf="1d", curr=curr,
        f_dates=[], f_prices=[], ai_score=0,
        show_cloud=False, show_pred=False, show_ai=False, show_all_pats=False,
        f_wm=False, f_candle=False, f_advanced=False,
        items_raw=[], lines=[],
    )


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
