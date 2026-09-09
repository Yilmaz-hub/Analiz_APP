import pandas as pd
import pytest
import streamlit as st

import ui_components
from ui_components import (
    render_main_chart,
    chart_height,
    resolve_zoom_count,
    is_mobile_mode,
    is_chart_renderable,
    is_empty_data_reason,
)
from config import UIConfig


def _capture_fig(monkeypatch, **overrides):
    """Render once and return the captured Plotly figure."""
    captured = {}
    monkeypatch.setattr(st, "plotly_chart",
                        lambda fig, **kwargs: captured.setdefault("fig", fig))
    render_main_chart(**overrides)
    return captured["fig"]


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


def test_chart_uses_thin_crosshair(monkeypatch, processed_df):
    """Regression test: thin axis spikes give a TradingView-style crosshair.
    hovermode='x unified' matches tooltips by date, so OHLC info always
    belongs to the hovered candle -- 'y unified' (a prior regression) matched
    by nearest price instead and could show a different candle's date than
    the one actually under the cursor."""
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(st, "plotly_chart", fake_plotly_chart)
    render_main_chart(**_base_kwargs(processed_df))

    layout = captured["fig"].layout
    assert layout.hovermode == "x unified"
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


def test_no_transparent_heatmap_hover_layer(monkeypatch):
    """Regression test: a transparent heatmap used to supply the cursor price.
    Verified in-browser that under unified hover it does NOT report the row
    under the cursor -- it showed a price hundreds of dollars off. The axis
    badge (theme.crosshair_axis_badges) reads the true cursor position from
    the axis instead, so the layer is gone rather than quietly lying."""
    captured = {}
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    df_view = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0], "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0], "Close": [101.0, 102.0, 103.0],
    }, index=index)
    monkeypatch.setattr(st, "plotly_chart",
                        lambda fig, **kwargs: captured.setdefault("fig", fig))
    monkeypatch.setattr(ui_components, "calculate_sr_advanced", lambda df, tf: ([], []))

    render_main_chart(**_prediction_kwargs(df_view))

    assert not any(trace.type == "heatmap" for trace in captured["fig"].data)
    assert captured["fig"].layout.hovermode == "x unified"


def test_axis_badges_rendered_with_timeframe_precision(monkeypatch, processed_df):
    """The TradingView-style axis badges are the only cursor price/date
    readout, so they must actually be emitted -- and intraday charts need the
    clock time, since a bare day cannot identify a 4-hourly bar."""
    calls = []
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **kwargs: None)
    monkeypatch.setattr(ui_components.theme, "crosshair_axis_badges",
                        lambda **kwargs: calls.append(kwargs))

    render_main_chart(**_base_kwargs(processed_df, view_tf="4h"))
    assert calls == [{"intraday": True}]

    calls.clear()
    render_main_chart(**_base_kwargs(processed_df, view_tf="1d"))
    assert calls == [{"intraday": False}]


def test_candlestick_defers_hover_to_ohlc_proxy(monkeypatch, processed_df):
    """Regression test: Plotly's candlestick trace mis-identifies the hovered
    point under unified hover (plotly.js#2095), so Open/High/Low/Close stayed
    frozen on one candle while the date and EMA values updated normally. The
    candle must not hover at all; an invisible Scatter carries OHLC instead."""
    captured = {}
    monkeypatch.setattr(st, "plotly_chart",
                        lambda fig, **kwargs: captured.setdefault("fig", fig))
    render_main_chart(**_base_kwargs(processed_df))

    candle = next(t for t in captured["fig"].data if t.type == "candlestick")
    assert candle.hoverinfo == "skip"

    proxy = next(t for t in captured["fig"].data if t.name == "OHLC")
    assert proxy.customdata.shape == (len(processed_df), 4)
    assert list(proxy.customdata[0]) == [
        processed_df["Open"].iloc[0], processed_df["High"].iloc[0],
        processed_df["Low"].iloc[0], processed_df["Close"].iloc[0],
    ]
    for field in ("Açılış", "Yüksek", "Düşük", "Kapanış"):
        assert field in proxy.hovertemplate
    assert proxy.showlegend is False


def test_stale_candle_does_not_follow_cursor_into_future(monkeypatch, processed_df):
    """Regression test: hoverdistance=-1 (infinite) let the last candle's OHLC
    trail the cursor arbitrarily far into empty future dates, so that area
    showed stale values rather than the hovered date's own cursor price."""
    captured = {}
    monkeypatch.setattr(st, "plotly_chart",
                        lambda fig, **kwargs: captured.setdefault("fig", fig))
    render_main_chart(**_base_kwargs(processed_df))

    hoverdistance = captured["fig"].layout.hoverdistance
    assert hoverdistance is not None and hoverdistance > 0


# ---------------------------------------------------------------------------
# Spec 0001 — Mobil Görünüm Modu. Criterion <-> test mapping below.
# ---------------------------------------------------------------------------

def _all_layers_kwargs(df_view, mobile, f_advanced=True):
    """Every layer switched on, so feature-parity counts are meaningful.
    f_advanced defaults True: advanced patterns are the biggest source of
    shapes/annotations, so parity must be checked with them on (QA BULGU-5)."""
    curr = df_view['Close'].iloc[-1]
    last_date = df_view.index[-1]
    delta = df_view.index[-1] - df_view.index[-2]
    f_dates = [last_date + delta * step for step in range(1, 16)]
    f_prices = [float(df_view['Close'].iloc[-1])] * 15
    return _base_kwargs(
        df_view, curr=curr, mobile=mobile,
        show_cloud=True, show_pred=True, show_ai=True, show_all_pats=True,
        f_wm=True, f_candle=True, f_advanced=f_advanced,
        f_dates=f_dates, f_prices=f_prices, ai_score=55,
        lines=[{'x0': df_view.index[0], 'y0': curr, 'x1': df_view.index[-1],
                'y1': curr, 'color': 'red'}],
    )


# C2 — Varsayılan mod = Masaüstü (mobil bayrağı olmadan masaüstü davranışı).
def test_default_mode_is_desktop(monkeypatch, processed_df):
    assert is_mobile_mode(None) is False          # kontrol temizlenmiş
    assert is_mobile_mode("Masaüstü") is False
    assert chart_height() == 900                   # varsayılan = masaüstü
    fig = _capture_fig(monkeypatch, **_base_kwargs(processed_df))  # mobile default False
    assert fig.layout.height == 900


# C9 — Oturum kalıcılığı: mod çözücü mobil etiketini onurlandırır (widget
# key'i rerun'lar arası korur; runtime kalıcılığı ayrıca AppTest ile test edilir).
def test_mode_resolver_honors_mobile_selection():
    assert is_mobile_mode(UIConfig.VIEW_MODE_MOBILE) is True


# BULGU-7 — UI etiketi ile is_mobile_mode mantığı aynı sabite bağlı olmalı;
# etiket değişirse mantık sessizce masaüstüne düşmemeli.
def test_mode_logic_bound_to_shared_label_constant():
    assert is_mobile_mode(UIConfig.VIEW_MODE_MOBILE) is True
    assert is_mobile_mode(UIConfig.VIEW_MODE_DESKTOP) is False
    assert UIConfig.VIEW_MODE_MOBILE != UIConfig.VIEW_MODE_DESKTOP


# C3 — Mobil yükseklik 560.
def test_mobile_chart_height_is_560(monkeypatch, processed_df):
    assert chart_height(mobile=True) == 560
    fig = _capture_fig(monkeypatch, **_base_kwargs(processed_df, mobile=True))
    assert fig.layout.height == 560


# C4 — Mobil zoom penceresi masaüstünün yarısı (1wk:25, 1d:40, diğer:50).
@pytest.mark.parametrize("view_tf, expected", [("1wk", 25), ("1d", 40), ("4h", 50)])
def test_mobile_zoom_window_is_half(monkeypatch, processed_df, view_tf, expected):
    assert resolve_zoom_count(view_tf, mobile=True) == expected
    fig = _capture_fig(monkeypatch, **_base_kwargs(processed_df, view_tf=view_tf, mobile=True))
    # Opening window starts `expected` bars before the last candle.
    assert pd.Timestamp(fig.layout.xaxis.range[0]) == processed_df.index[-expected]


# C5 — Zoom penceresi veri gizlemez: aynı veri noktaları, yalnızca açılış
# x-range başlangıcı farklı.
def test_mobile_zoom_hides_no_data(monkeypatch, processed_df):
    desktop = _capture_fig(monkeypatch, **_base_kwargs(processed_df, mobile=False))
    mobile = _capture_fig(monkeypatch, **_base_kwargs(processed_df, mobile=True))

    d_candle = next(t for t in desktop.data if t.type == "candlestick")
    m_candle = next(t for t in mobile.data if t.type == "candlestick")
    assert len(m_candle.x) == len(d_candle.x)                             # veri gizlenmedi
    assert mobile.layout.xaxis.range[0] != desktop.layout.xaxis.range[0]  # yalnızca pencere
    assert mobile.layout.xaxis.range[1] == desktop.layout.xaxis.range[1]  # sağ kenar aynı


# C6 — Masaüstü regresyonu yok: yükseklik 900 ve yerleşim değişmemiş.
def test_desktop_layout_unchanged(monkeypatch, processed_df):
    fig = _capture_fig(monkeypatch, **_base_kwargs(processed_df, mobile=False))
    layout = fig.layout
    assert layout.height == 900
    assert layout.yaxis.side == "right"
    assert layout.hovermode == "x unified"
    assert layout.dragmode == "pan"
    assert (layout.margin.l, layout.margin.r, layout.margin.t, layout.margin.b) == (10, 60, 10, 20)


def _inject_advanced_patterns(monkeypatch, df_view):
    """Make the f_advanced branch deterministic: real detect_advanced_patterns
    finds 0 formations on the synthetic fixture (QA BULGU-5), so inject a known
    triangle + reversal that exercise every advanced draw path (boundary lines,
    HEDEF/target lines, neckline, 'Baş' annotation). Returns nothing; both modes
    then draw the same extra shapes/annotations."""
    idx = df_view.index
    close = df_view['Close']
    d0, d1, dmid = idx[10], idx[-5], idx[len(idx) // 2]
    adv = [
        {'type': 'triangle', 'color': 'blue', 'target': float(close.max() * 1.1),
         'lines': [{'x0': d0, 'y0': float(close.iloc[10]), 'x1': d1, 'y1': float(close.iloc[-5])},
                   {'x0': d0, 'y0': float(close.iloc[10]) * 0.98, 'x1': d1, 'y1': float(close.iloc[-5]) * 0.98}]},
        {'type': 'reversal', 'color': 'red', 'target': float(close.min() * 0.9),
         'neckline': float(close.iloc[-5]), 'x0': d0, 'y0': float(close.iloc[10]),
         'head_x': dmid, 'head_y': float(close.loc[dmid]) * 1.03, 'x1': d1, 'y1': float(close.iloc[-5])},
    ]
    monkeypatch.setattr(ui_components, "detect_advanced_patterns", lambda d: adv)
    monkeypatch.setattr(ui_components, "get_pattern_status", lambda a, c: None)


# C7 — Özellik eşitliği (temel katmanlar): trace + shape + annotation eşit.
def test_feature_parity_element_counts(monkeypatch, processed_df):
    desktop = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=False, f_advanced=False))
    mobile = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=True, f_advanced=False))
    assert len(mobile.data) == len(desktop.data)
    assert len(mobile.layout.shapes) == len(desktop.layout.shapes)
    assert len(mobile.layout.annotations) == len(desktop.layout.annotations)


# C7 (çekirdek) — Özellik eşitliği, en ağır katman (gelişmiş formasyonlar) AÇIKKEN.
# Sentetik fixture'da gerçek detektör 0 formasyon bulduğu için bilinen bir
# formasyon seti enjekte edilir; hem parity doğrulanır hem de gelişmiş katmanın
# gerçekten çizildiği (True sayımı > False sayımı) kanıtlanır (QA BULGU-5).
def test_feature_parity_with_advanced_patterns(monkeypatch, processed_df):
    # Baseline (advanced kapalı) — enjeksiyon öncesi gerçek detektörle.
    base = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=False, f_advanced=False))
    base_elems = len(base.layout.shapes) + len(base.layout.annotations)

    _inject_advanced_patterns(monkeypatch, processed_df)
    desktop = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=False, f_advanced=True))
    mobile = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=True, f_advanced=True))

    # Gelişmiş katman gerçekten öğe ekledi mi? (test boşa koşmuyor)
    adv_elems = len(desktop.layout.shapes) + len(desktop.layout.annotations)
    assert adv_elems > base_elems

    # Ve iki mod bu ağır katmanla da birebir eşit.
    assert len(mobile.data) == len(desktop.data)
    assert len(mobile.layout.shapes) == len(desktop.layout.shapes)
    assert len(mobile.layout.annotations) == len(desktop.layout.annotations)


# C8 — Etkileşim eşitliği: pan/zoom/hover ayarları ve modebar config'i eşit.
def test_interaction_parity(monkeypatch, processed_df):
    captured = []
    monkeypatch.setattr(st, "plotly_chart",
                        lambda fig, **kwargs: captured.append((fig, kwargs.get("config"))))
    render_main_chart(**_base_kwargs(processed_df, mobile=False))
    render_main_chart(**_base_kwargs(processed_df, mobile=True))
    (d_fig, d_cfg), (m_fig, m_cfg) = captured
    assert m_fig.layout.dragmode == d_fig.layout.dragmode
    assert m_fig.layout.hovermode == d_fig.layout.hovermode
    assert m_cfg == d_cfg  # scrollZoom, modebar butonları vb. aynı


# C10 — Değer koruması: mod değişimi fiyat/tahmin değerlerini değiştirmez.
def test_mode_change_preserves_values(monkeypatch, processed_df):
    desktop = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=False, f_advanced=True))
    mobile = _capture_fig(monkeypatch, **_all_layers_kwargs(processed_df, mobile=True, f_advanced=True))

    d_candle = next(t for t in desktop.data if t.type == "candlestick")
    m_candle = next(t for t in mobile.data if t.type == "candlestick")
    assert list(m_candle.close) == list(d_candle.close)
    assert list(m_candle.open) == list(d_candle.open)

    d_pred = next(t for t in desktop.data if t.name.startswith("AI Tahmini"))
    m_pred = next(t for t in mobile.data if t.name.startswith("AI Tahmini"))
    assert list(m_pred.y) == list(d_pred.y)


# C11 — Boş durum: None ve boş df çizilemez sayılır (veri yok), dolu df çizilir.
def test_empty_state_not_renderable():
    assert is_chart_renderable(None) is False
    assert is_chart_renderable(pd.DataFrame()) is False


def test_populated_df_is_renderable(processed_df):
    assert is_chart_renderable(processed_df) is True


# BULGU-4 — Boş durum ile hata durumu ayrı: "veri yok" sebepleri boş sayılır,
# fetch/işleme hataları hata sayılır.
@pytest.mark.parametrize("reason, is_empty", [
    ("Veri Yok (Yahoo)", True),
    ("Yetersiz Veri", True),
    ("Veri Alınamadı", False),
    ("Veri Hesaplanamadı", False),
    ("İşleme Hatası", False),
    (None, False),
])
def test_empty_vs_error_reason_classification(reason, is_empty):
    assert is_empty_data_reason(reason) is is_empty


# C12 — Hata durumu: kullanıcıya dostu mesaj, teknik metin/stack trace yok;
# detay log'a yazılır.
def test_draw_error_shows_friendly_message_and_logs(monkeypatch, processed_df):
    errors, logged = [], []
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **kwargs: None)
    monkeypatch.setattr(st, "error", lambda msg, *a, **k: errors.append(msg))
    monkeypatch.setattr(ui_components.logger, "error",
                        lambda *a, **k: logged.append((a, k)))
    # Force an exception inside the guarded layout block.
    monkeypatch.setattr(ui_components.theme, "plotly_base_layout",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom secret detail")))

    render_main_chart(**_base_kwargs(processed_df))

    assert errors, "kullanıcıya bir mesaj gösterilmeli"
    assert not any("boom secret detail" in str(m) for m in errors)  # sızıntı yok
    assert not any("Traceback" in str(m) for m in errors)
    assert logged and logged[0][1].get("exc_info") is True          # detay log'a
