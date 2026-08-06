# Chart UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix mobile chart cropping, replace the obstructive hover crosshair with a thin TradingView-style one, and make formation target lines visually distinct with lifecycle-aware status messages (forming / broke out / target reached).

**Architecture:** Two small Plotly `config`/`layout` tweaks in `render_main_chart` (`ui_components.py`) for the mobile + crosshair fixes; one new pure function `get_pattern_status` in `technical_analysis.py` for pattern lifecycle logic; `render_main_chart` calls it while drawing target lines and returns the resulting status messages as a third return value; `app.py` displays them in the existing decision-panel "🧠 Tespitler" block.

**Tech Stack:** Streamlit, Plotly (`plotly.graph_objects`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-06-chart-ux-improvements-design.md` — read it if anything below is ambiguous.
- Status messages are Turkish, matching the rest of the app's UI.
- Pattern-status scope is **triangle, reversal (H&S), and continuation (flag) only**. Harmonic patterns (`type == "harmonic"`) are explicitly out of scope — `get_pattern_status` returns `None` for them, and callers must leave their existing (target-line-less) rendering untouched.
- Target line style for all in-scope patterns: `theme.PALETTE["accent"]` color, `line_dash="longdash"`, `line_width=3` — replaces the old `adv['color']`-based styling so the target never blends into the pattern's own boundary lines.
- Exact message strings (stage-dependent, all in `get_pattern_status`):
  - forming: `"🔍 Formasyon giriş koşulları sağlandı, kırılım bekleniyor"`
  - broke_out bullish: `f"📈 {name} yukarı kırıldı, ${target:,.2f} hedefliyor"`
  - broke_out bearish: `f"📉 {name} aşağı kırıldı, ${target:,.2f} hedefliyor"`
  - target_reached: `f"🎯 Hedefe ulaşıldı (${target:,.2f})"`
- `render_main_chart`'s return signature changes from `(s_list, r_list)` to `(s_list, r_list, pattern_statuses)` — every call site must be updated in the same task that changes it (Task 3 also fixes `app.py`... no — see Task 4, which handles the `app.py` call site specifically, since it also adds the UI display).
- No new files outside `ui_components.py`, `technical_analysis.py`, `app.py`, and their test files. No new dependencies.

---

### Task 1: Mobile responsive fix + TradingView-style crosshair

**Files:**
- Modify: `ui_components.py:200-219` (inside `render_main_chart`)
- Test: `tests/test_chart_layout.py` (new)

**Interfaces:**
- Consumes: nothing new — pure edit of `render_main_chart`'s existing internal `base_layout`/`config` dicts. `render_main_chart`'s parameter list and first two return values (`s_list`, `r_list`) are unchanged by this task.
- Produces: nothing new is exposed — later tasks don't depend on this task's internals.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chart_layout.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chart_layout.py -v`
Expected: both tests FAIL — `captured["config"]["responsive"]` raises `KeyError` (no `responsive` key yet), and `layout.hovermode` is `"x unified"` not `"x"`.

- [ ] **Step 3: Implement the layout/config change**

In `ui_components.py`, replace (around line 200-219):

```python
        base_layout = theme.plotly_base_layout()
        base_layout.update(
            height=900, 
            xaxis_rangeslider_visible=False, 
            dragmode="pan",
            yaxis=dict(side="right", fixedrange=False, type=y_type, range=range_y, tickformat=".2f", exponentformat="none"),
            xaxis=dict(range=[zoom_start, zoom_end], type="date"),
            margin=dict(l=10, r=60, t=10, b=20),
            hovermode='x unified'
        )
        fig.update_layout(**base_layout)

        config = {
            'scrollZoom': True, 
            'displayModeBar': True, 
            'editable': False, 
            'showAxisRangeEntryBoxes': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
            'displaylogo': False
        }
```

with:

```python
        spike_style = dict(
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikethickness=1, spikedash='solid',
            spikecolor='rgba(230,234,242,0.35)',
        )
        base_layout = theme.plotly_base_layout()
        base_layout.update(
            height=900, 
            xaxis_rangeslider_visible=False, 
            dragmode="pan",
            yaxis=dict(side="right", fixedrange=False, type=y_type, range=range_y, tickformat=".2f", exponentformat="none", **spike_style),
            xaxis=dict(range=[zoom_start, zoom_end], type="date", **spike_style),
            margin=dict(l=10, r=60, t=10, b=20),
            hovermode='x'
        )
        fig.update_layout(**base_layout)

        config = {
            'scrollZoom': True, 
            'displayModeBar': True, 
            'editable': False, 
            'showAxisRangeEntryBoxes': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
            'displaylogo': False,
            'responsive': True
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chart_layout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui_components.py tests/test_chart_layout.py
git commit -m "$(cat <<'EOF'
Fix mobile chart cropping and replace obstructive hover line with thin crosshair

config['responsive'] wasn't set, so Plotly could render at a stale width
on mobile, cropping the chart. hovermode='x unified' also drew a thick
line across the whole chart; native axis spikes give a thin TradingView-
style crosshair with axis-edge price/date labels instead.
EOF
)"
```

---

### Task 2: `get_pattern_status` lifecycle function

**Files:**
- Modify: `technical_analysis.py` (add new function after `detect_advanced_patterns`, i.e. after line 468)
- Test: `tests/test_pattern_status.py` (new)

**Interfaces:**
- Consumes: pattern dicts shaped exactly like `detect_advanced_patterns`'s output (see that function, `technical_analysis.py:320-468`, for the exact keys per `type`).
- Produces: `get_pattern_status(pattern: dict, current_price: float) -> dict | None`. On a non-`None` return: `{"stage": "forming"|"broke_out"|"target_reached", "target": float, "direction": "BULLISH"|"BEARISH"|None, "message": str}`. Task 3 calls this directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pattern_status.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pattern_status.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_pattern_status' from 'technical_analysis'`

- [ ] **Step 3: Implement `get_pattern_status`**

In `technical_analysis.py`, add after `detect_advanced_patterns` (after line 468, before `def calculate_trade_setup`):

```python
def get_pattern_status(pattern, current_price):
    """Lifecycle status (forming -> broke_out -> target_reached) for
    triangle/reversal/continuation patterns.

    Harmonic patterns have no boundary-line concept (point D is already a
    confirmed historical pivot at detection time, so there's no "still
    forming" state to detect) -- this returns None for them; callers must
    keep rendering harmonic patterns exactly as before.
    """
    ptype = pattern["type"]
    name = pattern["name"]

    if ptype == "triangle":
        resistance_at_last = pattern["lines"][0]["y1"]
        support_at_last = pattern["lines"][1]["y1"]
        broke_up = current_price > resistance_at_last
        broke_down = current_price < support_at_last

        if pattern["direction"] == "BULLISH":
            broke_down = False  # ascending triangle only ever breaks up
        elif pattern["direction"] == "BEARISH":
            broke_up = False    # descending triangle only ever breaks down

        if not broke_up and not broke_down:
            return {"stage": "forming", "target": pattern["target"], "direction": None,
                    "message": "🔍 Formasyon giriş koşulları sağlandı, kırılım bekleniyor"}

        direction = "BULLISH" if broke_up else "BEARISH"
        if pattern["direction"] == "NEUTRAL":
            # Symmetric triangle: the stored target is a pre-breakout
            # placeholder (== current_price at detection). Project a real
            # measured-move target from the triangle's height at its
            # widest point, anchored to the fixed breakout boundary (not
            # to current_price, which would make the target unreachable).
            height = pattern["lines"][0]["y0"] - pattern["lines"][1]["y0"]
            breakout_level = resistance_at_last if direction == "BULLISH" else support_at_last
            target = breakout_level + height if direction == "BULLISH" else breakout_level - height
        else:
            target = pattern["target"]

        reached = current_price >= target if direction == "BULLISH" else current_price <= target
        if reached:
            return {"stage": "target_reached", "target": target, "direction": direction,
                    "message": f"🎯 Hedefe ulaşıldı (${target:,.2f})"}

        arrow = "📈" if direction == "BULLISH" else "📉"
        word = "yukarı" if direction == "BULLISH" else "aşağı"
        return {"stage": "broke_out", "target": target, "direction": direction,
                "message": f"{arrow} {name} {word} kırıldı, ${target:,.2f} hedefliyor"}

    if ptype == "reversal":
        neckline = pattern["neckline"]
        target = pattern["target"]
        if current_price >= neckline:
            return {"stage": "forming", "target": target, "direction": None,
                    "message": "🔍 Formasyon giriş koşulları sağlandı, kırılım bekleniyor"}
        if current_price <= target:
            return {"stage": "target_reached", "target": target, "direction": "BEARISH",
                    "message": f"🎯 Hedefe ulaşıldı (${target:,.2f})"}
        return {"stage": "broke_out", "target": target, "direction": "BEARISH",
                "message": f"📉 {name} aşağı kırıldı, ${target:,.2f} hedefliyor"}

    if ptype == "continuation":
        box_top = pattern["y1"]
        target = pattern["target"]
        if current_price <= box_top:
            return {"stage": "forming", "target": target, "direction": None,
                    "message": "🔍 Formasyon giriş koşulları sağlandı, kırılım bekleniyor"}
        if current_price >= target:
            return {"stage": "target_reached", "target": target, "direction": "BULLISH",
                    "message": f"🎯 Hedefe ulaşıldı (${target:,.2f})"}
        return {"stage": "broke_out", "target": target, "direction": "BULLISH",
                "message": f"📈 {name} yukarı kırıldı, ${target:,.2f} hedefliyor"}

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pattern_status.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add technical_analysis.py tests/test_pattern_status.py
git commit -m "$(cat <<'EOF'
Add get_pattern_status for formation lifecycle (forming/broke out/target reached)

Also fixes the symmetric triangle's placeholder target (== current_price
at detection, making "targeting $X" nonsensical) by projecting a real
measured-move target from the breakout boundary once the triangle
actually breaks out.
EOF
)"
```

---

### Task 3: Wire pattern status into `render_main_chart`

**Files:**
- Modify: `ui_components.py` (import line 6, top of `render_main_chart`, the `f_advanced` drawing block at lines ~126-163, and the `return` statement at line 225)
- Test: `tests/test_chart_patterns.py` (extend existing file)

**Interfaces:**
- Consumes: `technical_analysis.get_pattern_status(pattern, current_price)` from Task 2.
- Produces: `render_main_chart(...)` now returns `(s_list, r_list, pattern_statuses)` where `pattern_statuses` is `list[str]`. Task 4's `app.py` call site depends on this exact 3-tuple shape.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chart_patterns.py`:

```python
import streamlit as st

import ui_components
import theme
from ui_components import render_main_chart


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chart_patterns.py -v`
Expected: FAIL — `render_main_chart` still returns a 2-tuple (`len(result) == 3` fails / unpacking error), and no shape matches the new target-line style.

- [ ] **Step 3: Implement the wiring**

In `ui_components.py`, update the import (line 6):

```python
from technical_analysis import calculate_sr_advanced, detect_advanced_patterns, calculate_oracle_signal_v2, calculate_trade_setup, get_pattern_status
```

Add a module-level constant right after the imports (after line 7):

```python
TARGET_LINE_COLOR = theme.PALETTE["accent"]
```

Initialize `pattern_statuses` at the top of `render_main_chart`, right after `fig = go.Figure()` (line 58):

```python
    fig = go.Figure()
    pattern_statuses = []
```

Replace the `f_advanced` block's triangle/reversal/continuation branches (the existing harmonic branch is untouched):

```python
                for adv in advanced_items:
                    if adv['type'] == 'triangle':
                        # Draw BOTH boundary lines (resistance + support) so
                        # the actual triangle shape shows, not just one edge.
                        for line in adv.get('lines', []):
                            fig.add_shape(type="line", x0=line['x0'], y0=line['y0'], x1=line['x1'], y1=line['y1'],
                                          line=dict(color=adv['color'], width=2, dash='dot'))
                        status = get_pattern_status(adv, curr)
                        target = status['target'] if status else adv['target']
                        fig.add_hline(y=target, line_dash="longdash", line_width=3, line_color=TARGET_LINE_COLOR,
                                      annotation_text=f"🎯 Hedef: ${target:,.2f}")
                        if status:
                            pattern_statuses.append(status['message'])

                    elif adv['type'] == 'harmonic':
                        points = adv['points']
                        for i in range(len(points) - 1):
                            fig.add_shape(type="line", x0=df_view.index[points[i]['idx']], y0=points[i]['price'], x1=df_view.index[points[i+1]['idx']], y1=points[i+1]['price'], line=dict(color=adv['color'], width=2))
                        fig.add_annotation(x=df_view.index[points[2]['idx']], y=points[2]['price'], text=adv['name'], showarrow=True, arrowhead=2, bgcolor=adv['color'], font=dict(color='white'))

                    elif adv['type'] == 'reversal':
                        # Draw the actual left-shoulder -> head -> right-shoulder
                        # shape (previously only the neckline/target were shown).
                        if 'head_x' in adv:
                            fig.add_shape(type="line", x0=adv['x0'], y0=adv['y0'], x1=adv['head_x'], y1=adv['head_y'], line=dict(color=adv['color'], width=2))
                            fig.add_shape(type="line", x0=adv['head_x'], y0=adv['head_y'], x1=adv['x1'], y1=adv['y1'], line=dict(color=adv['color'], width=2))
                            fig.add_annotation(x=adv['head_x'], y=adv['head_y'], text="Baş", showarrow=True, arrowhead=2, bgcolor=adv['color'], font=dict(color='white'))
                        fig.add_hline(y=adv['neckline'], line_dash="solid", line_color=adv['color'], annotation_text="Boyun Çizgisi")
                        status = get_pattern_status(adv, curr)
                        target = status['target'] if status else adv['target']
                        fig.add_hline(y=target, line_dash="longdash", line_width=3, line_color=TARGET_LINE_COLOR,
                                      annotation_text=f"🎯 Hedef: ${target:,.2f}")
                        if status:
                            pattern_statuses.append(status['message'])

                    elif adv['type'] == 'continuation':
                        # Draw the flag/pole channel as a shaded box (previously
                        # this coordinate data was computed but discarded — only
                        # a text label was shown, which is the "one line" bug).
                        fig.add_shape(type="rect", x0=adv['x0'], y0=adv['y0'], x1=adv['x1'], y1=adv['y1'],
                                      line=dict(color=adv['color'], width=2), fillcolor=adv['color'], opacity=0.15)
                        status = get_pattern_status(adv, curr)
                        target = status['target'] if status else adv['target']
                        fig.add_hline(y=target, line_dash="longdash", line_width=3, line_color=TARGET_LINE_COLOR,
                                      annotation_text=f"🎯 Hedef: ${target:,.2f}")
                        if status:
                            pattern_statuses.append(status['message'])
```

Finally, change the return statement (line 225):

```python
    return s_list, r_list, pattern_statuses
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chart_patterns.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Run the full test suite to check for other call sites**

Run: `pytest -v`
Expected: everything passes except (if not yet updated) any test/code that unpacks `render_main_chart`'s return as a 2-tuple. If `app.py` is the only other caller (confirm with a repo-wide search: `grep -rn "render_main_chart(" --include=*.py .`), this is expected and is fixed in Task 4 — do not modify `app.py` in this task.

- [ ] **Step 6: Commit**

```bash
git add ui_components.py tests/test_chart_patterns.py
git commit -m "$(cat <<'EOF'
Give formation target lines a distinct style and lifecycle status messages

Target lines previously reused the pattern's own color, blending into its
boundary/neckline/box lines. They now use one consistent high-contrast
style, and render_main_chart returns a lifecycle status message
(forming / broke out / target reached) per active triangle, H&S, and flag
pattern for the decision panel to display.
EOF
)"
```

---

### Task 4: Display status messages in the decision panel

**Files:**
- Modify: `app.py:168` (call site) and `app.py:233-242` (the "🧠 Tespitler" block)

**Interfaces:**
- Consumes: `render_main_chart(...)` returning `(s_list, r_list, pattern_statuses)` from Task 3.
- Produces: nothing further downstream — this is the final consumer.

`app.py` has no existing automated test coverage (it's a top-level Streamlit script, not imported/tested anywhere in `tests/` — confirmed: no test file references it). This task is verified by a manual smoke test instead of pytest, consistent with how the rest of `app.py` is handled in this repo.

- [ ] **Step 1: Update the call site**

In `app.py`, change line 168 from:

```python
    s_l, r_l = render_main_chart(df_view, view_tf, curr, f_dates, f_prices, ai_score, show_cloud, show_pred, show_ai, show_all_pats, f_wm, f_candle, f_advanced, items_raw, lines)
```

to:

```python
    s_l, r_l, adv_pattern_statuses = render_main_chart(df_view, view_tf, curr, f_dates, f_prices, ai_score, show_cloud, show_pred, show_ai, show_all_pats, f_wm, f_candle, f_advanced, items_raw, lines)
```

- [ ] **Step 2: Display the statuses in the decision panel**

In `app.py`, the "🧠 Tespitler" block (lines 233-242) currently ends with:

```python
        st.markdown("---")
        st.markdown("**🧠 Tespitler**")
        if show_all_pats and items_raw:
            visible_names = []
            for item in items_raw:
                if (item['type'] == 'box' and f_wm) or (item['type'] == 'icon' and f_candle): visible_names.append(item['name'])
            if visible_names:
                for p in list(set(visible_names)): st.caption(PATTERN_INFO.get(p, p))
            else: st.caption("Filtreli formasyon yok.")
        else: st.caption("Formasyonlar kapalı.")
```

Add immediately after it (still inside the `with dash_col3:` block, same indentation):

```python

        if adv_pattern_statuses:
            st.markdown("---")
            st.markdown("**📐 Formasyon Durumu**")
            for msg in adv_pattern_statuses:
                st.caption(msg)
```

- [ ] **Step 3: Run the full automated test suite**

Run: `pytest -v`
Expected: all tests PASS (this confirms Task 3's changes didn't break anything now that the only other call site is updated).

- [ ] **Step 4: Manual smoke test**

Run: `streamlit run app.py`
In the browser: pick any asset, enable "Gelişmiş Formasyonlar (Üçgen, ABCD, Baş-Omuz)" in the sidebar, and confirm:
- the app loads without error/traceback,
- the main chart renders full-width (resize the browser window narrow to simulate mobile — the chart should re-flow, not require horizontal scrolling),
- hovering over the chart shows a thin crosshair with small price/date labels on the axes, not a thick line,
- if any triangle/H&S/flag pattern is currently active for that asset, its target line is a thick amber-gold dashed line distinct from the pattern's own color, and a matching status line appears under "📐 Formasyon Durumu" in the decision panel's third column.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "$(cat <<'EOF'
Show formation lifecycle status in the decision panel

Wires render_main_chart's new pattern_statuses return value into the
existing "Tespitler" block so users see e.g. "broke out targeting $X" or
"target reached" for active triangle/H&S/flag formations, not just their
name.
EOF
)"
```
