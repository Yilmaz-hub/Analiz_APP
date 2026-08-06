# Chart UX Improvements — Design

## Goal

Three independent fixes to the main price chart, driven directly by user
feedback on the deployed app:

1. On mobile, the chart renders wider than the viewport — only the
   left/right portion is visible, requiring horizontal scroll.
2. The hover crosshair (Plotly's default `hovermode='x unified'`) draws a
   thick line that obscures candles, with no TradingView-style axis-edge
   price/date readout.
3. Formation target lines (triangle / head-and-shoulders / flag) visually
   blend with the pattern's own boundary lines, making the target hard to
   spot. There's also no indication of where a pattern is in its lifecycle
   (still forming vs. broken out vs. target already reached).

All three are confined to `render_main_chart` in `ui_components.py`, plus a
new function in `technical_analysis.py` for part 3. No new files, no new
dependencies.

## 1. Mobile responsive fix

`st.plotly_chart(fig, use_container_width=True, config=config, ...)`
(`ui_components.py:221`) sizes the figure once at first paint via
`use_container_width`, but the `config` dict has no `'responsive': True`.
Without it, Plotly.js doesn't re-flow if the real mobile viewport width
settles differently after first paint — the documented cause of "chart
renders wider than the visible screen, needs horizontal scroll" in
Streamlit + Plotly on mobile.

Fix: add `'responsive': True` to the existing `config` dict
(`ui_components.py:212-219`), alongside `scrollZoom`/`displayModeBar`.
One-line change. Nothing else forces a fixed pixel width today, and no
custom CSS in `theme.py` constrains the chart's iframe, so no other change
is needed here.

## 2. Crosshair redesign

Current layout (`ui_components.py:200-209`) sets `hovermode='x unified'`,
which draws Plotly's merged hover box plus a thick vertical line across the
whole plot — the reported "white line obstructing candles."

Replace with:
- `hovermode='x'` (drops the unified box and its thick line).
- Native Plotly axis spikes on both axes, added to the existing
  `xaxis=dict(...)` / `yaxis=dict(...)` entries in `base_layout`:
  ```python
  spike_style = dict(
      showspikes=True, spikemode='across', spikesnap='cursor',
      spikethickness=1, spikedash='solid',
      spikecolor='rgba(230,234,242,0.35)',  # theme.PALETTE["text"] at low opacity
  )
  ```
  merged into both axis dicts.

Axis spikes are a built-in Plotly feature: enabling them on both axes
automatically renders a thin crosshair plus small highlighted labels right
on the axes themselves — price on the right y-axis, date on the bottom
x-axis — at the cursor's position. This is the TradingView look with no
custom tooltip/annotation code required. Hovering directly on a candle
still shows its OHLC via Plotly's normal per-trace hover, which is
independent of `hovermode='x'` vs `'x unified'`.

## 3. Formation target clarity + lifecycle status

Scope: **triangle, reversal (head-and-shoulders), and continuation (flag)**
pattern types — the three that already draw a target `hline` today
(`ui_components.py:126-163`). Harmonic patterns (ABCD, Butterfly) are
explicitly out of scope: they have no boundary-line concept (point D is
already a confirmed historical pivot at detection time, so there is no
"still forming" state to detect), and the existing chart code doesn't draw
a target line for them at all today. Extending status logic to harmonic
patterns would require new design decisions about what "target" even means
there — deferred (see "Rejected/deferred alternatives").

### Target line styling

Today the target `hline` reuses the pattern's own `adv['color']`
(`ui_components.py:137,153,161`), which is the same color as the
pattern's boundary/neckline/box lines — so where they cross, the target
blends in.

Fix: draw all in-scope target lines with one consistent, high-contrast
style instead of `adv['color']`:
```python
TARGET_LINE_COLOR = theme.PALETTE["accent"]  # #F5B841, amber-gold
fig.add_hline(y=target, line_dash="longdash", line_width=3,
              line_color=TARGET_LINE_COLOR,
              annotation_text=f"🎯 Hedef: ${target:,.2f}")
```
`longdash` + `width=3` is visually distinct from the boundary lines'
`dot`/`solid` `width=2` styling regardless of which pattern color it
crosses, and the amber-gold accent color is already the app's theme color
for "this is the important highlighted thing" (verdict lamp, accent
borders), so it reads as a deliberate convention rather than an arbitrary
new color. The annotation text also changes from the current bare
`f"🎯 {name}"` to include the price directly on the line.

### Symmetric triangle target fix

`detect_advanced_patterns` currently sets the symmetric triangle's target
to `current_price` at detection time (`technical_analysis.py:374`) — a
placeholder that makes "targeting $X" nonsensical, since X is always
today's price. Ascending/descending triangles already use a real
(percentage-based) projected target and are untouched by this fix.

Fix: once a symmetric triangle's status resolves to `broke_out` (see
below), compute a real measured-move target from the triangle's height at
its widest point, projected from the **breakout boundary level** — not
from `current_price` directly. (An earlier draft of this formula used
`current_price + height`, which is wrong: since `current_price` is what's
being compared against the target on every call, a target defined relative
to it can never be "reached" — it would always sit `height` above/below
wherever the price currently is. Anchoring to the fixed boundary level
instead gives a real, static target that price can actually catch up to.)
```python
height = pattern["lines"][0]["y0"] - pattern["lines"][1]["y0"]
breakout_level = resistance_at_last if resolved_direction == "BULLISH" else support_at_last
target = breakout_level + height if resolved_direction == "BULLISH" \
    else breakout_level - height
```
This target is computed by `get_pattern_status` (next section) at render
time — `detect_advanced_patterns` keeps storing `target: current_price`
as a pre-breakout placeholder (only ever shown/used once resolved).

### Pattern status function

New function in `technical_analysis.py`:

```python
def get_pattern_status(pattern: dict, current_price: float) -> dict | None:
    """Lifecycle status for triangle/reversal/continuation patterns.
    Returns None for out-of-scope types (harmonic).
    """
```

Returns `{"stage": ..., "message": ..., "target": ..., "direction": ...}`.

**Boundary/breakout check per type:**
| Type | Forming while | Breakout (bullish) | Breakout (bearish) |
|---|---|---|---|
| Ascending triangle | `current_price <= resistance_at_last` | `current_price > resistance_at_last` | — (direction fixed BULLISH) |
| Descending triangle | `current_price >= support_at_last` | — (direction fixed BEARISH) | `current_price < support_at_last` |
| Symmetric triangle | inside both lines | `current_price > resistance_at_last` | `current_price < support_at_last` |
| Reversal (H&S) | `current_price >= neckline` | n/a (BEARISH only) | `current_price < neckline` |
| Continuation (flag) | `current_price <= y1` (box top) | `current_price > y1` | n/a (BULLISH only) |

`resistance_at_last`/`support_at_last` read straight from
`pattern["lines"][0]["y1"]` / `pattern["lines"][1]["y1"]` (triangle only;
already computed by `detect_advanced_patterns`).

**Stages, in order:**
1. **`forming`** — price still inside the boundary.
   Message: `"🔍 Formasyon giriş koşulları sağlandı, kırılım bekleniyor"`
2. **`broke_out`** — price closed outside the boundary. Direction-aware:
   - Bullish: `f"📈 {name} yukarı kırıldı, ${target:,.2f} hedefliyor"`
   - Bearish: `f"📉 {name} aşağı kırıldı, ${target:,.2f} hedefliyor"`
   (For the symmetric triangle, `target` here is the freshly computed
   measured-move value, not the stored placeholder.)
3. **`target_reached`** — direction-aware check against `target`:
   - Bullish: `current_price >= target`
   - Bearish: `current_price <= target`
   Message: `f"🎯 Hedefe ulaşıldı (${target:,.2f})"`

Once `target_reached` is true it takes priority over `broke_out` (checked
in that order — reached implies already broken out).

### UI placement

`render_main_chart`'s existing advanced-pattern loop (`ui_components.py`,
inside `if f_advanced:`) already iterates `detect_advanced_patterns`
results to draw shapes. Extend it to also call `get_pattern_status(adv,
curr)` once per pattern, and:
- draw the target `hline` using the status's `target` (falls back to
  `adv['target']` when `get_pattern_status` returns `None`, i.e. harmonic),
- collect non-`None` statuses into a list.

`render_main_chart`'s return signature changes from `(s_list, r_list)` to
`(s_list, r_list, pattern_statuses)` — `pattern_statuses` is `[]` whenever
`show_all_pats`/`f_advanced` are off, matching the existing gating.

`app.py` (`app.py:168`) captures the third return value and passes it into
the decision panel's third column (`dash_col3`, `app.py:218-242`), which
already has a **"🧠 Tespitler"** block listing simple-pattern names via
`PATTERN_INFO` captions (`app.py:234-242`) — currently empty/generic when
no simple patterns are active. Add one `st.caption`-style line per
`pattern_statuses` entry there, right after the existing simple-pattern
list, using its `message` string directly. No new card component needed —
matches the existing caption styling already used in that block.

## Testing plan

- `technical_analysis.get_pattern_status`: one test per pattern type
  (ascending/descending/symmetric triangle, reversal, continuation) times
  each stage (forming / broke_out / target_reached) using hand-built
  pattern dicts + a `current_price`, asserting the returned `stage` and
  that `target_reached` correctly takes priority over `broke_out`.
  Harmonic pattern dict → asserts `None`.
- Symmetric triangle target recompute: a dedicated test asserting the
  measured-move formula (`current_price ± height`) rather than the stored
  `current_price` placeholder is returned once `broke_out`/`target_reached`.
- `render_main_chart`: existing chart-pattern tests
  (`tests/test_chart_patterns.py`) extended to assert the new 3-tuple
  return shape and that `pattern_statuses` is empty when
  `f_advanced`/`show_all_pats` are off.

## Rollout / safety

- All three changes are additive/visual — no change to signal generation,
  scoring, or backtesting. Safe to ship independently of the weight
  optimizer work.
- The mobile fix and crosshair redesign only touch Plotly `config`/`layout`
  dicts — no behavior change on desktop beyond the crosshair's visual
  style.
- The symmetric-triangle target fix only changes displayed/messaged values
  once a breakout is detected; it doesn't change pattern *detection*
  (`detect_advanced_patterns` still fires under the same conditions as
  before).

## Rejected/deferred alternatives

- **Harmonic pattern status** — deferred: no existing boundary-line or
  target-line concept to build a lifecycle state machine on top of; would
  need its own design pass (what does "broke out" even mean for a pattern
  whose defining point D is already historical at detection?).
- **On-chart full-sentence annotations** — rejected in favor of a short
  on-chart price label (`"🎯 Hedef: $X"`) plus the full lifecycle sentence
  in the decision panel's status card — keeps the chart itself
  uncluttered, per explicit user preference.
- **4-stage lifecycle (separate "target is here" stage)** — rejected in
  favor of 3 stages; "target is here" became the on-chart target line's
  label itself rather than a distinct status-card message.
