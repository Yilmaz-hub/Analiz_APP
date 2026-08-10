"""
AMBER TERMINAL theme — central styling for the whole app.

Design language: dark precision trading terminal with warm amber phosphor
accents (classic amber-CRT heritage + this app's gold assets). Chakra Petch
for display type, IBM Plex Sans for body, IBM Plex Mono for every number.
All UI colors live HERE (as CSS custom properties + the PALETTE dict) —
don't inline new hex values in app.py / ui_components.py.
"""
import streamlit as st

# Python-side palette (for Plotly traces, which can't read CSS variables)
PALETTE = {
    "bg": "#0A0E14",
    "surface": "#111722",
    "surface2": "#161E2C",
    "border": "#232D3F",
    "grid": "#1B2434",
    "text": "#E6EAF2",
    "muted": "#8B94A7",
    "accent": "#F5B841",     # amber phosphor
    "accent_soft": "#FFD166",
    "up": "#16C784",         # AL / bullish
    "down": "#EA3943",       # SAT / bearish
    "wait": "#8B94A7",       # BEKLE
    "info": "#5AB0FF",
}

# Verdict → (color, terminal glyph). Glyphs replace emojis on the big lamp.
VERDICT_STYLE = {
    "GÜÇLÜ AL":  (PALETTE["up"],   "▲▲"),
    "AL":        (PALETTE["up"],   "▲"),
    "BEKLE":     (PALETTE["wait"], "◆"),
    "SAT":       (PALETTE["down"], "▼"),
    "GÜÇLÜ SAT": (PALETTE["down"], "▼▼"),
}

def verdict_color(verdict: str) -> str:
    return VERDICT_STYLE.get(verdict, (PALETTE["wait"], "◆"))[0]

def tone_color(direction: str) -> str:
    """BULLISH/BEARISH/anything-else → palette color (advanced analysis cards)."""
    if direction == "BULLISH": return PALETTE["up"]
    if direction == "BEARISH": return PALETTE["down"]
    return PALETTE["wait"]

# Plotly constants
CANDLE_UP = PALETTE["up"]
CANDLE_DOWN = PALETTE["down"]
CHART_FONT = "IBM Plex Mono, monospace"

def plotly_base_layout() -> dict:
    """Shared Plotly layout so every chart sits on the app background."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, color=PALETTE["muted"], size=11),
        xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
        hoverlabel=dict(font_family=CHART_FONT),
        legend=dict(bgcolor="rgba(10,14,20,0.7)", bordercolor=PALETTE["border"], borderwidth=1),
    )

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0A0E14;
    --surface: #111722;
    --surface2: #161E2C;
    --border: #232D3F;
    --text: #E6EAF2;
    --muted: #8B94A7;
    --accent: #F5B841;
    --accent-soft: #FFD166;
    --up: #16C784;
    --down: #EA3943;
    --wait: #8B94A7;
    --display: 'Chakra Petch', sans-serif;
    --body: 'IBM Plex Sans', sans-serif;
    --mono: 'IBM Plex Mono', monospace;
}

/* Plotly versions bundled by Streamlit do not consistently honor the
   spikedash/spikecolor layout properties. Enforce the TradingView-like
   crosshair appearance on the generated SVG spike lines. */
.js-plotly-plot .spikeline {
    stroke: rgba(230, 234, 242, 0.22) !important;
    stroke-width: 1px !important;
    stroke-dasharray: 4px 4px !important;
}

/* ============ ATMOSPHERE: blueprint grid + amber glow ============ */
.stApp {
    background:
        radial-gradient(1100px 500px at 75% -10%, rgba(245,184,65,0.055), transparent 60%),
        repeating-linear-gradient(0deg, rgba(139,148,167,0.033) 0 1px, transparent 1px 44px),
        repeating-linear-gradient(90deg, rgba(139,148,167,0.033) 0 1px, transparent 1px 44px),
        linear-gradient(180deg, #0B1018 0%, #0A0E14 55%, #090C11 100%);
    font-family: var(--body);
}
.block-container { padding-top: 1.6rem; padding-bottom: 5rem; }

/* ============ TYPE ============ */
h1, h2, h3, h4 { font-family: var(--display) !important; letter-spacing: 0.04em; }
h3 { text-transform: uppercase; font-size: 1.05rem !important; color: var(--text); }
h3::before { content: "▞ "; color: var(--accent); }
.stMarkdown p { font-size: 14px; }
hr { margin: 1em 0; border-color: var(--border); }

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1219 0%, #0B0F16 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] hr { border-color: var(--border); }

/* ============ METRICS ============ */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] p {
    font-family: var(--display) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.45rem !important;
    color: var(--text);
}
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

/* ============ WIDGETS ============ */
.stButton > button {
    width: 100%;
    font-family: var(--display);
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.82rem;
    background: transparent;
    color: var(--accent);
    border: 1px solid rgba(245,184,65,0.45);
    border-radius: 6px;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: var(--accent);
    color: #0A0E14;
    border-color: var(--accent);
    box-shadow: 0 0 18px rgba(245,184,65,0.35);
}
[data-testid="stExpander"] {
    background: rgba(17,23,34,0.6);
    border: 1px solid var(--border);
    border-radius: 8px;
}
[data-testid="stExpander"] summary { font-family: var(--display); letter-spacing: 0.03em; }

/* ============ AMBER TERMINAL COMPONENTS ============ */
.at-header { margin-bottom: 0.4rem; }
.at-header .eyebrow {
    font-family: var(--display);
    font-size: 0.7rem;
    letter-spacing: 0.34em;
    text-transform: uppercase;
    color: var(--accent);
}
.at-header .asset {
    font-family: var(--display);
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.15;
    color: var(--text);
}
.at-header .srcbadge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px 10px;
    margin-top: 4px;
}
.at-header .srcbadge b { color: var(--accent-soft); font-weight: 500; }

/* Verdict lamp */
.at-verdict {
    --vc: var(--wait);
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(160deg, color-mix(in srgb, var(--vc) 14%, var(--surface)), var(--surface) 65%);
    border: 1px solid color-mix(in srgb, var(--vc) 55%, var(--border));
    border-radius: 12px;
    padding: 22px 20px 18px;
    text-align: center;
    animation: at-rise 0.5s ease both;
}
.at-verdict::after { /* scanlines */
    content: "";
    position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg, rgba(255,255,255,0.022) 0 1px, transparent 1px 3px);
    pointer-events: none;
}
.at-verdict .glyph {
    font-family: var(--display);
    font-size: 40px;
    line-height: 1;
    color: var(--vc);
    text-shadow: 0 0 22px color-mix(in srgb, var(--vc) 65%, transparent);
    animation: at-pulse 3s ease-in-out infinite;
}
.at-verdict .word {
    font-family: var(--display);
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--vc);
    margin-top: 6px;
}
.at-verdict .sub { font-family: var(--mono); font-size: 12px; color: var(--muted); margin-top: 8px; }
.at-verdict .meter {
    height: 5px; border-radius: 3px; background: var(--border);
    margin: 12px 30px 0; overflow: hidden;
}
.at-verdict .meter > div {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--vc) 55%, transparent), var(--vc));
    transition: width 0.6s ease;
}
.at-chip {
    display: inline-block;
    font-family: var(--mono);
    font-size: 11px;
    padding: 3px 10px;
    margin-top: 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
    background: rgba(10,14,20,0.5);
}
.at-chip.hot { border-color: rgba(245,184,65,0.5); color: var(--accent-soft); }

/* Trade plan */
.at-plan {
    --vc: var(--wait);
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--vc);
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 8px;
    animation: at-rise 0.5s 0.08s ease both;
}
.at-plan .title {
    font-family: var(--display);
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--vc);
    margin-bottom: 8px;
}
.at-plan .row {
    display: flex; justify-content: space-between;
    font-size: 13px; padding: 3px 0;
    border-bottom: 1px dashed rgba(35,45,63,0.8);
}
.at-plan .row:last-child { border-bottom: none; }
.at-plan .row .k { color: var(--muted); }
.at-plan .row .v { font-family: var(--mono); color: var(--text); }
.at-plan .row .v.up { color: var(--up); }
.at-plan .row .v.down { color: var(--down); }

/* Center-zero dimension bars */
.at-dim { margin-bottom: 9px; animation: at-rise 0.45s ease both; }
.at-dim .lbl { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 3px; }
.at-dim .lbl b { font-family: var(--mono); }
.at-dim .track {
    position: relative; height: 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
}
.at-dim .track::before { /* zero line */
    content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px;
    width: 1px; background: var(--muted); opacity: 0.55;
}
.at-dim .fill { position: absolute; top: 2px; bottom: 2px; border-radius: 2px; transition: all 0.5s ease; }
.at-dim .fill.pos { left: 50%; background: linear-gradient(90deg, rgba(22,199,132,0.45), var(--up)); }
.at-dim .fill.neg { right: 50%; background: linear-gradient(270deg, rgba(234,57,67,0.45), var(--down)); }

/* Generic analysis card */
.at-card {
    --vc: var(--wait);
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--vc);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.at-card .title { font-family: var(--display); font-weight: 600; font-size: 15px; margin-bottom: 6px; color: var(--text); }
.at-card .desc { font-size: 13px; color: var(--muted); margin-top: 6px; }
.at-card b { color: var(--vc); }
.at-badge {
    display: inline-block; font-family: var(--mono); font-size: 10px;
    padding: 1px 7px; border-radius: 4px; margin-left: 6px;
    color: #0A0E14; background: var(--accent);
}
.at-badge.alarm { background: var(--down); color: #fff; }

/* Timeframe summary + reasons */
.at-tfrow {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 13px; padding: 5px 2px; border-bottom: 1px dashed rgba(35,45,63,0.8);
}
.at-tfrow .tf { color: var(--muted); font-family: var(--mono); font-size: 12px; }
.at-tfrow .vd { font-family: var(--display); font-weight: 600; letter-spacing: 0.05em; }
.at-reason { font-size: 13px; padding: 4px 0 4px 2px; border-bottom: 1px solid rgba(35,45,63,0.6); color: var(--text); }
.at-reason::before { content: "› "; color: var(--accent); font-family: var(--mono); }

/* Sidebar signal block */
.at-sig .tf { font-family: var(--display); font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); }
.at-sig .vd { font-family: var(--display); font-size: 1.25rem; font-weight: 700; letter-spacing: 0.08em; }

@keyframes at-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
@keyframes at-rise {
    from { opacity: 0; transform: translateY(7px); }
    to { opacity: 1; transform: none; }
}
</style>
"""

def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


# Plotly cannot draw TradingView's crosshair axis badges natively -- showing
# the cursor's own price/date against the axes is an open upstream feature
# request (plotly.js#7518), and the only in-library workaround is a Dash
# callback, which Streamlit has no equivalent for. So we read the cursor
# position off the rendered plot and position the two badges ourselves.
_AXIS_BADGE_JS = """
<script>
(function () {
    var doc;
    // Streamlit renders components in a same-origin iframe; if a deployment
    // ever sandboxes it differently, bail out and leave the chart untouched
    // rather than throwing on every mouse move.
    try { doc = window.parent.document; } catch (e) { return; }
    if (!doc) { return; }

    var BADGE_CSS = 'position:absolute;z-index:1000;pointer-events:none;display:none;'
        + 'font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:500;'
        + 'padding:2px 6px;border-radius:3px;white-space:nowrap;'
        + 'color:#0A0E14;background:__ACCENT__;';

    function pad(n) { return (n < 10 ? '0' : '') + n; }

    function formatDate(raw) {
        var d = (raw instanceof Date) ? raw : new Date(raw);
        if (isNaN(d.getTime())) { return String(raw); }
        var out = pad(d.getDate()) + '.' + pad(d.getMonth() + 1) + '.' + d.getFullYear();
        // On intraday charts the day alone cannot identify a bar.
        if (__INTRADAY__) { out += ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()); }
        return out;
    }

    function formatPrice(v) {
        if (!isFinite(v)) { return ''; }
        var digits = Math.abs(v) < 10 ? 4 : 2;
        return '$' + v.toLocaleString('en-US', {
            minimumFractionDigits: digits, maximumFractionDigits: digits
        });
    }

    function attach(gd) {
        if (!gd || gd.__atAxisBadges) { return; }
        if (!gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.yaxis) { return; }
        gd.__atAxisBadges = true;

        function make() {
            var el = doc.createElement('div');
            el.style.cssText = BADGE_CSS;
            gd.appendChild(el);
            return el;
        }
        if (!gd.style.position || gd.style.position === 'static') {
            gd.style.position = 'relative';
        }
        var priceBadge = make();
        var dateBadge = make();

        function hide() {
            priceBadge.style.display = 'none';
            dateBadge.style.display = 'none';
        }

        gd.addEventListener('mousemove', function (ev) {
            var L = gd._fullLayout;
            if (!L || !L._size) { hide(); return; }
            var box = gd.getBoundingClientRect();
            var x = ev.clientX - box.left;
            var y = ev.clientY - box.top;
            var s = L._size;
            // Only inside the plotting area -- not over the axes/margins.
            if (x < s.l || x > box.width - s.r || y < s.t || y > box.height - s.b) {
                hide();
                return;
            }
            var xa = L.xaxis, ya = L.yaxis;
            if (!xa.p2d || !ya.p2d) { hide(); return; }

            var price = ya.p2d(y - s.t);
            // Log axes carry their values as exponents.
            if (ya.type === 'log') { price = Math.pow(10, price); }
            priceBadge.textContent = formatPrice(price);
            priceBadge.style.display = 'block';
            priceBadge.style.top = (y - 9) + 'px';
            // The price axis sits on the right-hand side of this chart.
            priceBadge.style.left = (box.width - s.r + 3) + 'px';

            dateBadge.textContent = formatDate(xa.p2d(x - s.l));
            dateBadge.style.display = 'block';
            dateBadge.style.top = (box.height - s.b + 4) + 'px';
            dateBadge.style.left = Math.max(0, x - dateBadge.offsetWidth / 2) + 'px';
        });
        gd.addEventListener('mouseleave', hide);
    }

    // Streamlit re-creates the plot element on every rerun, so keep looking
    // rather than binding once at load.
    setInterval(function () {
        var plots = doc.querySelectorAll('.js-plotly-plot');
        for (var i = 0; i < plots.length; i++) { attach(plots[i]); }
    }, 400);
})();
</script>
""".replace("__ACCENT__", PALETTE["accent"])


def crosshair_axis_badges(intraday: bool = False):
    """Render the cursor's price/date against the chart axes, TradingView-style.

    Must be a component (not st.markdown) because Streamlit strips <script>
    from unsafe_allow_html markup.
    """
    import streamlit.components.v1 as components
    html = _AXIS_BADGE_JS.replace("__INTRADAY__", "true" if intraday else "false")
    components.html(html, height=0)

def page_header(asset: str, source: str) -> str:
    src = source if source else "—"
    return f"""
<div class="at-header">
  <div class="eyebrow">PRO TRADER TERMİNALİ</div>
  <div class="asset">{asset}</div>
  <span class="srcbadge">VERİ KAYNAĞI · <b>{src}</b></span>
</div>"""

def verdict_card(sig) -> str:
    color, glyph = VERDICT_STYLE.get(sig.verdict, (PALETTE["wait"], "◆"))
    conf = max(0, min(100, sig.confidence))
    if sig.raw_verdict != sig.verdict:
        chip = f'<div class="at-chip hot">⏳ Ham sinyal: {sig.raw_verdict} — onay bekliyor</div>'
    elif sig.bars_held > 1:
        chip = f'<div class="at-chip">{sig.bars_held} bardır stabil</div>'
    else:
        chip = ""
    return f"""
<div class="at-verdict" style="--vc:{color};">
  <div class="glyph">{glyph}</div>
  <div class="word">{sig.verdict}</div>
  <div class="meter"><div style="width:{conf:.0f}%"></div></div>
  <div class="sub">GÜVEN %{conf:.0f} · SKOR {sig.final_score:+.0f}</div>
  {chip}
</div>"""

def trade_plan_card(sig) -> str:
    color = verdict_color(sig.verdict)
    direction = "LONG İŞLEM PLANI" if "AL" in sig.verdict else "SHORT İŞLEM PLANI"
    return f"""
<div class="at-plan" style="--vc:{color};">
  <div class="title">{direction}</div>
  <div class="row"><span class="k">Giriş</span><span class="v">${sig.entry_price:,.2f}</span></div>
  <div class="row"><span class="k">Stop Loss</span><span class="v down">${sig.stop_loss:,.2f} (−%{sig.risk_amount_pct:.1f})</span></div>
  <div class="row"><span class="k">TP1 (1.5:1)</span><span class="v up">${sig.take_profit_1:,.2f}</span></div>
  <div class="row"><span class="k">TP2 (3:1)</span><span class="v up">${sig.take_profit_2:,.2f}</span></div>
  <div class="row"><span class="k">Risk / Ödül</span><span class="v">1:{sig.risk_reward:.1f}</span></div>
</div>"""

def exit_warning_card(sig) -> str:
    """SAT verdict card. Short trades are backtest-falsified (12 assets,
    avg -21%, 10/12 losers) — SAT's validated meaning is exit/stay out."""
    color = PALETTE["down"]
    return f"""
<div class="at-plan" style="--vc:{color};">
  <div class="title">POZİSYONDAN ÇIK / UZAK DUR</div>
  <div class="row"><span class="k">Sinyal</span><span class="v down">{sig.verdict}</span></div>
  <div class="row"><span class="k">Anlam</span><span class="v">Long pozisyonları kapat/azalt, yeni alım yapma</span></div>
  <div style="font-size:12px; color:var(--muted); margin-top:8px;">
    ⚠️ SAT sinyali short (açığa satış) önerisi DEĞİLDİR — short işlemler
    backtest'te 12 varlıkta ortalama −%21 ile doğrulanamadı.
  </div>
</div>"""

def dimension_bar(label: str, value: float, delay_idx: int = 0) -> str:
    v = max(-100.0, min(100.0, float(value)))
    color = PALETTE["up"] if v > 20 else (PALETTE["down"] if v < -20 else PALETTE["muted"])
    half_pct = abs(v) / 2  # each side of center is 50% of the track
    side = "pos" if v >= 0 else "neg"
    return f"""
<div class="at-dim" style="animation-delay:{delay_idx * 0.05:.2f}s">
  <div class="lbl"><span>{label}</span><b style="color:{color}">{v:+.0f}</b></div>
  <div class="track"><div class="fill {side}" style="width:{max(half_pct, 1.5):.1f}%"></div></div>
</div>"""

def analysis_card(title: str, rows_html: str, direction: str = "NEUTRAL",
                  desc: str = "", badges: str = "") -> str:
    color = tone_color(direction)
    desc_html = f'<div class="desc">{desc}</div>' if desc else ""
    return f"""
<div class="at-card" style="--vc:{color};">
  <div class="title">{title}{badges}</div>
  {rows_html}
  {desc_html}
</div>"""

def tf_row(tf_label: str, verdict: str, confidence: float) -> str:
    color = verdict_color(verdict)
    return f"""
<div class="at-tfrow">
  <span class="tf">{tf_label}</span>
  <span class="vd" style="color:{color}">{verdict} <span style="color:{PALETTE['muted']}; font-family:var(--mono); font-size:11px; font-weight:400">%{confidence:.0f}</span></span>
</div>"""

def reason_line(text: str) -> str:
    return f'<div class="at-reason">{text}</div>'

def sidebar_signal(tf_label: str, sig) -> str:
    color = verdict_color(sig.verdict)
    return f"""
<div class="at-sig">
  <div class="tf">{tf_label}</div>
  <div class="vd" style="color:{color}">{sig.emoji} {sig.verdict}</div>
</div>"""
