import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import yfinance as yf
import time
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

# ==========================================
# 🛠️ KULLANICI AYARLARI
# ==========================================
DEFAULT_TOKEN = "BURAYA_TOKEN_YAPIŞTIR"
DEFAULT_CHAT_ID = "BURAYA_CHAT_ID_YAZ"
# ==========================================

st.set_page_config(layout="wide", page_title="Pro Trader V33 (4H Loop)")

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem !important; 
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 { font-size: 2rem !important; margin-bottom: 0rem; }
        .stMarkdown p { font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# --- EĞİTİM SÖZLÜĞÜ ---
PATTERN_INFO = {
    "İkili Dip (W)": "📉 **W Formasyonu:** Yükseliş.",
    "İkili Tepe (M)": "📈 **M Formasyonu:** Düşüş.",
    "Doji": "⚠️ **Doji:** Kararsızlık.",
    "Hammer": "🔨 **Çekiç:** Dönüş.",
    "Yutan Boğa": "🚀 **Yutan Boğa:** Alım."
}

# --- VERİ MOTORLARI ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data_smart(source, symbol, interval):
    if symbol in ["XAU_GOLD", "EURUSD=X"]:
        targets = ["XAUUSD=X", "GC=F"] if symbol == "XAU_GOLD" else ["EURUSD=X"]
        for t in targets:
            df = fetch_yahoo_safe(t, interval)
            if df is not None: return process_data(df, f"Yahoo ({t})")
        return None, "Veri Yok"

    if source == "Binance":
        urls = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]
        s_bin = symbol.replace("-", "").replace("USD", "USDT")
        for url in urls:
            try:
                r = requests.get(url, params={"symbol": s_bin, "interval": interval, "limit": 1000}, timeout=2)
                if r.status_code == 200:
                    df = pd.DataFrame(r.json(), columns=["OpT", "Open", "High", "Low", "Close", "Vol", "x", "x", "x", "x", "x", "x"])
                    df["Date"] = pd.to_datetime(df["OpT"], unit='ms')
                    df.set_index("Date", inplace=True)
                    return process_data(df[["Open", "High", "Low", "Close", "Volume"]].astype(float), "Binance (Hızlı)")
            except: continue

    elif source == "OKX":
        try:
            s_okx = symbol.replace("USD", "USDT")
            omap = {"4h": "4H", "1d": "1D", "1wk": "1W"}
            r = requests.get("https://www.okx.com/api/v5/market/candles", params={"instId": s_okx, "bar": omap.get(interval,"1D"), "limit": 300}, timeout=3)
            data = r.json()
            if data['code'] == '0':
                df = pd.DataFrame(data['data'], columns=["ts", "Open", "High", "Low", "Close", "Vol", "x", "x", "x"])
                df["Date"] = pd.to_datetime(df["ts"], unit='ms')
                df.set_index("Date", inplace=True)
                df = df.sort_index()
                return process_data(df[["Open", "High", "Low", "Close", "Volume"]].astype(float), "OKX")
        except: pass

    df = fetch_yahoo_safe(symbol, interval)
    if df is not None: return process_data(df, "Yahoo (Yedek)")
    return None, "Veri Yok"

def fetch_yahoo_safe(symbol, interval):
    try:
        p = "max" if interval in ["1d", "1wk"] else "59d"
        i = "1h" if interval == "4h" else ("1d" if interval == "1d" else "1wk")
        df = yf.download(symbol, period=p, interval=i, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if interval == "4h":
            agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('4h', closed='left', label='left').agg(agg).dropna()
        return df
    except: return None

def process_data(df, src):
    if df is not None and len(df) > 10:
        df['RSI'] = df.ta.rsi(length=14)
        df['EMA_50'] = df.ta.ema(length=50)
        bb = df.ta.bbands(length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={df.columns[-5]: 'BB_Lower', df.columns[-3]: 'BB_Upper'}, inplace=True)
        return df, src
    return None, "Hata"

# --- DESTEK / DİRENÇ ---
def calculate_sr(df, timeframe):
    supports, resistances = [], []
    n = 5 if timeframe == "4h" else 15
    work_df = df.tail(300)
    for i in range(n, len(work_df)-n):
        l = work_df['Low'].iloc[i]
        h = work_df['High'].iloc[i]
        if l == work_df['Low'].iloc[i-n:i+n+1].min(): supports.append(l)
        if h == work_df['High'].iloc[i-n:i+n+1].max(): resistances.append(h)
    return sorted(list(set([round(x,2) for x in supports]))), sorted(list(set([round(x,2) for x in resistances])))

# --- ORACLE SİNYAL ---
def calculate_oracle_signal_fixed(df, supports, resistances):
    if df is None: return "Veri Yok", "gray", ""
    
    last = df.iloc[-1]
    rsi = last['RSI']
    price = last['Close']
    bb_lower = last['BB_Lower']
    bb_upper = last['BB_Upper']
    
    target_msg = ""
    
    if rsi < 45:
        status = "AL (UCUZ)"
        color = "blue"
        if rsi < 30:
            status = "GÜÇLÜ AL (DİP)"
            color = "green"
        
        if price < bb_lower:
            lower_supports = [s for s in supports if s < price]
            if lower_supports:
                next_sup = max(lower_supports)
                diff = ((price - next_sup) / price) * 100
                target_msg = f"📉 Sonraki Destek: {next_sup:,.2f} (-{diff:.1f}%)"
            else:
                target_msg = "Dip belirsiz (Tarihi Dip)"
        else:
            diff = ((price - bb_lower) / price) * 100
            target_msg = f"📉 Bollinger Dibi: {bb_lower:,.2f} (-{diff:.1f}%)"

    elif rsi > 55:
        status = "SAT (PAHALI)"
        color = "orange"
        if rsi > 70:
            status = "GÜÇLÜ SAT (TEPE)"
            color = "red"
            
        if price > bb_upper:
            upper_resistances = [r for r in resistances if r > price]
            if upper_resistances:
                next_res = min(upper_resistances)
                diff = ((next_res - price) / price) * 100
                target_msg = f"📈 Sonraki Direnç: {next_res:,.2f} (+{diff:.1f}%)"
            else:
                target_msg = "Tepe belirsiz (ATH)"
        else:
            diff = ((bb_upper - price) / price) * 100
            target_msg = f"📈 Bollinger Tepesi: {bb_upper:,.2f} (+{diff:.1f}%)"
    else:
        status = "NÖTR (İZLE)"
        color = "gray"
        target_msg = f"Yön Arıyor | RSI: {rsi:.0f}"
        
    return status, color, target_msg

# --- AI TAHMİN ---
def calculate_smart_prediction(df, periods=10):
    try:
        work_df = df.tail(150).copy()
        x = np.arange(len(work_df))
        y = work_df['Close'].values
        z = np.polyfit(x, y, 2) 
        p = np.poly1d(z)
        future_x = np.arange(len(work_df), len(work_df) + periods)
        predictions = p(future_x)
        last_date = work_df.index[-1]
        time_delta = work_df.index[-1] - work_df.index[-2]
        future_dates = [last_date + (time_delta * i) for i in range(1, periods + 1)]
        return future_dates, predictions
    except: return [], []

# --- TREND ÇİZGİLERİ ---
def calculate_extended_trendlines(df, extend_candles=15):
    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index
    if len(dates) > 2: delta = dates[-1] - dates[-2]
    else: return []
    last_date = dates[-1]
    future_date = last_date + (delta * extend_candles)
    max_idxs = argrelextrema(highs, np.greater, order=10)[0]
    min_idxs = argrelextrema(lows, np.less, order=10)[0]
    lines = []
    if len(max_idxs) >= 2:
        p1, p2 = max_idxs[-2], max_idxs[-1]
        if highs[p2] < highs[p1]:
            slope = (highs[p2] - highs[p1]) / (p2 - p1)
            y_extended = highs[p1] + slope * (len(df) - 1 + extend_candles - p1)
            lines.append({"x0": dates[p1], "y0": highs[p1], "x1": future_date, "y1": y_extended, "color": "red"})
    if len(min_idxs) >= 2:
        p1, p2 = min_idxs[-2], min_idxs[-1]
        if lows[p2] > lows[p1]:
            slope = (lows[p2] - lows[p1]) / (p2 - p1)
            y_extended = lows[p1] + slope * (len(df) - 1 + extend_candles - p1)
            lines.append({"x0": dates[p1], "y0": lows[p1], "x1": future_date, "y1": y_extended, "color": "green"})
    return lines

# --- FORMASYONLAR ---
def detect_patterns(df):
    patterns = []
    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    min_idxs = argrelextrema(lows, np.less, order=5)[0]
    max_idxs = argrelextrema(highs, np.greater, order=5)[0]

    if len(min_idxs) >= 2:
        i1, i2 = min_idxs[-2], min_idxs[-1]
        d1, d2 = lows[i1], lows[i2]
        if (abs(d1-d2)/d1 < 0.02) and ((i2-i1) > 5):
            neck = highs[i1:i2].max()
            patterns.append({"type": "box", "name": "İkili Dip (W)", "color": "green", "x0": dates[i1], "x1": dates[i2], "y0": min(d1,d2)*0.99, "y1": neck, "target": neck+(neck-(d1+d2)/2)})
    if len(max_idxs) >= 2:
        i1, i2 = max_idxs[-2], max_idxs[-1]
        t1, t2 = highs[i1], highs[i2]
        if (abs(t1-t2)/t1 < 0.02) and ((i2-i1) > 5):
            neck = lows[i1:i2].min()
            patterns.append({"type": "box", "name": "İkili Tepe (M)", "color": "red", "x0": dates[i1], "x1": dates[i2], "y0": neck, "y1": max(t1,t2)*1.01, "target": neck-((t1+t2)/2-neck)})
    for i in range(-5, 0):
        idx = i
        O, H, L, C = opens[idx], highs[idx], lows[idx], closes[idx]
        body = abs(C-O)
        if body < np.mean(np.abs(closes-opens))*0.1:
            patterns.append({"type": "icon", "name": "Doji", "color": "yellow", "x": dates[idx], "y": H, "msg": "⚠️", "anchor": "bottom"})
        if (min(O,C)-L) > 2*body and (H-max(O,C)) < 0.5*body:
            patterns.append({"type": "icon", "name": "Hammer", "color": "lime", "x": dates[idx], "y": L, "msg": "🔨", "anchor": "top"})
        if i < -1:
            if (closes[idx] > opens[idx]) and (closes[idx-1] < opens[idx-1]) and (closes[idx] > opens[idx-1]) and (opens[idx] < closes[idx-1]):
                 patterns.append({"type": "icon", "name": "Yutan Boğa", "color": "cyan", "x": dates[idx], "y": lows[idx], "msg": "🚀", "anchor": "top"})
    return patterns

def send_tg(token, chat_id, msg):
    try: requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- ARAYÜZ ---
st.sidebar.header("⚙️ Kontrol Paneli")
src_pref = st.sidebar.radio("📡 Kaynak:", ["Binance", "OKX", "Yahoo Finance"])
coin_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD", "AVAX": "AVAX-USD", "DOGE": "DOGE-USD", "PEPE": "PEPE-USD", "ALTIN": "XAU_GOLD", "EUR/USD": "EURUSD=X"}
sel_c = st.sidebar.selectbox("Enstrüman:", list(coin_map.keys()))
symbol = coin_map[sel_c]

st.sidebar.divider()
show_ai = st.sidebar.checkbox("🤖 AI Trend", value=True)
show_pred = st.sidebar.checkbox("🔮 AI Tahmin", value=True)
st.sidebar.subheader("🔍 Filtreler")
show_all_pats = st.sidebar.checkbox("Hepsini Aç/Kapat", value=True)
f_wm = st.sidebar.checkbox("- W ve M", value=True)
f_candle = st.sidebar.checkbox("- Mumlar", value=True)

intervals = {"4h": "4 Saatlik", "1d": "Günlük", "1wk": "Haftalık"}
results = {}
active_src = ""

# --- ANALİZ DÖNGÜSÜ ---
for tf, label in intervals.items():
    df, src = fetch_data_smart(src_pref, symbol, tf)
    if tf == "1d": active_src = src
    results[tf] = df
    
    if df is not None:
        s_list, r_list = calculate_sr(df, tf)
        status, color, target_msg = calculate_oracle_signal_fixed(df, s_list, r_list)
        
        st.sidebar.markdown(f"---")
        st.sidebar.markdown(f"### {label}")
        st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{status}</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"{target_msg}")
    else: 
        st.sidebar.warning(f"{label}: Bekleniyor...")

st.title(f"📈 {sel_c} V33 (4H Loop)")
c = "green" if "Binance" in active_src else ("blue" if "OKX" in active_src else "orange")
st.markdown(f"**Veri Kaynağı:** <span style='color:{c}; font-weight:bold'>{active_src}</span>", unsafe_allow_html=True)

view_tf = st.selectbox("Periyot:", list(intervals.keys()), format_func=lambda x: intervals[x])
df_view = results[view_tf]

if df_view is not None:
    curr = df_view['Close'].iloc[-1]
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='Fiyat'))
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'))
    fig.add_hline(y=curr, line_dash="dot", line_color="cyan", annotation_text=f" {curr:,.2f}", annotation_position="right")

    if show_pred:
        f_dates, f_prices = calculate_smart_prediction(df_view)
        if len(f_dates) > 0:
            line_x = [df_view.index[-1]] + f_dates
            line_y = [df_view['Close'].iloc[-1]] + list(f_prices)
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='yellow', width=2, dash='dash'), name='AI Tahmini'))
            fig.add_annotation(x=f_dates[-1], y=f_prices[-1], text=f"🔮 Hedef: {f_prices[-1]:,.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30, bgcolor="yellow", bordercolor="black", font=dict(color="black"))

    if show_ai:
        lines = calculate_extended_trendlines(df_view)
        for l in lines:
            fig.add_shape(type="line", x0=l['x0'], y0=l['y0'], x1=l['x1'], y1=l['y1'], line=dict(color=l['color'], width=2, dash='dot'))

    if show_all_pats:
        items = detect_patterns(df_view)
        for i in items:
            draw = False
            if i['type'] == 'box' and f_wm: draw = True
            if i['type'] == 'icon' and f_candle: draw = True
            if draw:
                if i['type'] == 'box':
                    fig.add_shape(type="rect", x0=i['x0'], y0=i['y0'], x1=i['x1'], y1=i['y1'], line=dict(color=i['color'], width=2), fillcolor=i['color'], opacity=0.15)
                    fig.add_hline(y=i['target'], line_dash="dashdot", line_color="magenta", annotation_text="HEDEF")
                elif i['type'] == 'icon':
                    fig.add_annotation(x=i['x'], y=i['y'], text=i['msg'], showarrow=False, yshift=15 if i.get('anchor')=='bottom' else -15)

    s_list, r_list = calculate_sr(df_view, view_tf)
    for s in [x for x in s_list if x < curr][-3:]:
        fig.add_hline(y=s, line_dash="dash", line_color="#00FF00", annotation_text=f"Dst: {s}")
    for r in [x for x in r_list if x > curr][:3]:
        fig.add_hline(y=r, line_dash="dash", line_color="#FF0000", annotation_text=f"Dir: {r}")

    zoom_count = 30 if view_tf == "1wk" else (60 if view_tf == "1d" else 80)
    if len(df_view) > zoom_count:
        visible_df = df_view.tail(zoom_count)
        zoom_start = visible_df.index[0]
        y_min = visible_df['Low'].min() * 0.98
        y_max = visible_df['High'].max() * 1.02
    else:
        zoom_start = df_view.index[0]
        y_min = df_view['Low'].min()
        y_max = df_view['High'].max()

    gap_multiplier = 2 if view_tf == "1wk" else 5
    if len(df_view) > 2:
        delta = df_view.index[-1] - df_view.index[-2]
        zoom_end = df_view.index[-1] + (delta * gap_multiplier)
    else:
        zoom_end = df_view.index[-1]

    y_type = "log" if view_tf == "1wk" else "linear"

    config = {'scrollZoom': True, 'displayModeBar': True, 'editable': True, 'modeBarButtons_add': ['drawline', 'drawrect', 'eraseshape']}
    
    fig.update_layout(
        height=900, 
        template="plotly_dark", 
        xaxis_rangeslider_visible=False, 
        dragmode="pan", 
        yaxis=dict(side="right", fixedrange=False, type=y_type, range=[y_min, y_max] if y_type == "linear" else None),
        xaxis=dict(range=[zoom_start, zoom_end]),
        margin=dict(l=10, r=50, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config=config)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("### 🧠 Tespitler")
        if show_all_pats and items:
            visible_names = []
            for item in items:
                if (item['type'] == 'box' and f_wm) or (item['type'] == 'icon' and f_candle): visible_names.append(item['name'])
            if visible_names:
                for p in list(set(visible_names)): st.write(PATTERN_INFO.get(p, p))
            else: st.write("Seçili filtrede formasyon yok.")
        else: st.write("Formasyon gösterimi kapalı.")
        
    with col2:
        trend = "YÜKSELİŞ" if curr > df_view['EMA_50'].iloc[-1] else "DÜŞÜŞ"
        pred_dir = "YUKARI ↗️" if show_pred and len(f_prices)>0 and f_prices[-1] > curr else "AŞAĞI ↘️"
        st.metric("Trend", trend)
        st.metric("AI Tahmin", pred_dir)
        st.metric("RSI", f"{df_view['RSI'].iloc[-1]:.1f}")

else: st.error("Veri Alınamadı.")

# --- OTOMATİK BOT DÖNGÜSÜ (4 SAATTE BİR) ---
if st.session_state.get('auto_mode', False) or st.sidebar.checkbox("Otomatik Bot", key='auto_mode'):
    tg_token = st.sidebar.text_input("Bot Token", value=DEFAULT_TOKEN, type="password")
    tg_chat = st.sidebar.text_input("Chat ID", value=DEFAULT_CHAT_ID)
    
    msg = ""
    for tf, res in results.items():
        if res is not None:
            # Her periyot için destekleri yeniden hesapla
            s_l, r_l = calculate_sr(res, tf)
            stat, _, target = calculate_oracle_signal_fixed(res, s_l, r_l)
            if "GÜÇLÜ" in stat or "AL" in stat:
                msg += f"\n⏰ {tf}: {stat} | {target}"
    
    if msg and tg_token and tg_chat:
        full_msg = f"🚨 **{sel_c} OTOMATİK ANALİZ** 🚨\n{msg}\nFiyat: {curr:.2f}"
        # Son mesajla aynı değilse gönder
        if 'last_msg' not in st.session_state or st.session_state['last_msg'] != full_msg:
            send_tg(tg_token, tg_chat, full_msg)
            st.session_state['last_msg'] = full_msg
            st.toast("Bildirim Gönderildi!")
    
    # --- İŞTE BURADA 4 SAAT BEKLİYOR ---
    time.sleep(14400) 
    st.rerun()
