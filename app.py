import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import yfinance as yf
import time
import numpy as np
import json
import os
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

# ==========================================
# 🛠️ KULLANICI AYARLARI
# ==========================================
DEFAULT_TOKEN = "BURAYA_TOKEN_YAPIŞTIR"
DEFAULT_CHAT_ID = "BURAYA_CHAT_ID_YAZ"
PORTFOLIO_FILE = "portfolio.json"
# ==========================================

st.set_page_config(layout="wide", page_title="Pro Trader V43 (Cloud & Money)")

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { font-size: 2rem !important; margin-bottom: 0rem; }
        .stMarkdown p { font-size: 14px; }
        .profit { color: #4CAF50; font-weight: bold; }
        .loss { color: #FF5252; font-weight: bold; }
        /* Tablo Yazılarını Ortalama */
        .stDataFrame { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- VERİTABANI ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except: return []
    return []

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f)

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = load_portfolio()

# --- COIN HARİTASI ---
COIN_MAP = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", 
    "Solana (SOL)": "SOL-USD", "Ripple (XRP)": "XRP-USD", 
    "Avax (AVAX)": "AVAX-USD", "Dogecoin (DOGE)": "DOGE-USD", 
    "Pepe": "PEPE-USD", "ONS ALTIN": "XAU_GOLD", "EUR/USD": "EURUSD=X"
}

# --- EĞİTİM SÖZLÜĞÜ ---
PATTERN_INFO = {
    "İkili Dip (W)": "📉 **W Formasyonu:** Yükseliş sinyali.",
    "İkili Tepe (M)": "📈 **M Formasyonu:** Düşüş sinyali.",
    "Doji": "⚠️ **Doji:** Kararsızlık.",
    "Hammer": "🔨 **Çekiç:** Dipten dönüş.",
    "Yutan Boğa": "🚀 **Yutan Boğa:** Güçlü alım."
}

# --- VERİ MOTORLARI (STABIL) ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance_simple(symbol, interval, limit=500):
    s_bin = symbol.replace("-", "").replace("USD", "USDT")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": s_bin, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=["OpT", "Open", "High", "Low", "Close", "Vol", "x", "x", "x", "x", "x", "x"])
            df["Date"] = pd.to_datetime(df["OpT"], unit='ms')
            df.set_index("Date", inplace=True)
            return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except: return None
    return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_okx_simple(symbol, interval, limit=300):
    s_okx = symbol.replace("USD", "USDT")
    omap = {"4h": "4H", "1d": "1D", "1wk": "1W"}
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": s_okx, "bar": omap.get(interval, "1D"), "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data['code'] == '0':
            df = pd.DataFrame(data['data'], columns=["ts", "Open", "High", "Low", "Close", "Vol", "x", "x", "x"])
            df["Date"] = pd.to_datetime(df["ts"], unit='ms')
            df.set_index("Date", inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df.sort_index()
    except: return None
    return None

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

def get_market_data(source_pref, symbol, interval):
    if symbol == "XAU_GOLD":
        for t in ["XAUUSD=X", "GC=F"]:
            df = fetch_yahoo_safe(t, interval)
            if df is not None: return process_data(df, f"Yahoo ({t})")
        return None, "Veri Yok"
    if symbol == "EURUSD=X":
        return process_data(fetch_yahoo_safe("EURUSD=X", interval), "Yahoo (Forex)")

    df = None
    src_name = ""
    if source_pref == "Binance":
        df = fetch_binance_simple(symbol, interval)
        src_name = "Binance" if df is not None else ""
    elif source_pref == "OKX":
        df = fetch_okx_simple(symbol, interval)
        src_name = "OKX" if df is not None else ""
    
    if df is None:
        df = fetch_yahoo_safe(symbol, interval)
        src_name = "Yahoo (Yedek)"

    if df is None or df.empty: return None, "Veri Alınamadı"
    return process_data(df, src_name)

def process_data(df, src):
    if df is not None and len(df) > 10:
        df['RSI'] = df.ta.rsi(length=14)
        df['EMA_50'] = df.ta.ema(length=50)
        # BULUT İÇİN EMA 20 ve 50
        df['EMA_20'] = df.ta.ema(length=20)
        bb = df.ta.bbands(length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={df.columns[-5]: 'BB_Lower', df.columns[-3]: 'BB_Upper'}, inplace=True)
        df['ATR'] = df.ta.atr(length=14)
        return df, src
    return None, "Hata"

# --- CÜZDAN CANLI FİYAT ---
@st.cache_data(ttl=30, show_spinner=False)
def get_live_price_for_portfolio(coin_name):
    try:
        ticker_symbol = COIN_MAP.get(coin_name)
        if ticker_symbol == "XAU_GOLD": ticker_symbol = "XAUUSD=X"
        if not ticker_symbol: return 0
        ticker = yf.Ticker(ticker_symbol)
        return ticker.fast_info['last_price']
    except: return 0

# --- HESAPLAMALAR ---
def calculate_trade_setup(df, signal_type):
    if df is None: return None
    last = df.iloc[-1]
    price = last['Close']
    atr = last['ATR']
    if np.isnan(atr): atr = price * 0.02
    
    setup = {}
    if "AL" in signal_type:
        setup['type'] = "LONG (YÜKSELİŞ)"
        setup['entry'] = price
        setup['sl'] = price - (atr * 1.5)
        setup['tp'] = price + (atr * 2.5)
    elif "SAT" in signal_type:
        setup['type'] = "SHORT (DÜŞÜŞ)"
        setup['entry'] = price
        setup['sl'] = price + (atr * 1.5)
        setup['tp'] = price - (atr * 2.5)
    else: return None
    return setup

def calculate_setup_dynamic(entry_price, atr, direction):
    if np.isnan(atr) or atr == 0: atr = entry_price * 0.02
    setup = {}
    if direction == "LONG":
        setup['sl'] = entry_price - (atr * 1.5)
        setup['tp'] = entry_price + (atr * 2.5)
    else:
        setup['sl'] = entry_price + (atr * 1.5)
        setup['tp'] = entry_price - (atr * 2.5)
    return setup

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

def calculate_oracle_signal_fixed(df, supports, resistances):
    if df is None: return "Veri Yok", "gray", ""
    last = df.iloc[-1]
    rsi = last['RSI']
    price = last['Close']
    bb_lower = last['BB_Lower']
    bb_upper = last['BB_Upper']
    target_msg = ""
    
    if rsi < 45:
        status, color = "AL (UCUZ)", "blue"
        if rsi < 30: status, color = "GÜÇLÜ AL (DİP)", "green"
        if price < bb_lower:
            low_s = [s for s in supports if s < price]
            target_msg = f"📉 Hedef: {max(low_s):,.2f}" if low_s else "Dip Belirsiz"
        else: target_msg = f"📉 Hedef: {bb_lower:,.2f}"
    elif rsi > 55:
        status, color = "SAT (PAHALI)", "orange"
        if rsi > 70: status, color = "GÜÇLÜ SAT", "red"
        if price > bb_upper:
            up_r = [r for r in resistances if r > price]
            target_msg = f"📈 Hedef: {min(up_r):,.2f}" if up_r else "ATH Belirsiz"
        else: target_msg = f"📈 Hedef: {bb_upper:,.2f}"
    else:
        status, color, target_msg = "NÖTR", "gray", f"RSI: {rsi:.0f}"
    return status, color, target_msg

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
            y_ext = highs[p1] + slope * (len(df)-1+extend_candles-p1)
            lines.append({"x0": dates[p1], "y0": highs[p1], "x1": future_date, "y1": y_ext, "color": "red"})
    if len(min_idxs) >= 2:
        p1, p2 = min_idxs[-2], min_idxs[-1]
        if lows[p2] > lows[p1]:
            slope = (lows[p2] - lows[p1]) / (p2 - p1)
            y_ext = lows[p1] + slope * (len(df)-1+extend_candles-p1)
            lines.append({"x0": dates[p1], "y0": lows[p1], "x1": future_date, "y1": y_ext, "color": "green"})
    return lines

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
sel_c = st.sidebar.selectbox("Enstrüman:", list(COIN_MAP.keys()))
symbol = COIN_MAP[sel_c]

st.sidebar.divider()
show_cloud = st.sidebar.checkbox("☁️ Destek/Direnç Bulutu", value=True)
show_ai = st.sidebar.checkbox("🤖 AI Trend", value=True)
show_pred = st.sidebar.checkbox("🔮 AI Tahmin", value=True)
st.sidebar.subheader("🔍 Filtreler")
show_all_pats = st.sidebar.checkbox("Hepsini Aç/Kapat", value=True)
f_wm = st.sidebar.checkbox("- W ve M", value=True)
f_candle = st.sidebar.checkbox("- Mumlar", value=True)

tg_token = st.sidebar.text_input("Bot Token", value=DEFAULT_TOKEN, type="password")
tg_chat = st.sidebar.text_input("Chat ID", value=DEFAULT_CHAT_ID)
auto = st.sidebar.checkbox("Otomatik Bot")

intervals = {"4h": "4 Saatlik", "1d": "Günlük", "1wk": "Haftalık"}
results = {}
active_src = ""

# --- ANALİZ DÖNGÜSÜ ---
for tf, label in intervals.items():
    df, src = get_market_data(src_pref, symbol, tf)
    if tf == "1d": active_src = src
    results[tf] = df
    
    if df is not None:
        s_list, r_list = calculate_sr(df, tf)
        status, color, target_msg = calculate_oracle_signal_fixed(df, s_list, r_list)
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {label}")
        st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{status}</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"{target_msg}")
    else: 
        st.sidebar.warning(f"{label}: Bekleniyor...")

st.title(f"📈 {sel_c} V43 (Cloud & Money)")
c = "green" if "Binance" in active_src else ("blue" if "OKX" in active_src else "orange")
st.markdown(f"**Veri Kaynağı:** <span style='color:{c}; font-weight:bold'>{active_src}</span>", unsafe_allow_html=True)

view_tf = st.selectbox("Periyot:", list(intervals.keys()), format_func=lambda x: intervals[x])
df_view = results[view_tf]

if df_view is not None:
    curr = df_view['Close'].iloc[-1]
    current_atr = df_view.iloc[-1]['ATR'] if 'ATR' in df_view.columns else curr * 0.02
    
    fig = go.Figure()
    
    # --- 1. BULUT (EMA CLOUD) ---
    if show_cloud:
        # Bulut rengini belirle: EMA20 > EMA50 ise Yeşil, değilse Kırmızı
        # Ancak Plotly'de tek trace ile renk değişimi zordur, o yüzden yarı saydam turuncu/gri yapalım
        # ki genel "Dinamik Alan" belli olsun.
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view['EMA_20'],
            line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
            name='EMA 20'
        ))
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view['EMA_50'],
            fill='tonexty', # Bir önceki trace ile arasını doldur
            fillcolor='rgba(255, 165, 0, 0.2)', # Yarı saydam turuncu bulut
            line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
            name='EMA 50 (Bulut Altı)'
        ))

    fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='Fiyat'))
    fig.add_hline(y=curr, line_dash="dot", line_color="cyan", annotation_text=f" {curr:,.2f}", annotation_position="right")

    if show_pred:
        f_dates, f_prices = calculate_smart_prediction(df_view)
        if len(f_dates) > 0:
            fig.add_trace(go.Scatter(x=[df_view.index[-1]]+f_dates, y=[df_view['Close'].iloc[-1]]+list(f_prices), mode='lines', line=dict(color='yellow', width=2, dash='dash'), name='AI Tahmini'))

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
        height=900, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode="pan", 
        yaxis=dict(side="right", fixedrange=False, type=y_type, range=[y_min, y_max] if y_type == "linear" else None),
        xaxis=dict(range=[zoom_start, zoom_end]),
        margin=dict(l=10, r=50, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config=config)

    # --- ANALİZ VE YÖNETİM ---
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.info("### 🧠 Tespitler")
        if show_all_pats and items:
            visible_names = []
            for item in items:
                if (item['type'] == 'box' and f_wm) or (item['type'] == 'icon' and f_candle): visible_names.append(item['name'])
            if visible_names:
                for p in list(set(visible_names)): st.write(PATTERN_INFO.get(p, p))
            else: st.write("Filtreli formasyon yok.")
        else: st.write("Formasyonlar kapalı.")
    
    with col2:
        st.warning("### 📊 Piyasa Özeti")
        trend = "YÜKSELİŞ" if curr > df_view['EMA_50'].iloc[-1] else "DÜŞÜŞ"
        pred_dir = "YUKARI ↗️" if show_pred and len(f_prices)>0 and f_prices[-1] > curr else "AŞAĞI ↘️"
        st.metric("Trend (EMA50)", trend)
        st.metric("Tahmin", pred_dir)
        st.metric("RSI", f"{df_view['RSI'].iloc[-1]:.1f}")

    with col3:
        st.success("### 🎯 AI Stratejisi")
        signal_status, _, _ = calculate_oracle_signal_fixed(df_view, s_list, r_list)
        
        setup = calculate_trade_setup(df_view, "AL" if "AL" in signal_status else ("SAT" if "SAT" in signal_status else "NÖTR"))
        if setup:
            st.markdown(f"#### ✅ {setup['type']}")
            st.write(f"**Giriş:** ${setup['entry']:,.2f}")
            st.write(f"**🛑 Stop:** ${setup['sl']:,.2f}")
            st.write(f"**🎯 TP:** ${setup['tp']:,.2f}")
        else:
            st.write("Nötr Bölge. Bekle.")

# --- CÜZDAN (DÜZENLENEBİLİR & GELİŞMİŞ SATIŞ) ---
    st.divider()
    st.header("💼 Portföy ve Risk Yönetimi")
    col_risk, col_wallet = st.columns([1, 2])
    
    with col_risk:
        st.subheader("🧮 Planlama")
        entry_price = st.number_input("Giriş Fiyatı ($)", value=float(curr), step=0.01, format="%.2f")
        investment = st.number_input("Yatırım Tutarı ($)", value=1000.0, step=100.0)
        plan_dir = st.radio("Yön:", ["LONG", "SHORT"], horizontal=True)
        
        plan_setup = calculate_setup_dynamic(entry_price, current_atr, "LONG" if "LONG" in plan_dir else "SHORT")
        
        risk_usd = abs(entry_price - plan_setup['sl']) * (investment / entry_price)
        reward_usd = abs(plan_setup['tp'] - entry_price) * (investment / entry_price)
        
        st.markdown(f"**🛑 Stop:** ${plan_setup['sl']:,.2f} (Risk: -${risk_usd:.1f})")
        st.markdown(f"**🎯 TP:** ${plan_setup['tp']:,.2f} (Kar: +${reward_usd:.1f})")
        
        if st.button("➕ Cüzdana Ekle"):
            new_trade = {
                "Coin": sel_c,
                "Giriş": entry_price,
                "Adet": investment / entry_price,
                "Yatırım": investment,
                "Realized": 0.0,
                "Tarih": time.strftime("%Y-%m-%d")
            }
            st.session_state['portfolio'].append(new_trade)
            save_portfolio(st.session_state['portfolio'])
            st.success("Kaydedildi!")
            st.rerun()

    with col_wallet:
        st.subheader("💰 Varlıklarım")
        
        if st.session_state['portfolio']:
            # Veri standardizasyonu (Eski verilerde eksik key varsa tamamla)
            for item in st.session_state['portfolio']:
                if 'Realized' not in item: item['Realized'] = 0.0

            portfolio_df = pd.DataFrame(st.session_state['portfolio'])

            # --- 1. GELİŞMİŞ SATIŞ PANELİ ---
            with st.expander("💸 Kar Al / Kısmi Satış Yap"):
                # Coin Seçimi
                p_coins = [c for c in portfolio_df['Coin'].unique().tolist() if portfolio_df[portfolio_df['Coin']==c]['Adet'].sum() > 0]
                
                if p_coins:
                    s_col1, s_col2, s_col3 = st.columns([1.5, 1.5, 1])
                    sell_coin = s_col1.selectbox("Coin", p_coins)
                    
                    # Seçilen coin verisi
                    coin_data = next((item for item in st.session_state['portfolio'] if item["Coin"] == sell_coin), None)
                    current_qty = coin_data['Adet']
                    
                    # Satış Fiyatı (Manuel Düzenlenebilir)
                    live_p_sell = curr if sell_coin == sel_c else get_live_price_for_portfolio(sell_coin)
                    if live_p_sell == 0: live_p_sell = coin_data['Giriş']
                    
                    manual_sell_price = s_col1.number_input("Satış Fiyatı ($)", value=float(live_p_sell), format="%.4f")

                    # Satış Yöntemi (Adet vs Yüzde)
                    sell_method = s_col2.radio("Satış Tipi", ["Adet Gir", "Yüzde (%) Seç"], horizontal=True)
                    
                    sell_qty = 0.0
                    if sell_method == "Adet Gir":
                        sell_qty = s_col2.number_input("Miktar (Adet)", min_value=0.0, max_value=float(current_qty), step=0.01)
                    else:
                        sell_pct = s_col2.slider("Satış Yüzdesi (%)", 0, 100, 50)
                        sell_qty = current_qty * (sell_pct / 100)
                        s_col2.caption(f"Denk gelen adet: {sell_qty:.4f}")

                    # Onay Butonu ve Hesaplama
                    est_total = sell_qty * manual_sell_price
                    s_col3.write(f"**Toplam: ${est_total:,.2f}**")
                    
                    if s_col3.button("SATIŞI ONAYLA", type="primary"):
                        if sell_qty > 0:
                            cost_basis = coin_data['Giriş']
                            cost_of_sold = sell_qty * cost_basis
                            sale_value = sell_qty * manual_sell_price
                            realized_pnl = sale_value - cost_of_sold
                            
                            # State Güncelle
                            for item in st.session_state['portfolio']:
                                if item['Coin'] == sell_coin:
                                    item['Adet'] -= sell_qty
                                    item['Yatırım'] = item['Adet'] * item['Giriş'] # Kalan yatırımı güncelle
                                    item['Realized'] += realized_pnl
                                    break
                            
                            save_portfolio(st.session_state['portfolio'])
                            st.success(f"Satış Başarılı! Kar/Zarar: ${realized_pnl:.2f}")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.warning("Satılacak varlık bulunamadı.")

            # --- 2. DÜZENLENEBİLİR TABLO (ANA VERİ) ---
            st.markdown("##### 📝 Varlık Listesi (Düzenlenebilir)")
            # Kullanıcı burada Giriş, Adet ve Realized kısımlarını elle düzeltebilir
            edited_df = st.data_editor(
                portfolio_df, 
                num_rows="dynamic", 
                use_container_width=True,
                column_config={
                    "Coin": st.column_config.TextColumn("Coin", disabled=False),
                    "Giriş": st.column_config.NumberColumn("Giriş Fiyatı", format="$%.4f"),
                    "Adet": st.column_config.NumberColumn("Adet", format="%.4f"),
                    "Yatırım": st.column_config.NumberColumn("Ana Para (Oto)", disabled=True), # Bunu otomatik hesaplayacağız
                    "Realized": st.column_config.NumberColumn("Realized ($)", format="$%.2f"),
                    "Tarih": st.column_config.TextColumn("Tarih", disabled=True),
                },
                key="editor_key"
            )

            # Değişiklik Kontrolü
            if not edited_df.equals(portfolio_df):
                # Yatırım sütununu (Adet * Giriş) olarak yeniden hesapla ki tutarsızlık olmasın
                edited_df['Yatırım'] = edited_df['Giriş'] * edited_df['Adet']
                st.session_state['portfolio'] = edited_df.to_dict('records')
                save_portfolio(st.session_state['portfolio'])
                st.rerun()

            # --- 3. ANALİZ VE TOPLAM TABLOSU (READ-ONLY) ---
            # Düzenlenebilir tabloda hesaplanan PNL görünmez, o yüzden aşağıya özet tablo koyuyoruz
            total_inv = 0
            total_val = 0
            total_realized = 0
            
            analysis_data = []
            
            for item in st.session_state['portfolio']:
                if item['Adet'] > 0:
                    live_p = curr if item['Coin'] == sel_c else get_live_price_for_portfolio(item['Coin'])
                    if live_p == 0: live_p = item['Giriş']
                    
                    val = item['Adet'] * live_p
                    unrealized = val - item['Yatırım']
                    pnl_pct = (unrealized / item['Yatırım']) * 100
                    
                    # Öneri
                    advice = "➖"
                    if pnl_pct > 10: advice = "KAR AL 💰"
                    elif pnl_pct < -5: advice = "STOP 🛑"
                    
                    analysis_data.append({
                        "Coin": item['Coin'],
                        "Fiyat": live_p,
                        "Değer": val,
                        "Kar/Zarar ($)": unrealized,
                        "Kar/Zarar (%)": f"%{pnl_pct:.2f}",
                        "Öneri": advice
                    })
                    
                    total_inv += item['Yatırım']
                    total_val += val
                
                total_realized += item['Realized']

            if analysis_data:
                st.markdown("##### 📊 Canlı Analiz (Sadece Gösterim)")
                st.dataframe(pd.DataFrame(analysis_data).set_index("Coin"), use_container_width=True)

            # METRİKLER
            unrealized_total = total_val - total_inv
            net_total = unrealized_total + total_realized
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Anlık Varlık", f"${total_val:,.2f}")
            m2.metric("Kasa (Realized)", f"${total_realized:,.2f}", delta_color="normal")
            m3.metric("NET DURUM", f"${net_total:,.2f}", delta=f"{net_total:.2f}")

            if st.button("🗑️ Tümünü Temizle"):
                st.session_state['portfolio'] = []
                save_portfolio([])
                st.rerun()
        else:
            st.info("Henüz portföy eklemediniz.")

else: st.error("Veri Alınamadı.")

# BOT
if auto or st.session_state.get('auto_mode', False):
    msg = ""
    for tf, res in results.items():
        if res is not None:
            s_l, r_l = calculate_sr(res, tf)
            stat, _, target = calculate_oracle_signal_fixed(res, s_l, r_l)
            if "GÜÇLÜ" in stat or "AL" in stat:
                msg += f"\n⏰ {tf}: {stat} | {target}"
    
    if msg and tg_token and tg_chat:
        full_msg = f"🚨 **{sel_c} BOT** 🚨\n{msg}\nFiyat: {curr:.2f}"
        if 'last_msg' not in st.session_state or st.session_state['last_msg'] != full_msg:
            send_tg(tg_token, tg_chat, full_msg)
            st.session_state['last_msg'] = full_msg
    
    time.sleep(14400) 
    st.rerun()
