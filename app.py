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
DEFAULT_TOKEN = ""
DEFAULT_CHAT_ID = ""
#try:
 #   DEFAULT_TOKEN = st.secrets["TELEGRAM_TOKEN"]
  #  DEFAULT_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
#except FileNotFoundError:
    # Eğer secrets bulunamazsa boş kalsın (veya hata versin)
 #   st.error("Lütfen .streamlit/secrets.toml dosyasını oluşturun veya Cloud Secrets ayarını yapın!")
  #  DEFAULT_TOKEN = ""
   # DEFAULT_CHAT_ID = ""
PORTFOLIO_FILE = "portfolio.json"
# ==========================================

st.set_page_config(layout="wide", page_title="Pro Trader V47 (Gold Edition)")

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { font-size: 2rem !important; margin-bottom: 0rem; }
        .stMarkdown p { font-size: 14px; }
        .profit { color: #4CAF50; font-weight: bold; }
        .loss { color: #FF5252; font-weight: bold; }
        .stDataFrame { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- VERİTABANI (YENİ YAPI) ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                # Eski yapıyı (liste) yeni yapıya (dict) dönüştür
                if isinstance(data, list):
                    return {"balance": 0.0, "positions": data}
                return data
        except: return {"balance": 1000.0, "positions": []}
    return {"balance": 1000.0, "positions": []}

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f)

if 'portfolio_data' not in st.session_state:
    st.session_state['portfolio_data'] = load_portfolio()

# --- COIN HARİTASI (GRAM ALTIN EKLENDİ) ---
COIN_MAP = {
    "Bitcoin (BTC)": "BTC-USD", 
    "Ethereum (ETH)": "ETH-USD", 
    "Solana (SOL)": "SOL-USD", 
    "Ripple (XRP)": "XRP-USD", 
    "Avax (AVAX)": "AVAX-USD", 
    "Dogecoin (DOGE)": "DOGE-USD", 
    "Pepe": "PEPE-USD", 
    "ONS ALTIN ($)": "XAU_GOLD",    # ONS Altın
    "GRAM ALTIN (TL)": "GRAM_TRY",  # Gram Altın (Hesaplamalı)
    "EUR/USD": "EURUSD=X",
    "Türk Hava Yolları (THYAO)": "THYAO.IS", # YENİ EKLENDİ
    "Pegasus (PGSUS)": "PGSUS.IS"            # YENİ EKLENDİ
}

# --- EĞİTİM SÖZLÜĞÜ ---
PATTERN_INFO = {
    "İkili Dip (W)": "📉 **W Formasyonu:** Yükseliş sinyali.",
    "İkili Tepe (M)": "📈 **M Formasyonu:** Düşüş sinyali.",
    "Doji": "⚠️ **Doji:** Kararsızlık.",
    "Hammer": "🔨 **Çekiç:** Dipten dönüş.",
    "Yutan Boğa": "🚀 **Yutan Boğa:** Güçlü alım."
}

# --- STANDART HEADERS (BOT ENGELİNİ AŞMAK İÇİN) ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- VERİ MOTORLARI (DÜZELTİLMİŞ VERSİYON) ---

# Standart Tarayıcı Kimliği (Bot Engelini Aşmak İçin)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance_simple(symbol, interval, limit=500):
    # Sembol Düzeltme
    s_bin = symbol.replace("-", "").replace("USD", "USDT")
    
    # URL Seçenekleri (Sırasıyla deneyecek)
    base_urls = [
        "https://data-api.binance.vision/api/v3/klines", # 1. En güvenlisi (Kısıtlama az)
        "https://api.binance.us/api/v3/klines",          # 2. ABD Sunucuları için
        "https://api.binance.com/api/v3/klines"          # 3. Global (Türkiye vb. için)
    ]
    
    params = {"symbol": s_bin, "interval": interval, "limit": limit}
    
    for url in base_urls:
        try:
            # Timeout süresini kısa tutuyoruz ki diğer URL'ye hızlı geçsin
            r = requests.get(url, params=params, headers=HEADERS, timeout=3)
            
            if r.status_code == 200:
                data = r.json()
                # API Hata kontrolü
                if isinstance(data, dict) and 'code' in data: continue # Hata varsa diğer URL'ye geç
                
                # Veri geldiyse DataFrame oluştur
                df = pd.DataFrame(data, columns=["OpT", "Open", "High", "Low", "Close", "Volume", "x", "x", "x", "x", "x", "x"])
                df["Date"] = pd.to_datetime(df["OpT"], unit='ms')
                df.set_index("Date", inplace=True)
                
                # Başarılı olduysa veriyi döndür ve döngüyü kır
                return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                
        except Exception as e:
            # Bu URL çalışmadıysa sessizce diğerine geç
            continue
            
    # Hiçbiri çalışmazsa None dön
    print("Binance: Tüm URL'ler başarısız oldu.")
    return None
@st.cache_data(ttl=60, show_spinner=False)
def fetch_okx_simple(symbol, interval, limit=300):
    s_okx = symbol.replace("USD", "USDT")
    omap = {"4h": "4H", "1d": "1D", "1wk": "1W"}
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": s_okx, "bar": omap.get(interval, "1D"), "limit": limit}
    
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=5)
        data = r.json()
        if data.get('code') == '0':
            # OKX Sıralaması: ts, o, h, l, c, vol, ...
            df = pd.DataFrame(data['data'], columns=["ts", "Open", "High", "Low", "Close", "Volume", "x", "x", "x"])
            df["Date"] = pd.to_datetime(df["ts"], unit='ms')
            df.set_index("Date", inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df.sort_index()
    except Exception as e:
        print(f"OKX Error: {e}")
        return None
    return None

def fetch_yahoo_safe(symbol, interval):
    try:
        # Periyot ayarları
        p = "2y" if interval == "1wk" else ("1y" if interval == "1d" else "1mo")
        i = "1h" if interval == "4h" else ("1d" if interval == "1d" else "1wk")
        
        # Yahoo'dan veri çek
        df = yf.download(symbol, period=p, interval=i, progress=False, auto_adjust=True)
        
        if df.empty: return None

        # DÜZELTME: MultiIndex sütunlarını temizle (örn: ('Close', 'BTC-USD') -> 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Saat dilimini kaldır (Timezone-naive yap)
        if df.index.tz is not None: 
            df.index = df.index.tz_localize(None)
            
        # 4 Saatlik veriyi Yahoo vermez, 1 saatlik alıp biz birleştiriyoruz
        if interval == "4h":
            agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            # Volume yoksa hata vermesin diye kontrol
            if 'Volume' not in df.columns: 
                df['Volume'] = 0
            
            df = df.resample('4h').agg(agg).dropna()
            
        return df
    except Exception as e:
        print(f"Yahoo Error ({symbol}): {e}")
        return None

# --- YARDIMCI: GÜVENLİ YAHOO ÇEKİCİ (YEDEKLİ) ---
def fetch_yahoo_retry(tickers, interval):
    """Verilen sembol listesini sırayla dener, hangisi çalışırsa onu döndürür."""
    for sym in tickers:
        df = fetch_yahoo_safe(sym, interval)
        if df is not None and not df.empty and len(df) > 5:
            return df
    return None

# --- GRAM ALTIN HESAPLAYICI (GÜÇLENDİRİLMİŞ) ---
def fetch_gram_gold_calculated(interval):
    """
    Gram Altın (TL) = (ONS Altın ($) * USD/TRY) / 31.1035
    Yedekli semboller kullanır (XAUUSD=X çalışmazsa GC=F devreye girer).
    """
    try:
        # 1. ONS Verisini Çek (Spot Altın -> Vadeli Altın)
        df_ons = fetch_yahoo_retry(["XAUUSD=X", "GC=F"], interval)
        
        # 2. Dolar Verisini Çek (TRY=X -> USDTRY=X)
        df_usd = fetch_yahoo_retry(["TRY=X", "USDTRY=X"], interval)
        
        if df_ons is not None and df_usd is not None:
            # Sütun isimlerini temizle ve yeniden adlandır
            df_ons = df_ons[['Close']].rename(columns={'Close': 'Ons'})
            df_usd = df_usd[['Close']].rename(columns={'Close': 'Usd'})
            
            # Zaman çizelgelerini eşleştir (join)
            # 'inner' join ile sadece iki verinin de olduğu saatleri alırız
            df = df_ons.join(df_usd, how='inner')
            
            # Formül: (Ons * Dolar) / 31.1035
            df['Close'] = (df['Ons'] * df['Usd']) / 31.1035
            
            # Yapay OHLC oluştur
            df['Open'] = df['Close']
            df['High'] = df['Close'] * 1.002
            df['Low'] = df['Close'] * 0.998
            df['Volume'] = 10000 
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Gram Gold Calculation Error: {e}")
        return None
    return None
def process_data(df, src):
    if df is not None and len(df) > 10:
        try:
            # 'Volume' sütunu yoksa oluştur (Hata önleyici)
            if 'Volume' not in df.columns: df['Volume'] = 0
            
            # İndikatör Hesaplamaları
            df['RSI'] = df.ta.rsi(length=14)
            df['EMA_50'] = df.ta.ema(length=50)
            df['EMA_20'] = df.ta.ema(length=20)
            
            # Bollinger Bantları
            bb = df.ta.bbands(length=20, std=2)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                # BBands sütun isimleri bazen değişebilir, düzeltme:
                cols = df.columns
                # Genellikle BBL_20_2.0 vb. gelir. Son eklenenleri alalım
                df.rename(columns={cols[-5]: 'BB_Lower', cols[-3]: 'BB_Upper'}, inplace=True)
                
            df['ATR'] = df.ta.atr(length=14)
            return df, src
        except Exception as e:
            print(f"Process Error: {e}")
            return None, "İşleme Hatası"
            
    return None, "Yetersiz Veri"
def get_market_data(source_pref, symbol, interval):
    # 1. GRAM ALTIN ÖZEL DURUMU
    if symbol == "GRAM_TRY":
        df = fetch_gram_gold_calculated(interval)
        if df is not None: return process_data(df, "Hesaplamalı (Ons x Dolar)")
        return None, "Veri Hesaplanamadı"

    # 2. ONS ALTIN (Yedekli Yapı)
    if symbol == "XAU_GOLD":
        # Önce Spot Altın, olmazsa Vadeli Altın
        df = fetch_yahoo_retry(["XAUUSD=X", "GC=F"], interval)
        if df is not None: return process_data(df, "Yahoo (Gold)")
        return None, "Veri Yok (Yahoo)"
    
    # 3. FOREX
    if symbol == "EURUSD=X":
        return process_data(fetch_yahoo_safe("EURUSD=X", interval), "Yahoo (Forex)")

    # 4. KRİPTO PARALAR
    df = None
    src_name = ""
    
    # Seçilen kaynağı dene
    if source_pref == "Binance":
        df = fetch_binance_simple(symbol, interval)
        src_name = "Binance"
    elif source_pref == "OKX":
        df = fetch_okx_simple(symbol, interval)
        src_name = "OKX"
    
    # Eğer seçilen kaynak veri getirmediyse Yahoo'ya git
    if df is None or df.empty:
        df = fetch_yahoo_safe(symbol, interval)
        src_name = "Yahoo (Yedek)"

    if df is None or df.empty: 
        return None, "Veri Alınamadı"
        
    return process_data(df, src_name)

# --- CÜZDAN CANLI FİYAT (YEDEKLİ) ---
@st.cache_data(ttl=30, show_spinner=False)
def get_live_price_for_portfolio(coin_name):
    try:
        ticker_symbol = COIN_MAP.get(coin_name)
        
        # GRAM ALTIN HESABI (CANLI)
        if ticker_symbol == "GRAM_TRY":
             # Ons fiyatını bul (Yedekli)
             ons_price = 0
             try: ons_price = yf.Ticker("XAUUSD=X").fast_info['last_price']
             except: pass
             
             if ons_price == 0 or ons_price is None:
                 try: ons_price = yf.Ticker("GC=F").fast_info['last_price']
                 except: pass
             
             # Dolar fiyatını bul (Yedekli)
             usd_price = 0
             try: usd_price = yf.Ticker("TRY=X").fast_info['last_price']
             except: pass
             
             if usd_price == 0 or usd_price is None:
                 try: usd_price = yf.Ticker("USDTRY=X").fast_info['last_price']
                 except: pass

             if ons_price > 0 and usd_price > 0:
                 return (ons_price * usd_price) / 31.1035
             return 0

        # ONS ALTIN (CANLI)
        if ticker_symbol == "XAU_GOLD": 
            try:
                price = yf.Ticker("XAUUSD=X").fast_info['last_price']
                if price and price > 0: return price
            except: pass
            
            # XAUUSD çalışmazsa GC=F dene
            try:
                return yf.Ticker("GC=F").fast_info['last_price']
            except: return 0
        
        # NORMAL KRİPTO
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

def calculate_smart_prediction(df, periods=15):
    try:
        # Veri Hazırlığı (Son 60 mum yeterli)
        work_df = df.tail(60).copy()
        if len(work_df) < 30: return [], []

        # X (Zaman) ve Y (Fiyat) verileri
        y = work_df['Close'].values
        x = np.arange(len(y)).reshape(-1, 1)
        
        # 1. DOĞRUSAL REGRESYON (ANA TREND)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]     # Eğim (Trendin Yönü)
        intercept = model.intercept_
        
        # 2. RSI BAZLI AKILLI DÜZELTME (MOMENTUM KONTROLÜ)
        current_rsi = work_df['RSI'].iloc[-1]
        current_price = y[-1]
        ema_50 = work_df['EMA_50'].iloc[-1]
        
        # Eğim Düzeltme Katsayısı
        adjustment_factor = 1.0
        
        # Senaryo A: Yükseliş Trendi
        if slope > 0:
            if current_rsi > 70: adjustment_factor = 0.4  # RSI Şişmiş, yükselişi frenle
            elif current_rsi > 60: adjustment_factor = 0.7 # Yorulma var
            elif current_rsi < 40: adjustment_factor = 1.3 # Güçlü momentum potansiyeli
            
            # Fiyat EMA50'den çok uzaksa (Ortalamaya Dönüş Riski)
            if current_price > ema_50 * 1.05:
                adjustment_factor *= 0.8
                
        # Senaryo B: Düşüş Trendi
        elif slope < 0:
            if current_rsi < 30: adjustment_factor = 0.4  # RSI Dipte, düşüşü frenle (Tepki Gelebilir)
            elif current_rsi < 40: adjustment_factor = 0.7
            elif current_rsi > 60: adjustment_factor = 1.3 # Düşüş derinleşebilir
        
        # Eğimi Revize Et
        smart_slope = slope * adjustment_factor
        
        # 3. GELECEK TAHMİNİ OLUŞTURMA
        future_dates = []
        predictions = []
        
        last_date = work_df.index[-1]
        time_delta = work_df.index[-1] - work_df.index[-2]
        
        # Tahmin çizgisini son kapanış fiyatından başlat (Süreklilik için)
        start_price = current_price
        
        for i in range(1, periods + 1):
            future_dates.append(last_date + (time_delta * i))
            # Her adımda eğim kadar ekle
            next_price = start_price + (smart_slope * i)
            predictions.append(next_price)
            
        return future_dates, predictions
    except Exception as e:
        print(f"Prediction Error: {e}")
        return [], []

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

# 1. GİZLİ AYARLAR (Token ve ID buraya saklandı)
with st.sidebar.expander("🔐 Bot & API Ayarları", expanded=False):
    st.caption("Telegram bildirimleri için gereklidir.")
    # Kullanıcıdan giriş alırken tg_token değişkenine atıyoruz
    tg_token = st.text_input("Bot Token", value=DEFAULT_TOKEN, type="password")
    tg_chat = st.text_input("Chat ID", value=DEFAULT_CHAT_ID)
    st.caption("Bu ayarlar varsayılan olarak kapalıdır.")

st.sidebar.divider()

# 2. NORMAL AYARLAR
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

# Otomatik Bot kutusu kolay erişim için dışarıda kalsın
auto = st.sidebar.checkbox("Otomatik Bot")

# --- BURADAN SONRA intervals... DİYE DEVAM EDEN KODUNUZ GELECEK ---

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

st.title(f"📈 {sel_c} V47 (Cloud & Money)")
c = "green" if "Binance" in active_src else ("blue" if "OKX" in active_src else "orange")
st.markdown(f"**Veri Kaynağı:** <span style='color:{c}; font-weight:bold'>{active_src}</span>", unsafe_allow_html=True)

view_tf = st.selectbox("Periyot:", list(intervals.keys()), format_func=lambda x: intervals[x])
df_view = results[view_tf]

if df_view is not None:
    curr = df_view['Close'].iloc[-1]
    current_atr = df_view.iloc[-1]['ATR'] if 'ATR' in df_view.columns else curr * 0.02
    # ---   (Risk Hesaplayıcı Başlangıcı) ---
    with st.sidebar.expander("🧮 Hızlı Risk Hesapla", expanded=False):
        st.caption("Pozisyon büyüklüğü hesaplar.")
        # Mevcut fiyatı otomatik getiriyoruz
        calc_price = st.number_input("Giriş Fiyatı", value=float(curr), format="%.4f")
        calc_sl = st.number_input("Stop Fiyatı", value=float(curr)*0.98, format="%.4f")
        calc_balance = st.number_input("Kasa ($)", value=1000.0)
        calc_risk = st.number_input("Risk (%)", value=1.0, step=0.5)
        
        if calc_price > 0 and calc_sl > 0 and calc_price != calc_sl:
            risk_amt = calc_balance * (calc_risk / 100)
            diff_pct = abs(calc_price - calc_sl) / calc_price
            pos_size = risk_amt / diff_pct
            coin_qty = pos_size / calc_price
            
            st.divider()
            st.write(f"💸 **Risk Tutarı:** ${risk_amt:.2f}")
            st.success(f"💰 **İşlem Büyüklüğü:** ${pos_size:.2f}")
            st.info(f"🪙 **Alınacak Adet:** {coin_qty:.4f}")
    # --- BURADA BİTİYOR (Risk Hesaplayıcı Sonu) ---
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

   # --- GRAFİK AYARLARI (GÜVENLİ & HİBRİT VERSİYON - DÜZELTİLDİ) ---
    import numpy as np 
    
    try:
        # 1. Ölçek Tipi Belirleme
        y_type = "log" if view_tf == "1wk" else "linear"

        # 2. Zoom ve Veri Aralığı Hesaplama
        zoom_count = 50 if view_tf == "1wk" else (80 if view_tf == "1d" else 100)
        
        if len(df_view) > zoom_count:
            visible_df = df_view.tail(zoom_count)
            zoom_start = visible_df.index[0]
            y_min_raw = visible_df['Low'].min()
            y_max_raw = visible_df['High'].max()
        else:
            zoom_start = df_view.index[0]
            y_min_raw = df_view['Low'].min()
            y_max_raw = df_view['High'].max()

        # 3. Y Ekseni Aralığı (Range) Belirleme
        range_y = None 
        
        if y_type == "log":
            safe_min = max(y_min_raw, 0.000001) 
            range_y = [np.log10(safe_min * 0.90), np.log10(y_max_raw * 1.10)]
        else:
            range_y = [y_min_raw * 0.95, y_max_raw * 1.05]

        # 4. Geleceğe Boşluk Bırakma
        gap_multiplier = 3 if view_tf == "1wk" else 5
        if len(df_view) > 2:
            delta = df_view.index[-1] - df_view.index[-2]
            zoom_end = df_view.index[-1] + (delta * gap_multiplier)
        else:
            zoom_end = df_view.index[-1]

        # 5. Layout Güncelleme
        fig.update_layout(
            height=900, 
            template="plotly_dark", 
            xaxis_rangeslider_visible=False, 
            dragmode="pan",
            title=None,
            yaxis=dict(
                side="right", 
                fixedrange=False, 
                type=y_type, 
                range=range_y,         
                tickformat=".2f",      
                exponentformat="none"  
            ),
            xaxis=dict(
                range=[zoom_start, zoom_end],
                type="date"
            ),
            margin=dict(l=10, r=60, t=10, b=20),
            hovermode='x unified'
        )

        # 6. Config (Sabitleme)
        config = {
            'scrollZoom': True, 
            'displayModeBar': True, 
            'editable': False, 
            'showAxisRangeEntryBoxes': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
            'displaylogo': False
        }
        
        # --- DÜZELTME BURADA YAPILDI ---
        # 1. use_container_width=True  ---> width="stretch" (Uyarıyı çözer)
        # 2. key="main_price_chart"    ---> (Duplicate ID hatasını çözer)
        st.plotly_chart(fig, width="stretch", config=config, key="main_price_chart")

    except Exception as e:
        print("GRAFİK HATASI:", e)
        traceback.print_exc()
        st.error(f"Grafik çizilirken hata oluştu: {e}")
        
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
        # Sinyali Hesapla
        signal_status, color, target_msg = calculate_oracle_signal_fixed(df_view, s_list, r_list)
        
        # Trend Kontrolü (EMA 50)
        is_uptrend = curr > df_view['EMA_50'].iloc[-1]
        trend_note = ""
        
        # ÇELİŞKİ ÇÖZÜCÜ & NÖTR MANTIK
        if "AL" in signal_status:
            if is_uptrend:
                signal_status = "GÜÇLÜ AL (Trend Yönünde) 🚀"
                trend_note = "Trend seninle, güvenli işlem."
            else:
                signal_status = "TEPKİ ALIMI (Riskli) ⚠️"
                trend_note = "Trend Düşüşte! Sadece kısa vadeli tepki (Scalp)."
                color = "orange"
        
        elif "SAT" in signal_status:
            if not is_uptrend:
                signal_status = "GÜÇLÜ SAT (Trend Yönünde) 🔻"
                trend_note = "Trend aşağı, düşüş derinleşebilir."
            else:
                signal_status = "DÜZELTME SATIŞI (Riskli) ⚠️"
                trend_note = "Trend Yükselişte! Fiyat sadece dinleniyor olabilir."
        
        else: # --- YENİ EKLENEN KISIM: NÖTR BÖLGE ---
            signal_status = "NÖTR (BEKLE) 💤"
            trend_note = "Piyasa kararsız veya yatay. İşlem yapma, izle."
            color = "gray"
            target_msg = "Yön Belirsiz"

        # Ekrana Yazdır
        st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:20px'>{signal_status}</span>", unsafe_allow_html=True)
        st.caption(trend_note)
        st.write(f"**{target_msg}**")
        
        # Setup Sadece AL veya SAT varsa gösterilir
        if "AL" in signal_status or "SAT" in signal_status:
            setup = calculate_trade_setup(df_view, "AL" if "AL" in signal_status else "SAT")
            if setup:
                st.divider()
                c_s1, c_s2 = st.columns(2)
                c_s1.write(f"**Giriş:** ${setup['entry']:,.2f}")
                c_s1.write(f"**🛑 Stop:** ${setup['sl']:,.2f}")
                c_s2.write(f"**🎯 TP:** ${setup['tp']:,.2f}")
        else:
            st.info("Setup oluşmadı. Güvenli bölge bekleniyor.")

# --- CÜZDAN (NAKİT YÖNETİMİ & AI TARAYICI) ---
    st.divider()
    st.header("💼 Varlık ve Fırsat Yönetimi")
    
# --- YENİ ÖZELLİK: AI PİYASA TARAYICI (BAKİYE ÖNERİLİ) ---
    with st.expander("🔍 Piyasayı Tara & Fırsat Bul (AI)", expanded=False):
        st.info("Bu modül RSI, Trend ve Destek/Direnç analizi yaparak kasa yönetimi önerir.")
        
        scan_tf = st.radio("Tarama Periyodu:", ["4h", "1d", "1wk"], horizontal=True, format_func=lambda x: intervals[x])
        
        if st.button("🚀 Taramayı Başlat"):
            best_opp = None
            best_score = -100
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            coins_to_scan = list(COIN_MAP.keys())
            total_coins = len(coins_to_scan)
            
            results_scan = []

            for i, c_name in enumerate(coins_to_scan):
                status_text.text(f"Analiz ediliyor: {c_name}...")
                progress_bar.progress((i + 1) / total_coins)
                
                sym = COIN_MAP[c_name]
                d_scan, _ = get_market_data("Yahoo Finance", sym, scan_tf)
                
                if d_scan is not None and len(d_scan) > 20:
                    last_price = d_scan['Close'].iloc[-1]
                    last_rsi = d_scan['RSI'].iloc[-1]
                    ema_50 = d_scan['EMA_50'].iloc[-1]
                    sup_list, res_list = calculate_sr(d_scan, scan_tf)
                    nearest_sup = max([s for s in sup_list if s < last_price], default=0)
                    
                    # --- PUANLAMA ---
                    score = 50 
                    reasons = []
                    
                    # RSI
                    if last_rsi < 35: score += 20; reasons.append("RSI Dip")
                    elif last_rsi > 65: score -= 20
                    
                    # Trend
                    if last_price > ema_50: score += 10
                    
                    # Destek
                    if nearest_sup > 0 and (last_price - nearest_sup)/last_price < 0.03:
                        score += 25; reasons.append("Desteğe Yakın")
                    
                    # --- BAKİYE YÖNETİMİ ÖNERİSİ ---
                    alloc_pct = 0
                    if score >= 85: alloc_pct = 10  # Çok Güçlü Fırsat -> %10
                    elif score >= 70: alloc_pct = 5 # İyi Fırsat -> %5
                    elif score >= 60: alloc_pct = 2.5 # Normal -> %2.5
                    
                    if score >= 60: signal_str = "AL"
                    elif score <= 40: signal_str = "SAT"
                    else: signal_str = "NÖTR"
                    
                    results_scan.append({
                        "Coin": c_name, 
                        "Sinyal": signal_str,
                        "Puan": score,
                        "Önerilen Kasa (%)": f"%{alloc_pct}" if alloc_pct > 0 else "-",
                        "Sebep": ", ".join(reasons)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_opp = results_scan[-1]

            progress_bar.empty()
            status_text.empty()
            
            if best_opp:
                st.success(f"🌟 **EN İYİ FIRSAT:** {best_opp['Coin']}")
                c1, c2 = st.columns(2)
                c1.metric("Puan", best_opp['Puan'])
                c2.metric("Önerilen Yatırım", best_opp['Önerilen Kasa (%)'], help="Toplam varlığının bu kadarı ile işlem açman önerilir.")
                st.caption(f"Sebep: {best_opp['Sebep']}")
            
            st.dataframe(pd.DataFrame(results_scan).sort_values(by="Puan", ascending=False), use_container_width=True)

    st.divider()

# --- PORTFÖY VE EMİR YÖNETİMİ (LİMİT EMİR İPTAL/DÜZENLE & DÜZELTİLMİŞ) ---
    st.divider()
    col_risk, col_wallet = st.columns([1, 2])
    
    # Global Bakiye
    current_balance = st.session_state['portfolio_data'].get('balance', 0.0)

    with col_risk:
        st.subheader("🧮 Emir Gir")
        
        # 1. Giriş Ayarları
        entry_price = st.number_input("Giriş Fiyatı ($)", value=float(curr), step=0.01, format="%.4f")
        investment = st.number_input("İşlem Tutarı ($)", value=1000.0, step=100.0)
        
        # Limit Emir Seçeneği
        is_limit = st.checkbox("⏳ Limit Emir (Fiyat Bekle)", value=False, help="İşaretlenirse hemen almaz, fiyat bu seviyeye gelinceye kadar bekler.")
        
        # Bakiye Kullanımı
        use_balance = st.checkbox(f"🏦 Bakiyeden Kullan (${current_balance:,.2f})", value=True)
        
        # Risk Bilgisi
        atr_val = current_atr if 'current_atr' in locals() else entry_price*0.02
        st.caption(f"Stop Önerisi: ${(entry_price - atr_val * 1.5):.2f}")

        if st.button("➕ Emri Gir / Ekle"):
            # Bakiye Kontrolü
            if use_balance:
                if investment > current_balance:
                    st.error("Yetersiz Bakiye! Lütfen Bakiye Düzenle kısmından para ekleyin.")
                    st.stop()
                else:
                    st.session_state['portfolio_data']['balance'] -= investment
            
            status = "PENDING" if is_limit else "ACTIVE"
            
            new_trade = {
                "Coin": sel_c,
                "Giriş": entry_price,
                "Adet": investment / entry_price,
                "Yatırım": investment,
                "Realized": 0.0,
                "Status": status, 
                "Tarih": time.strftime("%Y-%m-%d")
            }
            
            st.session_state['portfolio_data']['positions'].append(new_trade)
            save_portfolio(st.session_state['portfolio_data'])
            
            msg = "Limit Emir Girildi! Fiyat bekleniyor..." if is_limit else "Pozisyon Açıldı!"
            st.success(msg)
            time.sleep(1)
            st.rerun()

        # --- BAKİYE DÜZENLEME PANELİ ---
        st.write("---")
        with st.expander("💳 Cüzdan Bakiyesi Düzenle"):
            st.info("Borsadaki boş USDT miktarınızı buraya girin.")
            new_balance_input = st.number_input("Güncel USDT Bakiyesi", value=float(current_balance), step=100.0)
            if st.button("Bakiyeyi Güncelle"):
                st.session_state['portfolio_data']['balance'] = new_balance_input
                save_portfolio(st.session_state['portfolio_data'])
                st.success("Bakiye güncellendi!")
                time.sleep(0.5)
                st.rerun()

    with col_wallet:
        st.subheader("💰 Varlıklarım")
        
        positions = st.session_state['portfolio_data']['positions']
        
        # HATA DÜZELTME: Değişkeni döngüden önce sıfırlıyoruz
        total_active_value = 0 
        
        if positions:
            # --- TABLOLARI AYIR ---
            active_pos = [p for p in positions if p.get('Status', 'ACTIVE') == 'ACTIVE']
            pending_pos = [p for p in positions if p.get('Status') == 'PENDING']
            
            # 1. AKTİF POZİSYONLAR TABLOSU
            if active_pos:
                st.markdown("##### ✅ Aktif Pozisyonlar")
                # Satış Paneli (Sadece Aktifler İçin)
                with st.expander("💸 Kar Al / Satış Yap"):
                    p_coins = list(set([p['Coin'] for p in active_pos]))
                    s_coin = st.selectbox("Coin", p_coins, key="sell_sel")
                    
                    target_pos = next((p for p in active_pos if p['Coin'] == s_coin), None)
                    if target_pos:
                        cur_qty = target_pos['Adet']
                        sell_price = st.number_input("Satış Fiyatı", value=float(curr if s_coin == sel_c else target_pos['Giriş']))
                        sell_pct = st.slider("Satış %", 0, 100, 50)
                        
                        sell_amt = cur_qty * (sell_pct / 100)
                        total_return = sell_amt * sell_price
                        
                        st.write(f"**Gelecek Nakit:** ${total_return:,.2f}")
                        
                        if st.button("Satışı Onayla"):
                            st.session_state['portfolio_data']['balance'] += total_return
                            
                            cost_basis = target_pos['Giriş'] * sell_amt
                            pnl = total_return - cost_basis
                            
                            target_pos['Adet'] -= sell_amt
                            target_pos['Yatırım'] -= cost_basis
                            target_pos['Realized'] += pnl
                            
                            save_portfolio(st.session_state['portfolio_data'])
                            st.success("Satış gerçekleşti!")
                            st.rerun()

                # Aktif Tablo Verisi
                active_data = []
                for item in active_pos:
                    if item['Adet'] > 0:
                        lp = curr if item['Coin'] == sel_c else get_live_price_for_portfolio(item['Coin'])
                        if lp == 0: lp = item['Giriş']
                        val = item['Adet'] * lp
                        pnl_usd = val - item['Yatırım']
                        pnl_pct = (pnl_usd / item['Yatırım']) * 100
                        
                        total_active_value += val
                        
                        active_data.append({
                            "Coin": item['Coin'],
                            "Giriş": item['Giriş'],
                            "Adet": item['Adet'],
                            "Değer ($)": val,
                            "Kar/Zarar ($)": pnl_usd,
                            "Kar/Zarar (%)": f"%{pnl_pct:.2f}"
                        })
                
                if active_data:
                    st.dataframe(pd.DataFrame(active_data), use_container_width=True)

            # 2. BEKLEYEN EMİRLER TABLOSU (İPTAL VE DÜZENLE EKLENDİ)
            if pending_pos:
                st.markdown("##### ⏳ Bekleyen Limit Emirler")
                pending_data = []
                for item in pending_pos:
                    lp = curr if item['Coin'] == sel_c else get_live_price_for_portfolio(item['Coin'])
                    diff_pct = ((lp - item['Giriş']) / lp) * 100
                    
                    pending_data.append({
                        "Coin": item['Coin'],
                        "Hedef Giriş": item['Giriş'],
                        "Anlık Fiyat": lp,
                        "Uzaklık (%)": f"%{diff_pct:.2f}",
                        "Kilitli Tutar": item['Yatırım']
                    })
                
                st.dataframe(pd.DataFrame(pending_data), use_container_width=True)
                
                # --- EMİR YÖNETİM PANELİ ---
                with st.expander("🛠️ Emri Yönet (Düzenle / İptal / Başlat)", expanded=True):
                    # İşlem yapılacak coini seç (Index ile değil, Coin ismi ve Fiyat ile eşleştiriyoruz)
                    # Benzersiz olması için Coin + Fiyat birleşimi kullanıyoruz string olarak
                    p_opts = [f"{p['Coin']} - ${p['Giriş']}" for p in pending_pos]
                    selected_opt = st.selectbox("İşlem Yapılacak Emir", p_opts)
                    
                    # Seçilen emri bul
                    sel_coin_name = selected_opt.split(" - ")[0]
                    sel_price_val = float(selected_opt.split(" - $")[1])
                    
                    target_pending = next((p for p in pending_pos if p['Coin'] == sel_coin_name and abs(p['Giriş'] - sel_price_val) < 0.0001), None)
                    
                    if target_pending:
                        c_man1, c_man2, c_man3 = st.columns(3)
                        
                        # 1. İPTAL ET
                        with c_man1:
                            if st.button("❌ İptal Et (Para İade)", type="secondary"):
                                # Parayı iade et
                                st.session_state['portfolio_data']['balance'] += target_pending['Yatırım']
                                # Listeden sil
                                st.session_state['portfolio_data']['positions'].remove(target_pending)
                                save_portfolio(st.session_state['portfolio_data'])
                                st.success("Emir iptal edildi, para iade edildi.")
                                time.sleep(1)
                                st.rerun()

                        # 2. DÜZENLE
                        with c_man2:
                            new_limit_price = st.number_input("Yeni Hedef Fiyat", value=float(target_pending['Giriş']), format="%.4f")
                            if st.button("✏️ Fiyatı Güncelle"):
                                if new_limit_price > 0:
                                    target_pending['Giriş'] = new_limit_price
                                    # Tutarı sabit tutup adeti güncelliyoruz
                                    target_pending['Adet'] = target_pending['Yatırım'] / new_limit_price
                                    save_portfolio(st.session_state['portfolio_data'])
                                    st.success("Emir fiyatı güncellendi.")
                                    time.sleep(1)
                                    st.rerun()

                        # 3. AKTİFLEŞTİR
                        with c_man3:
                            if st.button("🚀 Manuel Başlat"):
                                target_pending['Status'] = 'ACTIVE'
                                # Giriş fiyatı limit fiyat olarak kalır
                                save_portfolio(st.session_state['portfolio_data'])
                                st.success("Emir aktife alındı!")
                                st.rerun()

            # --- TOPLAM ÖZET ---
            st.divider()
            
            pending_reserved = sum([p['Yatırım'] for p in pending_pos]) # Bekleyen emirlerdeki kilitli para
            
            total_equity = current_balance + total_active_value + pending_reserved
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Boştaki USDT", f"${current_balance:,.2f}")
            m2.metric("Aktif Pozisyonlar", f"${total_active_value:,.2f}")
            m3.metric("🏆 TOPLAM VARLIK", f"${total_equity:,.2f}")
            
            if st.button("🗑️ Portföyü Sıfırla"):
                st.session_state['portfolio_data'] = {"balance": 1000.0, "positions": []}
                save_portfolio(st.session_state['portfolio_data'])
                st.rerun()
                
        else:
            st.info("Portföy boş. Bakiye düzenleyebilir veya işlem açabilirsiniz.")
            st.metric("Mevcut Bakiye", f"${current_balance:,.2f}")

else: st.error("Veri Alınamadı.")
# BOT KISMI AYNEN KALACAK...

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



















