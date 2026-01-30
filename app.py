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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
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
ASSETS_FILE = "varliklar.json"
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
# İlk açılışta veya sıfırlamada kullanılacak varsayılan liste
DEFAULT_COIN_MAP = {
    "Bitcoin (BTC)": "BTC-USD", 
    "Ethereum (ETH)": "ETH-USD", 
    "Solana (SOL)": "SOL-USD", 
    "Ripple (XRP)": "XRP-USD", 
    "Avax (AVAX)": "AVAX-USD", 
    "Dogecoin (DOGE)": "DOGE-USD", 
    "Pepe": "PEPE-USD", 
    "ONS ALTIN ($)": "XAU_GOLD",
    "GRAM ALTIN (TL)": "GRAM_TRY",
    "EUR/USD": "EURUSD=X",
    "Türk Hava Yolları (THYAO)": "THYAO.IS",
    "Pegasus (PGSUS)": "PGSUS.IS"
}

def load_assets():
    """Varlık listesini dosyadan yükler, yoksa varsayılanı oluşturur."""
    if os.path.exists(ASSETS_FILE):
        try:
            with open(ASSETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return DEFAULT_COIN_MAP.copy()
    else:
        # Dosya yoksa varsayılanı kaydet ve döndür
        save_assets(DEFAULT_COIN_MAP)
        return DEFAULT_COIN_MAP.copy()

def save_assets(data):
    """Varlık listesini dosyaya kaydeder."""
    with open(ASSETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Uygulama Başlarken Varlıkları Yükle
if 'coin_map' not in st.session_state:
    st.session_state['coin_map'] = load_assets()

# Artık COIN_MAP global değişkenini session state'e bağlıyoruz
COIN_MAP = st.session_state['coin_map']
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
def fetch_binance_simple(symbol, interval, limit=1000):
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
        p = "10y" if interval == "1wk" else ("4y" if interval == "1d" else "1mo")
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
def validate_portfolio_risk(new_investment, current_balance, open_positions):
    """
    EKLENECEK YER: save_portfolio fonksiyonundan hemen sonra
    Kelly Criterion ve maksimum pozisyon büyüklüğü kontrolü
    """
    total_equity = current_balance + sum([p.get('Yatırım', 0) for p in open_positions if p.get('Status') == 'ACTIVE'])
    
    # Tek pozisyon max %20
    if new_investment > total_equity * 0.20:
        return False, "⚠️ Tek pozisyon toplam varlığın %20'sini aşamaz!"
    
    # Toplam risk max %50
    total_exposure = sum([p.get('Yatırım', 0) for p in open_positions if p.get('Status') == 'ACTIVE']) + new_investment
    if total_exposure > total_equity * 0.50:
        return False, "⚠️ Toplam açık pozisyon %50'yi geçemez!"
    
    return True, "✅ Risk kabul edilebilir"

# ========================================
# 🆕 YENİ EKLEME 2: Sentiment API
# ========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_fear_greed_index():
    """
    EKLENECEK YER: validate_portfolio_risk fonksiyonundan sonra
    Crypto Fear & Greed Index (0-100)
    """
    try:
        url = "https://api.alternative.me/fng/"
        r = requests.get(url, timeout=5)
        data = r.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return value, classification
    except:
        return 50, "Neutral"  # Hata durumunda nötr döndür
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

def check_active_positions_auto_close(portfolio_data):
    """
    EKLENECEK YER: calculate_smart_prediction fonksiyonundan ÖNCE
    (Ama calculate_setup_dynamic fonksiyonundan SONRA)
    
    Aktif pozisyonları kontrol eder, TP/SL'ye ulaşanları otomatik kapatır
    """
    positions = portfolio_data.get('positions', [])
    closed_count = 0
    closed_trades = []
    
    for pos in positions:
        if pos.get('Status') != 'ACTIVE': 
            continue
        
        coin_name = pos['Coin']
        entry = pos['Giriş']
        qty = pos['Adet']
        investment = pos['Yatırım']
        
        # Canlı fiyatı al
        live_price = get_live_price_for_portfolio(coin_name)
        if live_price == 0 or live_price is None: 
            continue
        
        # ATR al (4 saatlik için)
        symbol = COIN_MAP.get(coin_name)
        if not symbol:
            continue
            
        df_check, _ = get_market_data("Binance", symbol, "4h")
        
        if df_check is not None and len(df_check) > 20:
            atr = df_check['ATR'].iloc[-1]
            if np.isnan(atr) or atr == 0:
                atr = entry * 0.02
            
            # Yön belirle
            direction = "LONG"  # Varsayılan (sadece long pozisyon destekliyorsunuz)
            
            # Setup hesapla
            setup = calculate_setup_dynamic(entry, atr, direction)
            
            # TP kontrolü
            if live_price >= setup['tp']:
                profit = (live_price - entry) * qty
                pos['Status'] = 'CLOSED_TP'
                pos['Realized'] = pos.get('Realized', 0) + profit
                pos['Exit_Price'] = live_price
                pos['Exit_Date'] = time.strftime("%Y-%m-%d %H:%M")
                
                # Parayı iade et
                portfolio_data['balance'] = portfolio_data.get('balance', 0) + (qty * live_price)
                
                closed_count += 1
                closed_trades.append({
                    'coin': coin_name,
                    'type': 'TP',
                    'profit': profit,
                    'pct': (profit / investment) * 100
                })
            
            # SL kontrolü
            elif live_price <= setup['sl']:
                loss = (live_price - entry) * qty
                pos['Status'] = 'CLOSED_SL'
                pos['Realized'] = pos.get('Realized', 0) + loss
                pos['Exit_Price'] = live_price
                pos['Exit_Date'] = time.strftime("%Y-%m-%d %H:%M")
                
                portfolio_data['balance'] = portfolio_data.get('balance', 0) + (qty * live_price)
                
                closed_count += 1
                closed_trades.append({
                    'coin': coin_name,
                    'type': 'SL',
                    'profit': loss,
                    'pct': (loss / investment) * 100
                })
    
    if closed_count > 0:
        save_portfolio(portfolio_data)
    
    return closed_count, closed_trades
    
def calculate_sr_advanced(df, timeframe):
    """
    GELİŞTİRİLMİŞ S/R HESAPLAMA:
    - Pivot noktaları
    - Fibonacci seviyeleri
    - Volume profil (yüksek hacimli bölgeler)
    """
    supports, resistances = [], []
    
    # 1. Klasik Pivot
    n = 5 if timeframe == "4h" else (10 if timeframe == "1d" else 20)
    work_df = df.tail(500)
    
    for i in range(n, len(work_df) - n):
        l = work_df['Low'].iloc[i]
        h = work_df['High'].iloc[i]
        
        if l == work_df['Low'].iloc[i-n:i+n+1].min():
            supports.append(l)
        if h == work_df['High'].iloc[i-n:i+n+1].max():
            resistances.append(h)
    
    # 2. Fibonacci Seviyeleri (Son major swing)
    swing_high = work_df['High'].max()
    swing_low = work_df['Low'].min()
    diff = swing_high - swing_low
    
    # Fibonacci retracement seviyeleri
    fib_levels = [
        swing_low + diff * 0.236,
        swing_low + diff * 0.382,
        swing_low + diff * 0.5,
        swing_low + diff * 0.618,
        swing_low + diff * 0.786
    ]
    
    current_price = df['Close'].iloc[-1]
    
    # Mevcut fiyatın altındakiler destek, üstündekiler direnç
    for fib in fib_levels:
        if fib < current_price:
            supports.append(fib)
        else:
            resistances.append(fib)
    
    # 3. Volume Profile (Yüksek hacimli fiyat seviyeleri = güçlü S/R)
    if 'Volume' in work_df.columns and work_df['Volume'].sum() > 0:
        # Fiyatı 50 bine böl, her binde toplam hacmi hesapla
        price_bins = pd.cut(work_df['Close'], bins=50)
        volume_profile = work_df.groupby(price_bins)['Volume'].sum()
        
        # En yüksek hacimli 3 seviye
        top_volume_levels = volume_profile.nlargest(3).index
        for interval in top_volume_levels:
            mid_price = (interval.left + interval.right) / 2
            if mid_price < current_price:
                supports.append(mid_price)
            else:
                resistances.append(mid_price)
    
    # 4. Temizlik ve Sıralama
    supports = sorted(list(set([round(x, 2) for x in supports])))
    resistances = sorted(list(set([round(x, 2) for x in resistances])))
    
    return supports, resistances

def calculate_oracle_signal_v2(df, supports, resistances):
    """
    GELİŞTİRİLMİŞ SİNYAL SİSTEMİ:
    - Çoklu zaman dilimi onayı
    - Hacim doğrulaması
    - Destek/Direnç yakınlığı
    """
    if df is None or len(df) < 50: 
        return "Veri Yok", "gray", ""
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === 1. TEKNİK OKUMA ===
    rsi = last['RSI']
    price = last['Close']
    ema_20 = last['EMA_20']
    ema_50 = last['EMA_50']
    bb_lower = last['BB_Lower']
    bb_upper = last['BB_Upper']
    
    # MACD Çapraz
    macd_current = last.get('MACD', 0)
    macd_prev = prev.get('MACD', 0)
    macd_signal = last.get('MACD_Signal', 0) if 'MACD_Signal' in last else 0
    macd_cross_up = macd_prev < macd_signal and macd_current > macd_signal
    macd_cross_down = macd_prev > macd_signal and macd_current < macd_signal
    
    # Hacim Onayı
    vol_ratio = 1
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = last['Volume'] / vol_avg if vol_avg > 0 else 1
    
    # === 2. DESTEK/DİRENÇ YAKINLIĞI ===
    nearest_support = max([s for s in supports if s < price], default=0)
    nearest_resistance = min([r for r in resistances if r > price], default=price * 2)
    
    support_dist = ((price - nearest_support) / price * 100) if nearest_support > 0 else 100
    resist_dist = ((nearest_resistance - price) / price * 100) if nearest_resistance < price * 2 else 100
    
    # === 3. PUAN SİSTEMİ (0-100) ===
    score = 50  # Nötr başlangıç
    
    # RSI Puanı (-30 ile +30 arası)
    if rsi < 30: score += 30
    elif rsi < 40: score += 15
    elif rsi > 70: score -= 30
    elif rsi > 60: score -= 15
    
    # Trend Puanı
    if price > ema_20 > ema_50: score += 20  # Güçlü uptrend
    elif price < ema_20 < ema_50: score -= 20  # Güçlü downtrend
    
    # Bollinger
    if price < bb_lower: score += 15
    elif price > bb_upper: score -= 15
    
    # MACD Çapraz
    if macd_cross_up: score += 10
    elif macd_cross_down: score -= 10
    
    # Hacim Onayı
    if vol_ratio > 1.5: score += 10  # Yüksek hacim = güçlü hareket
    elif vol_ratio < 0.7: score -= 5  # Düşük hacim = zayıf sinyal
    
    # Destek/Direnç yakınlığı
    if support_dist < 2: score += 20  # Desteğe çok yakın
    if resist_dist < 2: score -= 20  # Dirence çok yakın
    
    # === 4. KARAR MANTĞI ===
    if score >= 75:
        status, color = "GÜÇLÜ AL 🚀", "green"
        target_msg = f"📈 Hedef: ${nearest_resistance:,.2f} (+%{resist_dist:.1f})"
    elif score >= 60:
        status, color = "AL (Dikkatli) 📊", "blue"
        target_msg = f"Hedef: ${bb_upper:,.2f}"
    elif score <= 25:
        status, color = "GÜÇLÜ SAT 📉", "red"
        target_msg = f"📉 Hedef: ${nearest_support:,.2f} (-%{support_dist:.1f})"
    elif score <= 40:
        status, color = "SAT (Kısmi) ⚠️", "orange"
        target_msg = f"Hedef: ${bb_lower:,.2f}"
    else:
        status, color = "NÖTR (BEKLE) 💤", "gray"
        target_msg = f"Skor: {score}/100 - Yön belirsiz"
    
    return status, color, target_msg
    
def multi_timeframe_confirmation(coin_name, symbol):
    """
    EKLENECEK YER: calculate_oracle_signal_v2 fonksiyonundan sonra
    3 zaman diliminde de aynı yönde sinyal varsa güçlü onay
    """
    signals = {}
    scores = []
    
    for tf in ["4h", "1d", "1wk"]:
        try:
            df, _ = get_market_data("Binance", symbol, tf)
            if df is not None and len(df) > 50:
                s_list, r_list = calculate_sr_advanced(df, tf)
                status, color, _ = calculate_oracle_signal_v2(df, s_list, r_list)
                
                if "AL" in status: 
                    signals[tf] = "AL"
                    scores.append(1)
                elif "SAT" in status: 
                    signals[tf] = "SAT"
                    scores.append(-1)
                else: 
                    signals[tf] = "NÖTR"
                    scores.append(0)
        except:
            signals[tf] = "HATA"
            scores.append(0)
    
    # Onay kontrolü
    if len(scores) == 3:
        if all(s > 0 for s in scores): return "✅ ÜÇ DİLİM AL ONAYI", signals
        elif all(s < 0 for s in scores): return "❌ ÜÇ DİLİM SAT ONAYI", signals
        elif sum(scores) > 0: return "⚠️ KARMA (AL Ağırlıklı)", signals
        elif sum(scores) < 0: return "⚠️ KARMA (SAT Ağırlıklı)", signals
    
    return "📊 Çelişkili Sinyaller", signals
    
def calculate_trailing_stop(entry, current_price, atr, trailing_pct=0.05):
    """
    EKLENECEK YER: multi_timeframe_confirmation fonksiyonundan sonra
    Trailing stop: Fiyat %X yükselince stop'u yukarı çek
    """
    initial_stop = entry - (atr * 1.5)
    
    if current_price > entry * (1 + trailing_pct):
        new_stop = current_price - (atr * 1.2)
        return max(initial_stop, new_stop)
    
    return initial_stop
    
def calculate_smart_prediction_FIXED(df, periods=15):
    """
    DATA LEAKAGE ÖNLENMİŞ VERSİYON
    - Scaler sadece train setine fit edilir
    - Test seti "görmemiş" gibi davranır
    - Simülasyon sırasında scaler sabittir
    """
    try:
        work_df = df.copy()
        if len(work_df) < 150: return [], [], 0
        
        # === İNDİKATÖRLER (Değişiklik yok) ===
        work_df['RSI'] = work_df.ta.rsi(length=14)
        work_df['CCI'] = work_df.ta.cci(length=20)
        work_df['ATR'] = work_df.ta.atr(length=14)
        
        macd = work_df.ta.macd(fast=12, slow=26, signal=9)
        work_df['MACD'] = macd['MACD_12_26_9'] if macd is not None else 0
        work_df['MACD_Signal'] = macd['MACDs_12_26_9'] if macd is not None else 0
        
        stoch = work_df.ta.stochrsi()
        work_df['StochRSI_K'] = stoch['STOCHRSIk_14_14_3_3'] if stoch is not None else 50
        
        if 'Volume' in work_df.columns and work_df['Volume'].sum() > 0:
            work_df['OBV'] = work_df.ta.obv()
            work_df['Volume_SMA'] = work_df['Volume'].rolling(20).mean()
            work_df['Volume_Ratio'] = work_df['Volume'] / work_df['Volume_SMA']
        else:
            work_df['OBV'] = 0
            work_df['Volume_Ratio'] = 1
        
        adx = work_df.ta.adx(length=14)
        work_df['ADX'] = adx['ADX_14'] if adx is not None else 25
        
        work_df['Price_SMA50'] = work_df['Close'].rolling(50).mean()
        work_df['Price_Distance'] = (work_df['Close'] - work_df['Price_SMA50']) / work_df['Price_SMA50'] * 100
        
        bb = work_df.ta.bbands(length=20, std=2)
        if bb is not None:
            work_df['BB_Width'] = (bb.iloc[:, -3] - bb.iloc[:, -5]) / work_df['Close'] * 100
        else:
            work_df['BB_Width'] = 2
        
        work_df['Return'] = work_df['Close'].pct_change()
        work_df['Return_5'] = work_df['Close'].pct_change(5)
        work_df['Volatility'] = work_df['Return'].rolling(20).std()
        
        for lag in [1, 2, 3, 5]:
            work_df[f'Lag{lag}'] = work_df['Return'].shift(lag)
        
        work_df['Target'] = work_df['Return'].shift(-1)
        
        work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        work_df.dropna(inplace=True)
        
        features = [
            'RSI', 'MACD', 'MACD_Signal', 'CCI', 'StochRSI_K',
            'ADX', 'BB_Width', 'Price_Distance',
            'Volume_Ratio', 'OBV',
            'Return', 'Return_5', 'Volatility',
            'Lag1', 'Lag2', 'Lag3', 'Lag5'
        ]
        
        X = work_df[features].values
        y = work_df['Target'].values
        
        # ====================================================
        # 🔥 KRİTİK DÜZELTME BURASI
        # ====================================================
        
        # 1️⃣ ÖNCE VERİYİ BÖL (Ham haliyle)
        test_size = int(len(X) * 0.25)
        X_train_raw = X[:-test_size]
        X_test_raw = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]
        
        # 2️⃣ SCALER'I SADECE TRAIN SETİNE GÖRE EĞİT
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)  # fit_transform sadece train'de
        
        # 3️⃣ TEST SETİNİ TRAIN KURALLARINA GÖRE DÖNÜŞTÜRr
        X_test = scaler.transform(X_test_raw)  # Sadece transform, fit yok!
        
        # ====================================================
        # Artık model gelecekteki min/max değerleri "görmüyor"
        # ====================================================
        
        # === BACKTEST ===
        rf_model = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_split=5, random_state=42)
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)
        ensemble_pred = 0.7 * rf_pred + 0.3 * lr_pred
        
        # Doğruluk hesabı
        mae = mean_absolute_error(y_test, ensemble_pred)
        volatility = np.std(y_test)
        
        direction_acc = np.mean((ensemble_pred > 0) == (y_test > 0)) * 100
        volatility_penalty = min(mae / (volatility + 1e-6), 1.0)
        
        accuracy_score = (direction_acc * 0.6) + ((1 - volatility_penalty) * 40)
        accuracy_score = max(0, min(100, accuracy_score))
        
        # === FINAL MODEL (Tüm veriyle eğit) ===
        # ⚠️ DİKKAT: Burada da scaler'ı sadece TÜM MEVCUT VERİ üzerinde fit ediyoruz
        # Gelecek için tahmin yaparken bu scaler sabit kalacak
        scaler_full = MinMaxScaler()
        X_scaled_full = scaler_full.fit_transform(X)
        
        rf_final = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
        lr_final = LinearRegression()
        rf_final.fit(X_scaled_full, y)
        lr_final.fit(X_scaled_full, y)
        
        # === GELECEK TAHMİNİ ===
        future_dates = []
        predictions = []
        
        last_date = work_df.index[-1]
        time_delta = work_df.index[-1] - work_df.index[-2]
        current_price = work_df['Close'].iloc[-1]
        
        sim_state = work_df[features].iloc[-1].copy()
        confidence_decay = 0.95
        
        for step in range(1, periods + 1):
            next_date = last_date + (time_delta * step)
            future_dates.append(next_date)
            
            # 🔥 ÖNEMLİ: Simülasyonda scaler_full kullanıyoruz (sabit kalıyor)
            sim_input = scaler_full.transform([sim_state.values])
            
            rf_change = rf_final.predict(sim_input)[0]
            lr_change = lr_final.predict(sim_input)[0]
            pred_change = (0.7 * rf_change + 0.3 * lr_change) * (accuracy_score / 100) * (confidence_decay ** step)
            
            next_price = current_price * (1 + pred_change)
            predictions.append(next_price)
            
            # State güncellemeleri (önceki gibi)
            sim_state['Lag5'] = sim_state['Lag3']
            sim_state['Lag3'] = sim_state['Lag2']
            sim_state['Lag2'] = sim_state['Lag1']
            sim_state['Lag1'] = pred_change
            sim_state['Return'] = pred_change
            
            if pred_change > 0:
                sim_state['RSI'] = min(85, sim_state['RSI'] + (pred_change * 100))
                sim_state['CCI'] = min(200, sim_state['CCI'] + (pred_change * 200))
                sim_state['StochRSI_K'] = min(100, sim_state['StochRSI_K'] + 5)
            else:
                sim_state['RSI'] = max(15, sim_state['RSI'] + (pred_change * 100))
                sim_state['CCI'] = max(-200, sim_state['CCI'] + (pred_change * 200))
                sim_state['StochRSI_K'] = max(0, sim_state['StochRSI_K'] - 5)
            
            sim_state['MACD'] += pred_change * 0.5
            sim_state['Price_Distance'] = ((next_price - current_price) / current_price) * 100
            sim_state['Volatility'] = sim_state['Volatility'] * 0.95 + abs(pred_change) * 0.05
            
            if abs(pred_change) < 0.005:
                sim_state['ADX'] = max(10, sim_state['ADX'] * 0.9)
            
            current_price = next_price
        
        return future_dates, predictions, accuracy_score
        
    except Exception as e:
        print(f"AI FIXED Error: {e}")
        import traceback
        traceback.print_exc()
        return [], [], 0

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
 # =========================================
# 🆕 GELİŞMİŞ FORMASYON TESPİT SİSTEMİ
# =========================================

def detect_advanced_patterns(df):
    """
    EKLENECEK YER: detect_patterns fonksiyonundan SONRA
    
    Gelişmiş teknik analiz formasyonlarını tespit eder:
    - Üçgen formasyonlar
    - Harmonik formasyonlar (ABCD, Butterfly, Gartley)
    - Baş-Omuz
    - Bayrak/Flama
    - Kama formasyonları
    """
    patterns = []
    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    if len(df) < 50:
        return patterns
    
    # =========================================
    # 1. ÜÇGEN FORMASYONLARI
    # =========================================
    try:
        # Son 60 mum için trend çizgileri çiz
        window = min(60, len(df))
        work_slice = df.tail(window)
        
        # Üst tepe noktaları
        peak_indices = argrelextrema(work_slice['High'].values, np.greater, order=5)[0]
        # Alt dip noktaları
        trough_indices = argrelextrema(work_slice['Low'].values, np.less, order=5)[0]
        
        if len(peak_indices) >= 2 and len(trough_indices) >= 2:
            # Direnci çizgisi eğimi (son 2 tepe)
            resistance_slope = (work_slice['High'].iloc[peak_indices[-1]] - work_slice['High'].iloc[peak_indices[-2]]) / (peak_indices[-1] - peak_indices[-2])
            
            # Destek çizgisi eğimi (son 2 dip)
            support_slope = (work_slice['Low'].iloc[trough_indices[-1]] - work_slice['Low'].iloc[trough_indices[-2]]) / (trough_indices[-1] - trough_indices[-2])
            
            current_price = df['Close'].iloc[-1]
            
            # A. YÜKSELEN ÜÇGEN (Ascending Triangle)
            if abs(resistance_slope) < 0.001 and support_slope > 0.001:
                patterns.append({
                    "type": "triangle",
                    "name": "Yükselen Üçgen ▲",
                    "color": "green",
                    "x0": work_slice.index[peak_indices[-2]],
                    "y0": work_slice['High'].iloc[peak_indices[-2]],
                    "x1": work_slice.index[-1],
                    "y1": work_slice['High'].iloc[peak_indices[-1]],
                    "direction": "BULLISH",
                    "target": current_price * 1.08,
                    "confidence": 75
                })
            
            # B. DÜŞEN ÜÇGEN (Descending Triangle)
            elif abs(support_slope) < 0.001 and resistance_slope < -0.001:
                patterns.append({
                    "type": "triangle",
                    "name": "Düşen Üçgen ▼",
                    "color": "red",
                    "x0": work_slice.index[trough_indices[-2]],
                    "y0": work_slice['Low'].iloc[trough_indices[-2]],
                    "x1": work_slice.index[-1],
                    "y1": work_slice['Low'].iloc[trough_indices[-1]],
                    "direction": "BEARISH",
                    "target": current_price * 0.92,
                    "confidence": 75
                })
            
            # C. SİMETRİK ÜÇGEN (Symmetrical Triangle)
            elif resistance_slope < -0.001 and support_slope > 0.001:
                patterns.append({
                    "type": "triangle",
                    "name": "Simetrik Üçgen ◇",
                    "color": "yellow",
                    "x0": work_slice.index[peak_indices[-2]],
                    "y0": work_slice['High'].iloc[peak_indices[-2]],
                    "x1": work_slice.index[-1],
                    "y1": work_slice['Low'].iloc[trough_indices[-1]],
                    "direction": "NEUTRAL",
                    "target": current_price,
                    "confidence": 60
                })
    except Exception as e:
        print(f"Üçgen tespit hatası: {e}")
    
    # =========================================
    # 2. ABCD HARMONİK FORMASYON
    # =========================================
    try:
        # ABCD: 4 noktalı klasik harmonik formasyon
        # A -> B (Hareket), B -> C (Düzeltme %38-88), C -> D (Hedef)
        
        all_peaks = argrelextrema(highs, np.greater, order=10)[0]
        all_troughs = argrelextrema(lows, np.less, order=10)[0]
        
        # Son 4 pivot noktayı bul
        pivots = []
        for i in range(len(df)):
            if i in all_peaks:
                pivots.append({'idx': i, 'price': highs[i], 'type': 'high'})
            elif i in all_troughs:
                pivots.append({'idx': i, 'price': lows[i], 'type': 'low'})
        
        if len(pivots) >= 4:
            # Son 4 pivot
            A, B, C, D_candidate = pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            
            # ABCD kuralları:
            # 1. A-B-C-D sırası: high-low-high-low veya low-high-low-high
            # 2. BC retracement: AB'nin %38-88'i
            # 3. CD projection: BC'nin %127-161.8'i
            
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            
            if AB > 0:
                BC_retracement = BC / AB
                
                # Fibonacci retracement aralığında mı?
                if 0.382 <= BC_retracement <= 0.886:
                    # ABCD tespiti
                    CD_projection = BC * 1.272  # Fibonacci 127.2%
                    
                    if A['type'] == 'low' and B['type'] == 'high':  # Bullish ABCD
                        target_D = C['price'] - CD_projection
                        
                        patterns.append({
                            "type": "harmonic",
                            "name": "ABCD Boğa 🦬",
                            "color": "cyan",
                            "points": [A, B, C, D_candidate],
                            "direction": "BULLISH",
                            "target": target_D,
                            "confidence": 80
                        })
                    
                    elif A['type'] == 'high' and B['type'] == 'low':  # Bearish ABCD
                        target_D = C['price'] + CD_projection
                        
                        patterns.append({
                            "type": "harmonic",
                            "name": "ABCD Ayı 🐻",
                            "color": "magenta",
                            "points": [A, B, C, D_candidate],
                            "direction": "BEARISH",
                            "target": target_D,
                            "confidence": 80
                        })
    except Exception as e:
        print(f"ABCD tespit hatası: {e}")
    
    # =========================================
    # 3. KELEBEK (BUTTERFLY) HARMONİK
    # =========================================
    try:
        # Kelebek: 5 noktalı (X-A-B-C-D)
        # Fibonacci oranları:
        # - AB = XA'nın %78.6'sı
        # - BC = AB'nin %38.2-88.6'sı
        # - CD = BC'nin %161.8-261.8'i
        # - D noktası X'i geçer (%127-161.8)
        
        if len(pivots) >= 5:
            X, A, B, C, D_cand = pivots[-5], pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            
            XA = abs(A['price'] - X['price'])
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D_cand['price'] - C['price'])
            
            if XA > 0 and AB > 0 and BC > 0:
                AB_ret = AB / XA
                BC_ret = BC / AB
                CD_ext = CD / BC
                XD_ext = abs(D_cand['price'] - X['price']) / XA
                
                # Kelebek Fibonacci kriterleri
                if (0.75 <= AB_ret <= 0.82 and 
                    0.35 <= BC_ret <= 0.90 and 
                    1.5 <= CD_ext <= 2.7 and
                    1.2 <= XD_ext <= 1.65):
                    
                    patterns.append({
                        "type": "harmonic",
                        "name": "Kelebek 🦋",
                        "color": "purple",
                        "points": [X, A, B, C, D_cand],
                        "direction": "REVERSAL",
                        "target": X['price'],
                        "confidence": 85
                    })
    except Exception as e:
        print(f"Kelebek tespit hatası: {e}")
    
    # =========================================
    # 4. BAŞ-OMUZ (HEAD & SHOULDERS)
    # =========================================
    try:
        # 3 tepe noktası: Sol Omuz - Baş - Sağ Omuz
        # Boyun çizgisi (neckline): İki dip arasındaki destek
        
        if len(all_peaks) >= 3 and len(all_troughs) >= 2:
            # Son 3 tepe
            left_shoulder_idx = all_peaks[-3]
            head_idx = all_peaks[-2]
            right_shoulder_idx = all_peaks[-1]
            
            # Aralarındaki dipler
            left_trough_idx = all_troughs[-2]
            right_trough_idx = all_troughs[-1]
            
            left_shoulder = highs[left_shoulder_idx]
            head = highs[head_idx]
            right_shoulder = highs[right_shoulder_idx]
            
            left_trough = lows[left_trough_idx]
            right_trough = lows[right_trough_idx]
            
            # Baş-Omuz kriterleri:
            # 1. Baş > Sol Omuz ve Baş > Sağ Omuz
            # 2. Sol Omuz ≈ Sağ Omuz (%10 tolerans)
            # 3. Neckline neredeyse yatay
            
            if (head > left_shoulder and head > right_shoulder and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.10 and
                abs(left_trough - right_trough) / left_trough < 0.05):
                
                neckline = (left_trough + right_trough) / 2
                target = neckline - (head - neckline)
                
                patterns.append({
                    "type": "reversal",
                    "name": "Baş-Omuz 👤",
                    "color": "red",
                    "x0": dates[left_shoulder_idx],
                    "y0": left_shoulder,
                    "x1": dates[right_shoulder_idx],
                    "y1": right_shoulder,
                    "neckline": neckline,
                    "direction": "BEARISH",
                    "target": target,
                    "confidence": 90
                })
    except Exception as e:
        print(f"Baş-Omuz tespit hatası: {e}")
    
    # =========================================
    # 5. BAYRAK (FLAG) FORMASYONU
    # =========================================
    try:
        # Bayrak: Güçlü trend sonrası kısa konsolidasyon
        # Sert yükseliş -> Dar kanal (bayrak direği + bayrak)
        
        # Son 30 mumda trend kontrolü
        recent_slice = df.tail(30)
        
        # Son 10 mum önceki 20 mumun max/min'ini geçtiyse bayrak olabilir
        pole_high = recent_slice['High'].iloc[:-10].max()
        pole_low = recent_slice['Low'].iloc[:-10].min()
        
        flag_highs = recent_slice['High'].iloc[-10:]
        flag_lows = recent_slice['Low'].iloc[-10:]
        
        # Bayrak genişliği (dar kanal)
        flag_width = (flag_highs.max() - flag_lows.min()) / pole_high
        
        # Direk yüksekliği
        pole_height = (pole_high - pole_low) / pole_low
        
        if pole_height > 0.05 and flag_width < 0.03:  # Direk %5+, Bayrak %3-
            patterns.append({
                "type": "continuation",
                "name": "Boğa Bayrağı 🚩",
                "color": "lime",
                "x0": recent_slice.index[-10],
                "y0": flag_lows.min(),
                "x1": recent_slice.index[-1],
                "y1": flag_highs.max(),
                "direction": "BULLISH",
                "target": pole_high + pole_height * pole_high,
                "confidence": 70
            })
    except Exception as e:
        print(f"Bayrak tespit hatası: {e}")
    
    # =========================================
    # 6. KAMA (WEDGE) FORMASYONU
    # =========================================
    try:
        # Yükselen Kama: Her iki çizgi de yukarı ama daralıyor (Bearish)
        # Düşen Kama: Her iki çizgi de aşağı ama daralıyor (Bullish)
        
        window_wedge = min(40, len(df))
        wedge_slice = df.tail(window_wedge)
        
        wedge_peaks = argrelextrema(wedge_slice['High'].values, np.greater, order=5)[0]
        wedge_troughs = argrelextrema(wedge_slice['Low'].values, np.less, order=5)[0]
        
        if len(wedge_peaks) >= 2 and len(wedge_troughs) >= 2:
            upper_slope = (wedge_slice['High'].iloc[wedge_peaks[-1]] - wedge_slice['High'].iloc[wedge_peaks[0]]) / len(wedge_peaks)
            lower_slope = (wedge_slice['Low'].iloc[wedge_troughs[-1]] - wedge_slice['Low'].iloc[wedge_troughs[0]]) / len(wedge_troughs)
            
            # Yükselen Kama (Bearish reversal)
            if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope * 1.5:
                patterns.append({
                    "type": "wedge",
                    "name": "Yükselen Kama ⬆️📐",
                    "color": "orange",
                    "direction": "BEARISH",
                    "confidence": 65
                })
            
            # Düşen Kama (Bullish reversal)
            elif upper_slope < 0 and lower_slope < 0 and abs(upper_slope) < abs(lower_slope) * 1.5:
                patterns.append({
                    "type": "wedge",
                    "name": "Düşen Kama ⬇️📐",
                    "color": "lightgreen",
                    "direction": "BULLISH",
                    "confidence": 65
                })
    except Exception as e:
        print(f"Kama tespit hatası: {e}")
    
    return patterns
 # MEVCUT detect_patterns ÇAĞRINIZDAN SONRA
# ========================================
# 🆕 YENİ EKLEME 8: Backtest Motor
# ========================================
def run_strategy_backtest(df, initial_balance=10000):
    """
    EKLENECEK YER: detect_patterns fonksiyonundan SONRA, send_tg fonksiyonundan ÖNCE
    
    Basit bir backtest motoru - Mevcut sinyal sisteminizi test eder
    """
    balance = initial_balance
    position = None
    trades = []
    equity_curve = []
    
    # En az 100 mum gerekli
    if len(df) < 100:
        return None
    
    # İlk 50 mumu eğitim için kullan, gerisi test
    for i in range(50, len(df)):
        current_slice = df.iloc[:i]
        row = df.iloc[i]
        price = row['Close']
        date = df.index[i]
        
        # S/R hesapla
        supports, resistances = calculate_sr_advanced(current_slice, "1d")
        
        # Sinyal al
        signal, color, _ = calculate_oracle_signal_v2(current_slice, supports, resistances)
        
        # Pozisyon yoksa ve AL sinyali varsa
        if position is None and "AL" in signal and balance > 0:
            qty = (balance * 0.95) / price  # %95'ini kullan
            position = {
                'entry': price,
                'entry_date': date,
                'qty': qty,
                'type': 'LONG'
            }
            balance -= (qty * price)
        
        # Pozisyon varsa ve SAT sinyali veya stop kontrolü
        elif position is not None:
            # Stop/TP hesapla
            atr = current_slice['ATR'].iloc[-1] if 'ATR' in current_slice.columns else price * 0.02
            tp = position['entry'] + (atr * 2.5)
            sl = position['entry'] - (atr * 1.5)
            
            should_close = False
            close_reason = ""
            
            if "SAT" in signal:
                should_close = True
                close_reason = "Sinyal"
            elif price >= tp:
                should_close = True
                close_reason = "TP"
            elif price <= sl:
                should_close = True
                close_reason = "SL"
            
            if should_close:
                pnl = (price - position['entry']) * position['qty']
                balance += (position['qty'] * price)
                
                trades.append({
                    'entry': position['entry'],
                    'exit': price,
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position['entry'] * position['qty'])) * 100,
                    'reason': close_reason
                })
                
                position = None
        
        # Equity curve kaydet
        current_equity = balance
        if position is not None:
            current_equity += (position['qty'] * price)
        equity_curve.append({'date': date, 'equity': current_equity})
    
    # Açık pozisyon varsa kapat
    if position is not None:
        final_price = df['Close'].iloc[-1]
        pnl = (final_price - position['entry']) * position['qty']
        balance += (position['qty'] * final_price)
        trades.append({
            'entry': position['entry'],
            'exit': final_price,
            'pnl': pnl,
            'pnl_pct': (pnl / (position['entry'] * position['qty'])) * 100,
            'reason': 'Final'
        })
    
    # İstatistikler
    if len(trades) == 0:
        return None
    
    total_return = ((balance - initial_balance) / initial_balance) * 100
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades and sum([t['pnl'] for t in losing_trades]) != 0 else 0
    
    return {
        'final_balance': balance,
        'total_return': total_return,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'trades': trades,
        'equity_curve': equity_curve
    }
    
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

# 📂 VARLIK YÖNETİMİ (DÜZELTİLMİŞ)
# ==========================================
# Varlık ekleyince listenin anında güncellenmesi için selectbox'tan ÖNCE burayı çalıştırıyoruz.

with st.sidebar.expander("➕ Varlık Yönetimi", expanded=False):
    st.info("Listeye yeni Coin, Hisse veya Emtia ekleyin.")
    
    # 1. YENİ VARLIK EKLEME
    with st.form("add_asset_form"):
        new_name = st.text_input("Görünen İsim (Örn: Pound)")
        # Kullanıcıyı uyarmak için placeholder ekledik
        new_symbol = st.text_input("Yahoo Kodu (Örn: GBPUSD=X)")
        submitted = st.form_submit_button("Listeye Ekle")
        
        if submitted:
            if new_name and new_symbol:
                # Session State güncelle
                st.session_state['coin_map'][new_name] = new_symbol
                # Dosyaya kaydet
                save_assets(st.session_state['coin_map'])
                # Global değişkeni anında güncelle ki aşağıda görünsün
                COIN_MAP = st.session_state['coin_map'] 
                st.success(f"{new_name} eklendi!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("İsim ve Sembol boş olamaz!")
    
    # 2. VARLIK SİLME
    st.write("---")
    # Silme listesini session state'den alıyoruz
    del_asset = st.selectbox("Silinecek Varlık", list(st.session_state['coin_map'].keys()), key="del_box")
    
    if st.button("Seçileni Sil"):
        if del_asset in st.session_state['coin_map']:
            if len(st.session_state['coin_map']) > 1:
                del st.session_state['coin_map'][del_asset]
                save_assets(st.session_state['coin_map'])
                COIN_MAP = st.session_state['coin_map'] # Anında güncelleme
                st.warning(f"{del_asset} silindi.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Listede en az 1 varlık kalmalı!")

    # 3. VARSAYILANA DÖN
    if st.button("🔄 Listeyi Sıfırla"):
        st.session_state['coin_map'] = DEFAULT_COIN_MAP.copy()
        save_assets(st.session_state['coin_map'])
        st.rerun()

st.sidebar.divider()
# ==========================================
# 📡 KAYNAK VE ENSTRÜMAN SEÇİMİ (KRİTİK YER)
# ==========================================
# Buradaki COIN_MAP artık yukarıda güncellenen session_state verisini kullanır.

# Güncel listeyi al
current_assets = list(st.session_state['coin_map'].keys())

src_pref = st.sidebar.radio("📡 Kaynak:", ["Binance", "OKX", "Yahoo Finance"])

# Eğer liste boşsa hata vermesin diye kontrol
if not current_assets:
    current_assets = ["Bitcoin (BTC)"]

sel_c = st.sidebar.selectbox("Enstrüman:", current_assets)

# Seçilen varlığın kodunu haritadan çek
symbol = st.session_state['coin_map'].get(sel_c, "BTC-USD")

st.sidebar.divider()
# 2. NORMAL AYARLAR

show_cloud = st.sidebar.checkbox("☁️ Destek/Direnç Bulutu", value=True)
show_ai = st.sidebar.checkbox("🤖 AI Trend", value=True)
show_pred = st.sidebar.checkbox("🔮 AI Tahmin", value=True)

st.sidebar.subheader("🔍 Filtreler")
show_all_pats = st.sidebar.checkbox("Hepsini Aç/Kapat", value=True)
f_wm = st.sidebar.checkbox("- W ve M", value=True)
f_candle = st.sidebar.checkbox("- Mumlar", value=True)

# Otomatik Bot kutusu kolay erişim için dışarıda kalsın
auto = st.sidebar.checkbox("Otomatik Bot")

with st.sidebar.expander("🔄 Çoklu Dilim Onayı", expanded=False):
    if st.button("Analiz Et"):
        confirmation, details = multi_timeframe_confirmation(sel_c, symbol)
        st.markdown(f"### {confirmation}")
        for tf, sig in details.items():
            color = "green" if "AL" in sig else ("red" if "SAT" in sig else "gray")
            st.markdown(f"**{tf}:** <span style='color:{color}'>{sig}</span>", unsafe_allow_html=True)

# ========================================
# 🆕 YENİ UI ELEMENT 2: Fear & Greed
# ========================================
try:
    fg_value, fg_class = get_fear_greed_index()
    fg_color = "green" if fg_value < 30 else ("red" if fg_value > 70 else "orange")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**😱 Piyasa Duygusu:** <span style='color:{fg_color}'>{fg_class} ({fg_value})</span>", unsafe_allow_html=True)
except:
    pass
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
        s_list, r_list = calculate_sr_advanced(df, tf)
        status, color, target_msg = calculate_oracle_signal_v2(df, s_list, r_list)
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {label}")
        st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{status}</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"{target_msg}")
        adv_patterns = detect_advanced_patterns(df)
    
        if adv_patterns:
            st.sidebar.markdown("**🔍 Tespit Edilen Formasyonlar:**")
            for pat in adv_patterns:
              emoji_dir = "🟢" if pat['direction'] == 'BULLISH' else ("🔴" if pat['direction'] == 'BEARISH' else "⚪")
              st.sidebar.caption(f"{emoji_dir} {pat['name']} (Güven: %{pat['confidence']})")
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
        # Fonksiyon artık 3 değer döndürüyor, üçünü de karşılıyoruz
        f_dates, f_prices, ai_score = calculate_smart_prediction_FIXED(df_view)
        
        if len(f_dates) > 0:
            # Tahmin Çizgisi
            fig.add_trace(go.Scatter(
                x=[df_view.index[-1]]+f_dates, 
                y=[df_view['Close'].iloc[-1]]+list(f_prices), 
                mode='lines', 
                line=dict(color='yellow', width=2, dash='dash'), 
                name=f'AI Tahmini (Güven: %{ai_score:.0f})'
            ))
            
            # Ekrana Bilgi Notu (Grafiğin üstüne veya altına)
            if ai_score > 70:
                st.success(f"🧠 **AI Güven Skoru:** %{ai_score:.1f} (Model bu coini çok iyi tanıyor!)")
            elif ai_score > 40:
                st.warning(f"🧠 **AI Güven Skoru:** %{ai_score:.1f} (Tahminler orta güvenilirlikte)")
            else:
                st.error(f"🧠 **AI Güven Skoru:** %{ai_score:.1f} (Piyasa çok belirsiz, AI zorlanıyor)")
             
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
# Gelişmiş formasyonları da tespit et
advanced_items = detect_advanced_patterns(df_view)

# Grafiğe çiz
       try:
       for adv in advanced_items:
           if adv['type'] == 'triangle':
               # Üçgen çizgisi
               fig.add_shape(
                   type="line",
                   x0=adv['x0'], y0=adv['y0'],
                   x1=adv['x1'], y1=adv['y1'],
                   line=dict(color=adv['color'], width=3, dash='dot')
               )
               
               # Hedef çizgisi
               fig.add_hline(
                   y=adv['target'],
                   line_dash="dashdot",
                   line_color=adv['color'],
                   annotation_text=f"🎯 {adv['name']}"
               )
           
           elif adv['type'] == 'harmonic':
               # ABCD noktalarını çiz
               points = adv['points']
               for i in range(len(points) - 1):
                   fig.add_shape(
                       type="line",
                       x0=df_view.index[points[i]['idx']],
                       y0=points[i]['price'],
                       x1=df_view.index[points[i+1]['idx']],
                       y1=points[i+1]['price'],
                       line=dict(color=adv['color'], width=2)
                   )
               
               # İsimlendirme
               fig.add_annotation(
                   x=df_view.index[points[2]['idx']],
                   y=points[2]['price'],
                   text=adv['name'],
                   showarrow=True,
                   arrowhead=2,
                   bgcolor=adv['color'],
                   font=dict(color='white')
               )
           
           elif adv['type'] == 'reversal':  # Baş-Omuz
               # Neckline çiz
               fig.add_hline(
                   y=adv['neckline'],
                   line_dash="solid",
                   line_color=adv['color'],
                   annotation_text="Boyun Çizgisi"
               )
               
               # Hedef
               fig.add_hline(
                   y=adv['target'],
                   line_dash="dot",
                   line_color="red",
                   annotation_text=f"🎯 {adv['name']}"
               )
           
           elif adv['type'] in ['continuation', 'wedge']:
               # Basit kutu gösterimi
               fig.add_annotation(
                   x=df_view.index[-5],
                   y=df_view['High'].iloc[-5],
                   text=adv['name'],
                   showarrow=False,
                   bgcolor=adv['color'],
                   font=dict(size=12, color='black')
               )
    s_list, r_list = calculate_sr_advanced(df_view, view_tf)
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
        signal_status, color, target_msg = calculate_oracle_signal_v2(df_view, s_list, r_list)
        
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
            
if df_view is not None:
    st.divider()
    
    with st.expander("📊 Backtest: Strateji Performansı", expanded=False):
        st.info("Mevcut sinyal sisteminizi geçmiş veride test eder. Gerçek sonuçları yansıtır.")
        
        if st.button("🚀 Backtest Başlat"):
            with st.spinner("Backtest çalışıyor..."):
                results = run_strategy_backtest(df_view, initial_balance=10000)
                
                if results is None:
                    st.warning("Yeterli işlem oluşmadı. Daha uzun veri gerekebilir.")
                else:
                    # Metrikler
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Toplam Getiri", f"%{results['total_return']:.2f}")
                    col2.metric("Kazanma Oranı", f"%{results['win_rate']:.1f}")
                    col3.metric("Toplam İşlem", results['total_trades'])
                    col4.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                    
                    # Detaylar
                    st.divider()
                    col_det1, col_det2 = st.columns(2)
                    col_det1.write(f"✅ Kazanan İşlem: {results['winning_trades']}")
                    col_det1.write(f"💰 Ort. Kazanç: ${results['avg_win']:.2f}")
                    col_det2.write(f"❌ Kaybeden İşlem: {results['losing_trades']}")
                    col_det2.write(f"💸 Ort. Kayıp: ${results['avg_loss']:.2f}")
                    
                    # Equity Curve Grafiği
                    st.divider()
                    st.subheader("📈 Sermaye Eğrisi")
                    
                    eq_df = pd.DataFrame(results['equity_curve'])
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq_df['date'], 
                        y=eq_df['equity'],
                        mode='lines',
                        name='Sermaye',
                        line=dict(color='cyan', width=2)
                    ))
                    fig_eq.add_hline(y=10000, line_dash="dot", line_color="gray", annotation_text="Başlangıç")
                    fig_eq.update_layout(
                        height=400,
                        template="plotly_dark",
                        hovermode='x unified',
                        yaxis_title="Bakiye ($)",
                        xaxis_title="Tarih"
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                    # İşlem Listesi
                    with st.expander("📋 Tüm İşlemler"):
                        trades_df = pd.DataFrame(results['trades'])
                        st.dataframe(trades_df, use_container_width=True)
                        
# --- CÜZDAN (NAKİT YÖNETİMİ & AI TARAYICI) ---
    st.divider()
    st.header("💼 Varlık ve Fırsat Yönetimi")
    
# --- YENİ ÖZELLİK: AI PİYASA TARAYICI (BAKİYE ÖNERİLİ) ---
    # --- YENİ ÖZELLİK: AI PİYASA TARAYICI (GÜNCELLENMİŞ + RENKLENDİRİLMİŞ) ---
# --- YENİ ÖZELLİK: AI PİYASA TARAYICI (GÜNCELLENMİŞ + RENKLENDİRİLMİŞ) ---
    with st.expander("🔍 Piyasayı Tara & Fırsat Bul (AI)", expanded=False):
        st.info("Bu modül RSI, Trend ve Destek/Direnç analizi yaparak kasa yönetimi önerir.")
    
        scan_tf = st.radio("Tarama Periyodu:", ["4h", "1d", "1wk"], horizontal=True, format_func=lambda x: intervals[x])
    
        if st.button("🚀 Taramayı Başlat"):
            best_opp = None
            best_score = -100
        
            progress_bar = st.progress(0)
            status_text = st.empty()
         # 🔥 DÜZELTME BURADA: Listeyi doğrudan session_state'den (güncel hafızadan) çekiyoruz
            if 'coin_map' in st.session_state:
                current_map = st.session_state['coin_map']
            else:
                current_map = COIN_MAP # Yedek
            coins_to_scan = list(current_map.keys())
            total_coins = len(coins_to_scan)
        
            results_scan = []

            for i, c_name in enumerate(coins_to_scan):
                status_text.text(f"Analiz ediliyor: {c_name}...")
                progress_bar.progress((i + 1) / total_coins)
            
                sym = current_map[c_name]
                d_scan, _ = get_market_data("Yahoo Finance", sym, scan_tf)
            
                if d_scan is not None and len(d_scan) > 20:
                    last_price = d_scan['Close'].iloc[-1]
                    last_rsi = d_scan['RSI'].iloc[-1]
                    ema_50 = d_scan['EMA_50'].iloc[-1]
                
                # Destek/Direnç
                    sup_list, res_list = calculate_sr_advanced(d_scan, scan_tf)
                    nearest_sup = max([s for s in sup_list if s < last_price], default=0)
                
                # === ANA SİNYAL SİSTEMİ ===
                    main_signal, main_color, target_msg = calculate_oracle_signal_v2(d_scan, sup_list, res_list)
                
                # --- TARAYICI PUANLAMASI ---
                    score = 50 
                    reasons = []
                
                    # RSI
                    if last_rsi < 35: score += 20; reasons.append("RSI Dip")
                    elif last_rsi > 65: score -= 20; reasons.append("RSI Zirve")
                
                    # Trend
                    if last_price > ema_50: 
                        score += 10
                        reasons.append("Uptrend")
                    else:
                        reasons.append("Downtrend")
                    
                    # Destek
                    if nearest_sup > 0 and (last_price - nearest_sup)/last_price < 0.03:
                        score += 25
                        reasons.append("Desteğe Yakın")
                    
                    # --- BAKİYE YÖNETİMİ ÖNERİSİ ---
                    alloc_pct = 0
                    if score >= 85: alloc_pct = 10
                    elif score >= 70: alloc_pct = 5
                    elif score >= 60: alloc_pct = 2.5
                
                    # ======================================================
                    # 🎨 GÖRSEL İYİLEŞTİRME: Emoji Ekle (Kolay Okuma)
                    # ======================================================
                    if "GÜÇLÜ AL" in main_signal or "AL" in main_signal:
                        signal_display = f"🟢 {main_signal}"
                    elif "GÜÇLÜ SAT" in main_signal or "SAT" in main_signal:
                        signal_display = f"🔴 {main_signal}"
                    else:
                        signal_display = f"⚪ {main_signal}"
                    
                    results_scan.append({
                        "Coin": c_name, 
                        "Ana Sinyal": signal_display,  # Renkli emoji ile
                        "Tarayıcı Puanı": score,
                        "Önerilen Kasa (%)": f"%{alloc_pct}" if alloc_pct > 0 else "-",
                        "Fiyat": f"${last_price:,.2f}",  # Fiyatı da ekledik
                        "RSI": f"{last_rsi:.0f}",
                        "Sebep": ", ".join(reasons) if reasons else "Standart"
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_opp = results_scan[-1]

            progress_bar.empty()
            status_text.empty()
        
        # === EN İYİ FIRSAT KARTI ===
            if best_opp:
                st.success(f"🌟 **EN İYİ FIRSAT:** {best_opp['Coin']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tarayıcı Puanı", best_opp['Tarayıcı Puanı'])
                c2.metric("Ana Sinyal", best_opp['Ana Sinyal'])
                c3.metric("Fiyat", best_opp['Fiyat'])
                c4.metric("Önerilen Yatırım", best_opp['Önerilen Kasa (%)'])
                st.caption(f"📌 Sebep: {best_opp['Sebep']}")
            
            # === TABLO (Sıralı & Filtrelenebilir) ===
            st.divider()
            st.markdown("### 📊 Tüm Tarama Sonuçları")
            
            # Filtre seçeneği ekle
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                filter_signal = st.multiselect(
                    "Sinyal Filtresi", 
                    options=["AL", "SAT", "NÖTR"],
                    default=["AL", "SAT", "NÖTR"]
                )
            
            with col_f2:
                min_score = st.slider("Minimum Puan", 0, 100, 0)
            
            # DataFrame oluştur
            df_results = pd.DataFrame(results_scan)
            
            # Filtreleme
            filtered_df = df_results[
                (df_results['Ana Sinyal'].str.contains('|'.join(filter_signal))) &
                (df_results['Tarayıcı Puanı'] >= min_score)
            ].sort_values(by="Tarayıcı Puanı", ascending=False)
            
            # Tabloyu göster
            st.dataframe(
                filtered_df, 
                use_container_width=True,
                height=400  # Sabit yükseklik (scroll yapılabilir)
            )
            
            # ======================================================
            # 🆕 İSTATİSTİKLER (Hızlı Özet)
            # ======================================================
            st.divider()
            st.markdown("### 📈 Tarama İstatistikleri")
            
            total_analyzed = len(df_results)
            buy_signals = len(df_results[df_results['Ana Sinyal'].str.contains('AL')])
            sell_signals = len(df_results[df_results['Ana Sinyal'].str.contains('SAT')])
            neutral_signals = total_analyzed - buy_signals - sell_signals
            
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            stat_c1.metric("Toplam Analiz", total_analyzed)
            stat_c2.metric("🟢 AL Sinyali", buy_signals)
            stat_c3.metric("🔴 SAT Sinyali", sell_signals)
            stat_c4.metric("⚪ NÖTR", neutral_signals)
            
            # Yüzdelik dağılım
            if total_analyzed > 0:
                st.info(f"""
                **📊 Piyasa Dağılımı:**
                - AL Sinyali: %{(buy_signals/total_analyzed*100):.1f}
                - SAT Sinyali: %{(sell_signals/total_analyzed*100):.1f}
                - NÖTR: %{(neutral_signals/total_analyzed*100):.1f}
                """)
            
            # ======================================================
            # 🆕 CSV İNDİRME SEÇENEĞİ
            # ======================================================
            st.divider()
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Sonuçları CSV olarak İndir",
                data=csv,
                file_name=f"piyasa_tarama_{scan_tf}_{time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
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
            # 1. Risk Kontrolü
            is_valid, risk_msg = validate_portfolio_risk(
                investment, 
                current_balance, 
                st.session_state['portfolio_data']['positions']
            )

            if not is_valid:
                st.error(risk_msg)
                # Hata varsa burada dur, aşağıya geçme
            else:
                # 2. Bakiye Kontrolü ve İşlem
                proceed = True
                if use_balance:
                    if investment > current_balance:
                        st.error("Yetersiz Bakiye! Lütfen Bakiye Düzenle kısmından para ekleyin.")
                        proceed = False
                    else:
                        st.session_state['portfolio_data']['balance'] -= investment
                
                # 3. İşlemi Kaydet
                if proceed:
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
    # Bu satır (st.write) "with col_risk:" bloğunun hizasında olmalı (1 TAB içeride)
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
else: 
    st.error("Veri Alınamadı.")

if st.session_state.get('portfolio_data'):
    closed_count, closed_trades = check_active_positions_auto_close(st.session_state['portfolio_data'])
    
    if closed_count > 0:
        st.toast(f"🔔 {closed_count} pozisyon otomatik kapandı!", icon="✅")
        
        # Bildirimleri göster
        for trade in closed_trades:
            emoji = "✅" if trade['profit'] > 0 else "❌"
            st.sidebar.success(
                f"{emoji} {trade['coin']}: {trade['type']} | "
                f"${trade['profit']:.2f} (%{trade['pct']:.1f})"
            )
# BOT KISMI AYNEN KALACAK...

# BOT
if auto or st.session_state.get('auto_mode', False):
    msg = ""
    for tf, res in results.items():
        if res is not None:
            s_l, r_l = calculate_sr_advanced(res, tf)
            stat, _, target = calculate_oracle_signal_v2(res, s_l, r_l)
            if "GÜÇLÜ" in stat or "AL" in stat:
                msg += f"\n⏰ {tf}: {stat} | {target}"
    
    if msg and tg_token and tg_chat:
        full_msg = f"🚨 **{sel_c} BOT** 🚨\n{msg}\nFiyat: {curr:.2f}"
        if 'last_msg' not in st.session_state or st.session_state['last_msg'] != full_msg:
            send_tg(tg_token, tg_chat, full_msg)
            st.session_state['last_msg'] = full_msg
    
    time.sleep(14400) 
    st.rerun()











































