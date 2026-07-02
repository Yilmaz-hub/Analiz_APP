import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import pandas_ta as ta
from config import DataFetchConfig, IndicatorConfig, Constants
from logger import logger

# ==========================================
# VERİ MOTORLARI (KULLANICI TARAFI YÜKLEMELERİ İÇİN)
# ==========================================

@st.cache_data(ttl=DataFetchConfig.CACHE_TTL, show_spinner=False)
def fetch_binance_simple(symbol, interval, limit=1000):
    s_bin = symbol.replace("-", "").replace("USD", "USDT")
    bmap = {"4h": "4h", "1d": "1d", "1wk": "1w"}
    b_interval = bmap.get(interval, "1d")
    base_urls = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines"
    ]
    params = {"symbol": s_bin, "interval": b_interval, "limit": limit}
    
    for url in base_urls:
        try:
            r = requests.get(url, params=params, headers=DataFetchConfig.HEADERS, timeout=3)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and 'code' in data: continue
                df = pd.DataFrame(data, columns=["OpT", "Open", "High", "Low", "Close", "Volume", "x", "x", "x", "x", "x", "x"])
                df["Date"] = pd.to_datetime(df["OpT"], unit='ms')
                df.set_index("Date", inplace=True)
                return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        except Exception as e:
            logger.debug(f"Binance URL failed: {url}, Error: {e}")
            continue
            
    logger.error("Binance: Tüm URL'ler başarısız oldu.")
    return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_okx_simple(symbol, interval, limit=300):
    s_okx = symbol.replace("USD", "USDT")
    omap = {"4h": "4H", "1d": "1D", "1wk": "1W"}
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": s_okx, "bar": omap.get(interval, "1D"), "limit": limit}
    
    try:
        r = requests.get(url, params=params, headers=DataFetchConfig.HEADERS, timeout=5)
        data = r.json()
        if data.get('code') == '0':
            df = pd.DataFrame(data['data'], columns=["ts", "Open", "High", "Low", "Close", "Volume", "x", "x", "x"])
            df["Date"] = pd.to_datetime(df["ts"], unit='ms')
            df.set_index("Date", inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df.sort_index()
    except Exception as e:
        logger.error(f"OKX Error: {e}")
        return None
    return None

@st.cache_data(ttl=DataFetchConfig.CACHE_TTL, show_spinner=False)
def fetch_yahoo_safe(symbol, interval):
    try:
        p = "10y" if interval == "1wk" else ("4y" if interval == "1d" else "1mo")
        i = "1h" if interval == "4h" else ("1d" if interval == "1d" else "1wk")
        
        df = yf.download(symbol, period=p, interval=i, progress=False, auto_adjust=True)
        if df.empty: return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.index.tz is not None: 
            df.index = df.index.tz_localize(None)
            
        if interval == "4h":
            agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            if 'Volume' not in df.columns: 
                df['Volume'] = 0
            df = df.resample('4h').agg(agg).dropna()
            
        return df
    except Exception as e:
        logger.error(f"Yahoo Error ({symbol}): {e}")
        return None

def fetch_yahoo_retry(tickers, interval):
    for sym in tickers:
        df = fetch_yahoo_safe(sym, interval)
        if df is not None and not df.empty and len(df) > 5:
            return df
    return None

@st.cache_data(ttl=DataFetchConfig.CACHE_TTL, show_spinner=False)
def fetch_gram_gold_calculated(interval):
    try:
        df_ons = fetch_yahoo_retry(["GC=F"], interval)
        df_usd = fetch_yahoo_retry(["TRY=X", "USDTRY=X"], interval)
        
        if df_ons is not None and df_usd is not None:
            df_ons = df_ons[['Close']].rename(columns={'Close': 'Ons'})
            df_usd = df_usd[['Close']].rename(columns={'Close': 'Usd'})
            
            df = df_ons.join(df_usd, how='inner')
            df['Close'] = (df['Ons'] * df['Usd']) / Constants.OUNCE_TO_GRAMS
            df['Open'] = df['Close']
            df['High'] = df['Close'] * Constants.GRAM_GOLD_HIGH_FACTOR
            df['Low'] = df['Close'] * Constants.GRAM_GOLD_LOW_FACTOR
            df['Volume'] = Constants.DEFAULT_GRAM_GOLD_VOLUME

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Gram Gold Calculation Error: {e}")
        return None
    return None

def process_data(df: pd.DataFrame, src: str):
    if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 10:
        try:
            if 'Volume' not in df.columns: df['Volume'] = 0
            
            rsi = df.ta.rsi(length=IndicatorConfig.RSI_LENGTH)
            df['RSI'] = rsi if rsi is not None else 50
            
            ema_50 = df.ta.ema(length=IndicatorConfig.EMA_LONG)
            df['EMA_50'] = ema_50 if ema_50 is not None else df['Close']
            
            ema_20 = df.ta.ema(length=IndicatorConfig.EMA_SHORT)
            df['EMA_20'] = ema_20 if ema_20 is not None else df['Close']

            bb = df.ta.bbands(length=IndicatorConfig.BOLLINGER_LENGTH, std=IndicatorConfig.BOLLINGER_STD)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                cols = df.columns
                df.rename(columns={cols[-5]: 'BB_Lower', cols[-3]: 'BB_Upper'}, inplace=True)

            atr = df.ta.atr(length=IndicatorConfig.ATR_LENGTH)
            df['ATR'] = atr if atr is not None else (df['Close'] * 0.02)
            
            adx_df = df.ta.adx(length=14)
            if adx_df is not None and not adx_df.empty:
                adx_col = [c for c in adx_df.columns if 'ADX' in c][0]
                df['ADX'] = adx_df[adx_col]
            else:
                df['ADX'] = 25
                
            df = df.bfill().ffill()
            return df, src
        except Exception as e:
            logger.error(f"Process Error: {e}")
            return None, "İşleme Hatası"
            
    return None, "Yetersiz Veri"

@st.cache_data(ttl=DataFetchConfig.CACHE_TTL, show_spinner=False)
def get_market_data(source_pref, symbol, interval):
    if symbol == "GRAM_TRY":
        df = fetch_gram_gold_calculated(interval)
        if df is not None: return process_data(df, "Hesaplamalı (Ons x Dolar)")
        return None, "Veri Hesaplanamadı"

    if symbol == "XAU_GOLD":
        df = fetch_yahoo_retry(["GC=F"], interval)
        if df is not None: return process_data(df, "Yahoo (Gold)")
        return None, "Veri Yok (Yahoo)"
    
    if symbol == "EURUSD=X":
        return process_data(fetch_yahoo_safe("EURUSD=X", interval), "Yahoo (Forex)")

    df = None
    src_name = ""
    
    if source_pref == "Binance":
        df = fetch_binance_simple(symbol, interval)
        src_name = "Binance"
    elif source_pref == "OKX":
        df = fetch_okx_simple(symbol, interval)
        src_name = "OKX"
    
    if df is None or df.empty:
        df = fetch_yahoo_safe(symbol, interval)
        src_name = "Yahoo (Yedek)"

    if df is None or df.empty: 
        return None, "Veri Alınamadı"
        
    return process_data(df, src_name)

@st.cache_data(ttl=3600, show_spinner=False)
def get_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        r = requests.get(url, timeout=5)
        data = r.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return value, classification
    except Exception as e:
        logger.warning(f"Failed to fetch Fear & Greed index: {e}. Using neutral default.")
        return 50, "Neutral"

@st.cache_data(ttl=30, show_spinner=False)
def get_live_price_for_portfolio(coin_name, coin_map):
    try:
        ticker_symbol = coin_map.get(coin_name)
        
        if ticker_symbol == "GRAM_TRY":
             ons_price = 0
             try: ons_price = yf.Ticker("GC=F").fast_info['last_price']
             except: pass
             
             if not ons_price:
                 try: ons_price = yf.Ticker("GC=F").fast_info['last_price']
                 except: pass
             
             usd_price = 0
             try: usd_price = yf.Ticker("TRY=X").fast_info['last_price']
             except: pass
             
             if not usd_price:
                 try: usd_price = yf.Ticker("USDTRY=X").fast_info['last_price']
                 except: pass

             if ons_price and usd_price:
                 return (ons_price * usd_price) / Constants.OUNCE_TO_GRAMS
             return 0

        if ticker_symbol == "XAU_GOLD": 
            try:
                price = yf.Ticker("GC=F").fast_info['last_price']
                if price and price > 0: return price
            except: pass
            try:
                return yf.Ticker("GC=F").fast_info['last_price']
            except: return 0
        
        if not ticker_symbol: return 0
                    # Try Binance first to avoid Yahoo rate limits for Crypto
        if "USD" in ticker_symbol and "XAU" not in ticker_symbol and "EUR" not in ticker_symbol:
            try:
                s_bin = ticker_symbol.replace("-", "").replace("USD", "USDT")
                r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={s_bin}", timeout=3)
                if r.status_code == 200:
                    return float(r.json()['price'])
            except: pass
        ticker = yf.Ticker(ticker_symbol)
        return ticker.fast_info['last_price']
    except: return 0
