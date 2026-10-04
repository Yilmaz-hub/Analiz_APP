import time

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from config import DataFetchConfig, IndicatorConfig, Constants
from logger import logger
import market_map

# ==========================================
# VERİ MOTORLARI (KULLANICI TARAFI YÜKLEMELERİ İÇİN)
# ==========================================

@st.cache_data(ttl=DataFetchConfig.CACHE_TTL, show_spinner=False)
def fetch_binance_simple(symbol, interval, limit=1000):
    s_bin = market_map.binance_symbol(symbol) or symbol.replace("-", "")
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
                df = pd.DataFrame(data, columns=[
                    "OpenTime", "Open", "High", "Low", "Close", "Volume",
                    "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore",
                ])
                df["Date"] = pd.to_datetime(pd.to_numeric(df["OpenTime"]), unit='ms')
                df.set_index("Date", inplace=True)
                now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
                provider_open = set(df.index[pd.to_numeric(df["CloseTime"]) >= now_ms])
                result = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                result.attrs["provider_open"] = provider_open
                return result
        except Exception as e:
            logger.debug(f"Binance URL failed: {url}, Error: {e}")
            continue

    logger.error("Binance: Tüm URL'ler başarısız oldu.")
    return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_okx_simple(symbol, interval, limit=300):
    s_okx = market_map.okx_symbol(symbol) or symbol
    # V1 daily decisions share a UTC day boundary across crypto providers.
    omap = {"4h": "4H", "1d": "1Dutc", "1wk": "1W"}
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": s_okx, "bar": omap.get(interval, "1D"), "limit": limit}

    try:
        r = requests.get(url, params=params, headers=DataFetchConfig.HEADERS, timeout=5)
        data = r.json()
        if data.get('code') == '0':
            df = pd.DataFrame(data['data'], columns=[
                "ts", "Open", "High", "Low", "Close", "Volume",
                "VolumeCcy", "VolumeQuote", "Confirm",
            ])
            df["Date"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit='ms')
            df.set_index("Date", inplace=True)
            provider_open = set(df.index[df["Confirm"].astype(str) == "0"])
            result = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
            result.attrs["provider_open"] = provider_open
            return result
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
            source_attrs = dict(df.attrs)
            if "Volume" not in df.columns:
                df["Volume"] = 0
            from indicator_calculations import add_core_indicators
            df = add_core_indicators(df, IndicatorConfig).ffill()
            df.attrs.update(source_attrs)
            df["Source"] = src
            return df, src
        except Exception as e:
            logger.error(f"Process Error: {e}")
            return None, "ISLEME_HATASI"
    return None, "Yetersiz Veri"

@st.cache_data(ttl=DataFetchConfig.CACHE_TTL, show_spinner=False)
def get_market_data(source_pref, symbol, interval):
    symbol = market_map.canonical_symbol(symbol)  # `LINKUSD` → `LINK-USD` (spec 0006)
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

# Grafikle aynı Binance adresleri, aynı sırayla (spec 0006 R09): ilk adres bazı
# bölgelerde engelli olabilir; grafik çalışırken fiyat 0 kalmasın.
BINANCE_PRICE_HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.us",
    "https://api.binance.com",
)


#: Üç adres için toplam bekleme üst sınırı; ekran pozisyon sayısıyla çarpıldığı için kısa tutulur.
BINANCE_PRICE_BUDGET_SECONDS = 4.0


def _positive(value):
    """Sonlu ve sıfırdan büyük fiyatı döndürür; aksi halde None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number > 0 and number != float("inf") else None


#: Fiyat kaynağı tanılaması: sembol -> {kaynak: sonuç}. Yalnız ekranda "neden alınamadı" göstermek içindir.
PRICE_DIAGNOSTICS = {}
_OUTCOME_RANK = {"yanıt yok": 0, "reddedildi": 1, "geçersiz yanıt": 2, "alındı": 3}


def _note(symbol, source, outcome):
    """Kaynağın en iyi sonucunu saklar (alındı > geçersiz yanıt > reddedildi > yanıt yok)."""
    seen = PRICE_DIAGNOSTICS.setdefault(symbol, {})
    if _OUTCOME_RANK[outcome] >= _OUTCOME_RANK.get(seen.get(source), -1):
        seen[source] = outcome


def price_diagnostics(symbol):
    """Ekranda gösterilecek kısa satırlar; adres, istisna metni ve parola içermez (spec 0007 R07)."""
    seen = PRICE_DIAGNOSTICS.get(market_map.canonical_symbol(symbol), {})
    return [f"{source}: {outcome}" for source, outcome in seen.items()]


def _binance_price(symbol):
    pair = market_map.binance_symbol(symbol)
    if pair is None:
        return None
    started = time.monotonic()
    for host in BINANCE_PRICE_HOSTS:
        for path, params, pick in (
            ("/api/v3/ticker/price", {"symbol": pair}, lambda body: body.get("price")),
            ("/api/v3/klines", {"symbol": pair, "interval": "1m", "limit": 1}, lambda body: body[-1][4]),
        ):
            remaining = BINANCE_PRICE_BUDGET_SECONDS - (time.monotonic() - started)
            if remaining <= 0.2:
                return None   # toplam süre aşıldı: kalan adresler denenmez, sıradaki kaynağa geçilir (R11)
            try:
                # Grafik istekleriyle aynı tarayıcı kimliği (R05); zaman aşımı kalan bütçeye sığar.
                r = requests.get(host + path, params=params, headers=DataFetchConfig.HEADERS,
                                 timeout=(min(1.5, remaining / 2), min(2.0, remaining / 2)))
                if r.status_code != 200:
                    _note(symbol, "Binance", "reddedildi")
                    if r.status_code == 451:
                        break   # bölge engeli adresin tamamı için geçerli: kline uç noktası da denenmez
                    continue
                try:
                    price = _positive(pick(r.json()))
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    price = None            # beklenmedik gövde: "yanıt yok" değil "geçersiz yanıt"
                if price is not None:
                    _note(symbol, "Binance", "alındı")
                    return price
                _note(symbol, "Binance", "geçersiz yanıt")
            except Exception as e:
                _note(symbol, "Binance", "yanıt yok")
                logger.debug(f"Binance price fetch failed ({host}{path}, {pair}): {e}")
    return None


def _okx_price(symbol):
    instrument = market_map.okx_symbol(symbol)
    if instrument is None:
        return None
    try:
        r = requests.get("https://www.okx.com/api/v5/market/ticker", params={"instId": instrument},
                         headers=DataFetchConfig.HEADERS, timeout=(1.0, OKX_PRICE_TIMEOUT_SECONDS - 1.0))
        if r.status_code != 200:
            _note(symbol, "OKX", "reddedildi")
            return None
        body = r.json()
        price = _positive(body["data"][0]["last"]) if body.get("code") == "0" else None
        _note(symbol, "OKX", "alındı" if price is not None else "geçersiz yanıt")
        return price
    except Exception as e:
        _note(symbol, "OKX", "yanıt yok")
        logger.debug(f"OKX price fetch failed ({instrument}): {e}")
        return None


def _yahoo_price(symbol):
    yahoo_symbol = symbol[:-1] if symbol.endswith("-USDT") else symbol   # Yahoo USDT çiftini tanımaz
    try:
        price = _positive(yf.Ticker(yahoo_symbol).fast_info['last_price'])
    except Exception as e:
        logger.debug(f"Yahoo price fetch failed for {symbol}: {e}")
        _note(symbol, "Yahoo", "yanıt yok")
        return None
    _note(symbol, "Yahoo", "alındı" if price is not None else "geçersiz yanıt")
    return price


def _live_price(ticker_symbol):
    """Kaynak sırası: Binance adresleri → OKX (yalnız kripto) → Yahoo (kanonik sembolle)."""
    canonical = market_map.canonical_symbol(ticker_symbol)
    PRICE_DIAGNOSTICS.pop(canonical, None)
    for source in (_binance_price, _okx_price, _yahoo_price):
        price = source(canonical)
        if price is not None:
            return price
    return 0


#: Bir çalıştırmada tüm pozisyon fiyatları için toplam bekleme üst sınırı (spec 0007 R03).
PRICE_BUDGET_SECONDS = 7.0
#: OKX isteğinin en kötü süresi; Binance bütçesiyle toplamı `PRICE_BUDGET_SECONDS`'ı aşmaz.
OKX_PRICE_TIMEOUT_SECONDS = 2.5


def fetch_prices(coin_names, coin_map, budget=None):
    """Pozisyon fiyatlarını eşzamanlı alır; süre bütçesini aşan fiyat 0 (alınamadı) döner.

    Sıralı çağrıda her kaynak ağ zaman aşımı kadar bekleyip sayfayı donduruyordu. Burada
    her coin ayrı iş parçacığında çalışır, tümü bütçe içinde beklenir; geç dönen fiyat
    arka planda tamamlanır ve sonraki çalıştırmada önbellekten gelir."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, wait

    names = list(dict.fromkeys(coin_names))
    if not names:
        return {}
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        context = get_script_run_ctx()
    except Exception:  # pragma: no cover - streamlit sürümüne bağlı yardımcı
        add_script_run_ctx, context = None, None

    def job(name):
        if add_script_run_ctx is not None and context is not None:
            add_script_run_ctx(threading.current_thread(), context)
        return get_live_price_for_portfolio(name, coin_map)

    pool = ThreadPoolExecutor(max_workers=len(names))
    futures = {name: pool.submit(job, name) for name in names}
    wait(futures.values(), timeout=PRICE_BUDGET_SECONDS if budget is None else budget)
    pool.shutdown(wait=False, cancel_futures=True)
    prices = {}
    for name, future in futures.items():
        try:
            prices[name] = future.result(timeout=0) if future.done() else 0
        except Exception as exc:
            logger.debug(f"fetch_prices failed for {name}: {exc}")
            prices[name] = 0
    return prices


@st.cache_data(ttl=30, show_spinner=False)
def get_live_price_for_portfolio(coin_name, coin_map):
    try:
        ticker_symbol = coin_map.get(coin_name)

        if ticker_symbol == "GRAM_TRY":
             ons_price = 0
             try: ons_price = yf.Ticker("GC=F").fast_info['last_price']
             except Exception as e: logger.debug(f"GC=F price fetch failed: {e}")

             if not ons_price:
                 try: ons_price = yf.Ticker("GC=F").fast_info['last_price']
                 except Exception as e: logger.debug(f"GC=F price retry failed: {e}")

             usd_price = 0
             try: usd_price = yf.Ticker("TRY=X").fast_info['last_price']
             except Exception as e: logger.debug(f"TRY=X price fetch failed: {e}")

             if not usd_price:
                 try: usd_price = yf.Ticker("USDTRY=X").fast_info['last_price']
                 except Exception as e: logger.debug(f"USDTRY=X price fetch failed: {e}")

             if ons_price and usd_price:
                 return (ons_price * usd_price) / Constants.OUNCE_TO_GRAMS
             return 0

        if ticker_symbol == "XAU_GOLD":
            try:
                price = yf.Ticker("GC=F").fast_info['last_price']
                if price and price > 0: return price
            except Exception as e:
                logger.debug(f"GC=F price fetch failed: {e}")
            try:
                return yf.Ticker("GC=F").fast_info['last_price']
            except Exception as e:
                logger.debug(f"GC=F price retry failed: {e}")
                return 0

        if not ticker_symbol: return 0
        return _live_price(ticker_symbol)
    except Exception as e:
        logger.debug(f"get_live_price_for_portfolio failed for {coin_name}: {e}")
        return 0
