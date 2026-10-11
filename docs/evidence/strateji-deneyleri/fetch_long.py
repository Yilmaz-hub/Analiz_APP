"""Binance günlük mumları 2018-01-01'den bugüne (sayfalı) indirir, uygulamanın process_data'sıyla işler."""
import os, pickle, sys, time
import pandas as pd, requests
sys.path.insert(0, "/home/user/Analiz_APP")
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT", "XRP-USD": "XRPUSDT",
         "AVAX-USD": "AVAXUSDT", "DOGE-USD": "DOGEUSDT", "LINK-USD": "LINKUSDT", "ADA-USD": "ADAUSDT"}
from data_fetchers import process_data
os.makedirs(f"{HERE}/data_long", exist_ok=True)
for sym, pair in PAIRS.items():
    rows, start = [], int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": pair, "interval": "1d", "startTime": start, "limit": 1000}, timeout=20)
        batch = r.json()
        if not batch: break
        rows += batch
        start = batch[-1][0] + 86400000
        if len(batch) < 1000: break
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume", *range(6)])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[df.index < pd.Timestamp.now(tz="UTC").normalize()]          # yalnız kapanmış günler
    out = process_data(df, "Binance")[0]
    pickle.dump(out, open(f"{HERE}/data_long/{sym}.pkl", "wb"))
    print(sym, len(out), out.index[0].date(), out.index[-1].date())
