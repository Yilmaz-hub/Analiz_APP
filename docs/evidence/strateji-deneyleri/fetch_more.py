"""Ek coinler: Binance günlük mumlar 2018+ (RENDER için eski RNDR çifti birleştirilir)."""
import os, pickle, sys, time
import pandas as pd, requests
sys.path.insert(0, "/home/user/Analiz_APP")
from data_fetchers import process_data
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = {"HBAR-USD": ["HBARUSDT"], "RENDER-USD": ["RNDRUSDT", "RENDERUSDT"], "BNB-USD": ["BNBUSDT"],
         "LTC-USD": ["LTCUSDT"], "DOT-USD": ["DOTUSDT"], "ATOM-USD": ["ATOMUSDT"], "NEAR-USD": ["NEARUSDT"],
         "TRX-USD": ["TRXUSDT"], "FET-USD": ["FETUSDT"], "INJ-USD": ["INJUSDT"], "SHIB-USD": ["SHIBUSDT"],
         "ALGO-USD": ["ALGOUSDT"], "VET-USD": ["VETUSDT"]}
def pull(pair):
    rows, start = [], int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
    while True:
        batch = requests.get("https://data-api.binance.vision/api/v3/klines",
                             params={"symbol": pair, "interval": "1d", "startTime": start, "limit": 1000}, timeout=20).json()
        if not isinstance(batch, list) or not batch: break
        rows += batch; start = batch[-1][0] + 86400000
        if len(batch) < 1000: break
        time.sleep(0.2)
    return rows
for sym, pairs in PAIRS.items():
    rows = []
    for p in pairs: rows += pull(p)
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume", *range(6)])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index < pd.Timestamp.now(tz="UTC").normalize()]
    out = process_data(df, "Binance")[0]
    pickle.dump(out, open(f"{HERE}/data_long/{sym}.pkl", "wb"))
    print(sym, len(out), out.index[0].date(), out.index[-1].date(), flush=True)
