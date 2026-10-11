import json, os, pickle, sys
import pandas as pd, requests
sys.path.insert(0, "/home/user/Analiz_APP")
from data_fetchers import process_data
HERE = os.path.dirname(os.path.abspath(__file__))
p1 = int(pd.Timestamp("2016-01-01", tz="UTC").timestamp()); p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/GC=F?period1={p1}&period2={p2}&interval=1d",
                 headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()["chart"]["result"][0]
q = r["indicators"]["quote"][0]
df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]},
                  index=pd.to_datetime(r["timestamp"], unit="s", utc=True).normalize()).dropna()
df = df[df.index < pd.Timestamp.now(tz="UTC").normalize()]
df = df[~df.index.duplicated(keep="last")]
out = process_data(df, "Yahoo GC=F")[0]
pickle.dump(out, open(f"{HERE}/data_long/XAU_GOLD.pkl", "wb"))
print(len(out), out.index[0].date(), out.index[-1].date(), round(float(out['Close'].iloc[-1]),1))
