import os, pickle, sys, time
import pandas as pd, requests
sys.path.insert(0, "/home/user/Analiz_APP")
from data_fetchers import process_data
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{HERE}/data_stk", exist_ok=True)
US = ["QQQ","AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","COST","NFLX","AMD","ADBE","PEP","CSCO","INTC"]
TR = ["XU100","THYAO","ASELS","GARAN","AKBNK","YKBNK","ISCTR","KCHOL","SAHOL","BIMAS","EREGL","TUPRS","FROTO","SISE","TCELL","PGSUS"]
p1 = int(pd.Timestamp("2014-01-01", tz="UTC").timestamp()); p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
def get(sym):
    for host in ("query2", "query1"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()["chart"]["result"][0]
            q = r["indicators"]["quote"][0]
            df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]},
                              index=pd.to_datetime(r["timestamp"], unit="s", utc=True).normalize()).dropna(subset=["Open","High","Low","Close"])
            df = df[~df.index.duplicated(keep="last")]
            return df[df.index < pd.Timestamp.now(tz="UTC").normalize()]
        except Exception as e:
            err = e; time.sleep(1)
    raise err
fx = get("USDTRY=X")["Close"]
print("USDTRY", len(fx), fx.index[0].date(), round(fx.iloc[-1], 2))
for sym in US + TR:
    df = get(sym if sym in US else sym + ".IS")
    if sym in TR:
        rate = fx.reindex(df.index).ffill().bfill()
        df[["Open","High","Low","Close"]] = df[["Open","High","Low","Close"]].div(rate, axis=0)
    df["Volume"] = df["Volume"].fillna(0)
    out = process_data(df, "Yahoo")[0]
    pickle.dump(out, open(f"{HERE}/data_stk/{sym}.pkl", "wb"))
    print(sym, len(out), out.index[0].date(), round(float(out["Close"].iloc[-1]), 2), flush=True)
    time.sleep(0.3)
