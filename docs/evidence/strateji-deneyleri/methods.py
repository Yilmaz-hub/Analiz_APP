import json, pickle, sys, statistics as stt
import pandas as pd
sys.path.insert(0, "/home/user/Analiz_APP"); sys.path.insert(0, "exp")
from per_asset import TIERS, load, run
import portfolio_backtest as pb
SPLIT = pd.Timestamp("2022-07-01", tz="UTC")
btc = pickle.load(open("exp/data_long/BTC-USD.pkl", "rb"))["Close"]

def donchian(df, v1):
    hi = df["High"].rolling(50).max().shift(1); lo = df["Low"].rolling(20).min().shift(1)
    return {ts: ("AL" if df["Close"][ts] > hi[ts] else "SAT" if df["Close"][ts] < lo[ts] else "BEKLE")
            for ts in v1 if pd.notna(hi[ts])}
def ratio_ok(df):
    r = df["Close"] / btc.reindex(df.index)
    return r > r.rolling(100).mean()
def rel_strength(df, v1):
    sma = df["Close"].rolling(200).mean(); ok = ratio_ok(df)
    return {ts: ("AL" if df["Close"][ts] > sma[ts] and ok[ts] else "SAT") for ts in v1 if pd.notna(sma[ts])}
def c_rel(df, v1, c):
    ok = ratio_ok(df); out = {}
    for ts, v in c.items():
        if v == "AL" and not ok[ts]: v = "BEKLE"
        if v != "AL" and not ok[ts]: v = "SAT"
        out[ts] = v
    return out
def rsi2(df, v1):
    d = df["Close"].diff(); up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean(); rsi = 100 - 100 / (1 + up / dn)
    sma200, sma5 = df["Close"].rolling(200).mean(), df["Close"].rolling(5).mean()
    return {ts: ("AL" if df["Close"][ts] > sma200[ts] and rsi[ts] < 10 else "SAT" if df["Close"][ts] > sma5[ts] else "BEKLE")
            for ts in v1 if pd.notna(sma200[ts])}
def hold(df, v1):
    return {ts: "AL" for ts in v1}

def periods(df, dec, first):
    dev_df = df[df.index < SPLIT]
    dev = run(dev_df, dec, first) if first < SPLIT and (SPLIT - first).days >= 365 else None
    ver = run(df, dec, max(first, df.index[df.index >= SPLIT][0]))
    return dev, ver

res = {}
alts = TIERS["Orta"] + TIERS["Küçük"]
for b in alts + ["XAU"]:
    df, v1, c = load(b)
    first = pb.first_tradable_day(df, list(v1))
    if b == "XAU":
        ms = {"G0 C": c, "G2 RSI2 dip": rsi2(df, v1), "G3 Donchian": donchian(df, v1), "G1b her gün AL (stoplu)": hold(df, v1)}
    else:
        ms = {"M0 C": c, "M1 Donchian": donchian(df, v1), "M2 Göreli güç": rel_strength(df, v1), "M3 C+göreli": c_rel(df, v1, c)}
    res[b] = {k: periods(df, d, first) for k, d in ms.items()}
    print(b, {k: tuple(None if x is None else (round(x["ann"],1), round(x["dd"]), x["n"]) for x in v) for k, v in res[b].items()},
          "AT", tuple(None if x is None else (round(x["h_ann"],1), round(x["h_dd"])) for x in res[b]["M0 C" if b != "XAU" else "G0 C"]), flush=True)
pickle.dump(res, open("exp/methods.pkl", "wb"))
print()
for name, idx in (("GELİŞTİRME", 0), ("DOĞRULAMA", 1)):
    for k in ("M0 C", "M1 Donchian", "M2 Göreli güç", "M3 C+göreli"):
        xs = [res[b][k][idx] for b in alts if res[b][k][idx] is not None]
        print(f"{name:10} {k:15} n={len(xs):2} medyan yıl %{stt.median(x['ann'] for x in xs):6.1f}  medyan düş %{stt.median(x['dd'] for x in xs):4.0f}  "
              f">=%10: {sum(x['ann']>=10 for x in xs):2}/{len(xs)}  M0'ı geçen coin: {sum(x['ann'] > res[b]['M0 C'][idx]['ann'] for b in alts for x in [res[b][k][idx]] if x)}")
    xs = [res[b]["M0 C"][idx] for b in alts if res[b]["M0 C"][idx] is not None]
    print(f"{name:10} {'al-tut':15} n={len(xs):2} medyan yıl %{stt.median(x['h_ann'] for x in xs):6.1f}  medyan düş %{stt.median(x['h_dd'] for x in xs):4.0f}")
