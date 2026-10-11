import json, pickle, sys, statistics as stt
from decimal import Decimal
import pandas as pd
sys.path.insert(0, "/home/user/Analiz_APP"); sys.path.insert(0, "exp")
from per_asset import TIERS, load, run
import portfolio_backtest as pb
btc = pickle.load(open("exp/data_long/BTC-USD.pkl", "rb"))
btc_ok = (btc["Close"] > btc["Close"].rolling(200).mean())
def btc_filter(dec, exit_too):
    out = {}
    for ts, v in dec.items():
        ok = bool(btc_ok.get(ts, False))
        if v == "AL" and not ok: v = "BEKLE"
        if exit_too and not ok and v != "AL": v = "SAT"
        out[ts] = v
    return out
res = {}
for tier, bases in TIERS.items():
    for b in bases:
        df, v1, c = load(b)
        start = pb.first_tradable_day(df, list(v1))
        for window, st in (("tam", start), ("ortak", max(start, pd.Timestamp("2022-07-01", tz="UTC")))):
            if b == "XAU":
                if window == "ortak": st = df.index[df.index >= pd.Timestamp("2022-07-01", tz="UTC")][0]
                sma = df["Close"].rolling(200).mean()
                g1 = {ts: ("AL" if df["Close"][ts] > sma[ts] else "SAT") for ts in v1}
                r = {"C": run(df, c, st), "SMA200": run(df, g1, st)}
            elif b == "BTC":
                continue
            else:
                r = {"C": run(df, c, st), "C+BTCgiris": run(df, btc_filter(c, False), st),
                     "C+BTCgiris+cikis": run(df, btc_filter(c, True), st)}
            res[(tier, b, window)] = r
            print(tier, b, window, {k: (round(x["ann"],1), round(x["dd"]), x["n"]) for k, x in r.items()}, "AT", round(r["C"]["h_ann"],1), round(r["C"]["h_dd"]), flush=True)
for w in ("tam", "ortak"):
    for tier in ("Orta", "Küçük"):
        rs = [v for (t, b, ww), v in res.items() if t == tier and ww == w]
        for k in ("C", "C+BTCgiris", "C+BTCgiris+cikis"):
            print(w, tier, k, "medyan yıl", round(stt.median(x[k]["ann"] for x in rs),1), "medyan düş", round(stt.median(x[k]["dd"] for x in rs)),
                  ">=%10:", sum(x[k]["ann"] >= 10 for x in rs), "/", len(rs))
