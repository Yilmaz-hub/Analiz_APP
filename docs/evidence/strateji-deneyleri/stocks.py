import json, pickle, sys, statistics as stt
import pandas as pd
sys.path.insert(0, "/home/user/Analiz_APP"); sys.path.insert(0, "exp")
from per_asset import run
from methods import donchian, rsi2, hold
import portfolio_backtest as pb, strategy_candidates as sc
SPLIT = pd.Timestamp("2022-07-01", tz="UTC")
US = ["QQQ","AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","COST","NFLX","AMD","ADBE","PEP","CSCO","INTC"]
TR = ["XU100","THYAO","ASELS","GARAN","AKBNK","YKBNK","ISCTR","KCHOL","SAHOL","BIMAS","EREGL","TUPRS","FROTO","SISE","TCELL","PGSUS"]
def load(s):
    df = pickle.load(open(f"exp/data_stk/{s}.pkl", "rb")); raw = json.load(open(f"exp/data_stk/{s}.dec.json"))
    v1 = {ts: raw[str(ts.date())] for ts in df.index if str(ts.date()) in raw}
    return df, v1, sc.candidate_decisions(df, v1, sc.trend_candidate("ABD" if s in US else "BIST"), None)
res = {}
for s in US + TR:
    df, v1, c = load(s); first = pb.first_tradable_day(df, list(v1))
    ms = {"S0 C": c, "S1 Kanal": donchian(df, v1), "S2 RSI2 dip": rsi2(df, v1)}
    dev_df = df[df.index < SPLIT]; ver_start = df.index[df.index >= SPLIT][0]
    res[s] = {k: (run(dev_df, d, first), run(df, d, ver_start)) for k, d in ms.items()}
    print(s, {k: tuple((round(x["ann"],1), round(x["dd"]), x["n"]) for x in v) for k, v in res[s].items()},
          "AT", tuple((round(x["h_ann"],1), round(x["h_dd"])) for x in res[s]["S0 C"]), flush=True)
pickle.dump(res, open("exp/stocks.pkl", "wb"))
def ratio(a, d): return a / abs(d) if d else 0
for market, syms in (("Nasdaq", US), ("BIST ($)", TR)):
    print(f"\n== {market} ({len(syms)}: endeks + {len(syms)-1} hisse) ==")
    for i, p in ((0, "geliştirme"), (1, "doğrulama ")):
        for k in ("S0 C", "S1 Kanal", "S2 RSI2 dip", "al-tut"):
            if k == "al-tut":
                xs = [(r["S0 C"][i]["h_ann"], r["S0 C"][i]["h_dd"]) for r in (res[s] for s in syms)]
            else:
                xs = [(r[k][i]["ann"], r[k][i]["dd"]) for r in (res[s] for s in syms)]
            print(f"  {p} {k:12} medyan yıl %{stt.median(a for a,_ in xs):6.1f}  medyan düş %{stt.median(d for _,d in xs):4.0f}  "
                  f"medyan getiri/düşüş {stt.median(ratio(a,d) for a,d in xs):5.2f}  >=%10: {sum(a>=10 for a,_ in xs)}/{len(xs)}")
        idx = syms[0]
        print(f"  {p} endeks {idx}: " + " | ".join(f"{k} %{res[idx][k][i]['ann']:.1f}/{res[idx][k][i]['dd']:.0f}" for k in ("S0 C","S1 Kanal","S2 RSI2 dip"))
              + f" | al-tut %{res[idx]['S0 C'][i]['h_ann']:.1f}/{res[idx]['S0 C'][i]['h_dd']:.0f}")
