import json, pickle, sys
from decimal import Decimal
import pandas as pd
sys.path.insert(0, "/home/user/Analiz_APP")
import portfolio_backtest as pb, strategy_candidates as sc
from technical_analysis import build_v1_decisions
from trade_execution import CostAssumptions
costs = CostAssumptions(Decimal(0), Decimal(0), Decimal("0.1"))
syms = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","AVAX-USD","LINK-USD","DOGE-USD"]
full, decs, v1s = {}, {}, {}
for s in syms:
    df = pickle.load(open(f"exp/data_long/{s}.pkl","rb")); raw = json.load(open(f"exp/data_long/{s}.dec.json"))
    v1 = {ts: raw[str(ts.date())] for ts in df.index if str(ts.date()) in raw}
    full[s], v1s[s] = df, v1; decs[s] = sc.candidate_decisions(df, v1, sc.trend_candidate("KRIPTO"), None)
g = pickle.load(open("exp/data_long/XAU_GOLD.pkl","rb")); gv = build_v1_decisions(g, include_ml=False)
full["XAU"], v1s["XAU"] = g, gv; decs["XAU"] = sc.candidate_decisions(g, gv, sc.trend_candidate("ALTIN"), None)
btc = full["BTC-USD"]; ok = btc["Close"] > btc["Close"].rolling(200).mean()
filt = {s: ({ts: ("BEKLE" if v == "AL" and not bool(ok.get(ts, False)) else v) for ts, v in d.items()}
            if s not in ("BTC-USD", "XAU") else d) for s, d in decs.items()}
def port(weights, dsets, start):
    fr = {s: full[s][full[s].index >= pd.Timestamp(start, tz="UTC")] for s in weights}
    return pb.run_portfolio(fr, {s: dsets[s] for s in weights}, weights, capital=Decimal("10000"), costs=costs,
                            quantity_steps={s: Decimal("0.00000001") for s in weights})
def stats(curve):
    s = pb.summary(curve); return f"yıllık %{float(s['annual']):5.1f} düşüş %{float(s['drawdown']):4.0f}"
def blend(curve_a, wa, curve_b, wb):   # iki ayrı kasa, yeniden dengeleme yok
    b = dict(curve_b); out = []
    a0, b0 = curve_a[0][1], curve_b[0][1]
    for d, v in curve_a:
        if d in b: out.append((d, wa * v / a0 + wb * b[d] / b0))
    return out
P1 = {s: pb.default_weight(s) for s in syms}
for start in ("2018-11-01", "2021-05-01", "2022-07-01", "2024-08-01"):
    base = port({**P1, "XAU": Decimal(20)}, decs, start)
    print(f"\n{start}  al-tut (P1+altın): {stats(base.hold_equity)}")
    print(f"  A) P1 + altın C (şimdiki)          : {stats(base.equity)}")
    f = port({**P1, "XAU": Decimal(20)}, filt, start)
    print(f"  B) A + altcoinlerde BTC filtresi    : {stats(f.equity)}")
    crypto = port(P1, decs, start).equity
    gold_hold = port({"XAU": Decimal(100)}, decs, start).hold_equity
    print(f"  C) %80 P1 kripto C + %20 altın tut  : {stats(blend(crypto, Decimal('0.8'), gold_hold, Decimal('0.2')))}")
    crypto_f = port(P1, filt, start).equity
    print(f"  D) C + altcoinlerde BTC filtresi    : {stats(blend(crypto_f, Decimal('0.8'), gold_hold, Decimal('0.2')))}")
    big = port({"BTC-USD": Decimal(40), "ETH-USD": Decimal(40), "XAU": Decimal(20)}, decs, start)
    print(f"  E) yalnız BTC/ETH %40+%40 + altın C : {stats(big.equity)}   (al-tut {stats(big.hold_equity)})")
