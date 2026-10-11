import sys, pickle, json; sys.path.insert(0, "exp")
from decimal import Decimal
import pandas as pd
import pf_variants as pv
from methods import donchian
import portfolio_backtest as pb, strategy_candidates as sc
dec = {s: (donchian(pv.full[s], pv.v1s[s]) if s not in ("BTC-USD", "ETH-USD", "XAU") else d) for s, d in pv.decs.items()}
for s in ("QQQ", "XU100"):
    df = pickle.load(open(f"exp/data_stk/{s}.pkl", "rb")); pv.full[s] = df; dec[s] = {}
def port(weights, start, hold):
    fr = {s: pv.full[s][pv.full[s].index >= pd.Timestamp(start, tz="UTC")] for s in weights}
    return pb.run_portfolio(fr, {s: dec[s] for s in weights}, weights, capital=Decimal("10000"), costs=pv.costs,
                            quantity_steps={s: Decimal("0.00000001") for s in weights}, hold=hold)
W = {**pv.P1, "XAU": Decimal(20)}
print("\n=== TUT ===")
for start in ("2018-11-01", "2021-05-01", "2022-07-01", "2024-08-01"):
    a = port(W, start, ()); b = port(W, start, {"XAU"})
    c = port({**W, "QQQ": Decimal(10)}, start, {"XAU", "QQQ"})
    print(f"{start} altın C: {pv.stats(a.equity)} | altın TUT: {pv.stats(b.equity)} | altın+QQQ%10 TUT: {pv.stats(c.equity)} | al-tut: {pv.stats(a.hold_equity)}")
