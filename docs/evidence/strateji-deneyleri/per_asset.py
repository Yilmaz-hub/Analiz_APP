import json, pickle, sys
from decimal import Decimal
import pandas as pd
sys.path.insert(0, "/home/user/Analiz_APP")
import portfolio_backtest as pb, strategy_candidates as sc
from technical_analysis import build_v1_decisions
from trade_execution import CostAssumptions
TIERS = {"Büyük": ["BTC", "ETH"], "Orta": ["XRP", "SOL", "BNB", "ADA", "TRX", "LINK", "AVAX", "LTC", "DOT"],
         "Küçük": ["DOGE", "HBAR", "RENDER", "ATOM", "NEAR", "FET", "INJ", "SHIB", "ALGO", "VET"], "Altın": ["XAU"]}
costs = CostAssumptions(Decimal(0), Decimal(0), Decimal("0.1"))
def load(b):
    if b == "XAU":
        df = pickle.load(open("exp/data_long/XAU_GOLD.pkl", "rb")); v1 = build_v1_decisions(df, include_ml=False); mk = "ALTIN"
    else:
        df = pickle.load(open(f"exp/data_long/{b}-USD.pkl", "rb")); raw = json.load(open(f"exp/data_long/{b}-USD.dec.json"))
        v1 = {ts: raw[str(ts.date())] for ts in df.index if str(ts.date()) in raw}; mk = "KRIPTO"
    return df, v1, sc.candidate_decisions(df, v1, sc.trend_candidate(mk), None)
def run(df, dec, start):
    r = pb.run_portfolio({"A": df}, {"A": dec}, {"A": Decimal("100")}, capital=Decimal("10000"), costs=costs,
                         quantity_steps={"A": Decimal("0.00000001")}, start=start)
    s, h = pb.summary(r.equity), pb.summary(r.hold_equity)
    days_in = 0
    for t in r.trades: days_in += (pd.Timestamp(t.exit_at) - pd.Timestamp(t.entry_at)).days
    return dict(ann=float(s["annual"]), tot=float(s["total"]), dd=float(s["drawdown"]), h_ann=float(h["annual"]),
                h_dd=float(h["drawdown"]), n=len(r.trades), win=sum(t.pnl > 0 for t in r.trades),
                years=float(s["years"]))
if __name__ == "__main__":
  out = []
  for tier, bases in TIERS.items():
      for b in bases:
          df, v1, c = load(b)
          start = pb.first_tradable_day(df, list(v1))
          for window, st in (("tam", start), ("ortak", max(start, pd.Timestamp("2022-07-01", tz="UTC")))):
              if b == "XAU" and window == "ortak":
                  st = df.index[df.index >= pd.Timestamp("2022-07-01", tz="UTC")][0]
              row = dict(tier=tier, asset=b, window=window, start=str(st.date()), C=run(df, c, st), V1=run(df, v1, st))
              out.append(row); print(json.dumps(row), flush=True)
  json.dump(out, open("exp/per_asset.json", "w"))
