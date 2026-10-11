import sys; sys.path.insert(0, "exp")
from decimal import Decimal
import pf_variants as pv
from methods import donchian
import portfolio_backtest as pb
mixed = {s: (donchian(pv.full[s], pv.v1s[s]) if s not in ("BTC-USD", "ETH-USD", "XAU") else d) for s, d in pv.decs.items()}
alld = {s: donchian(pv.full[s], pv.v1s[s]) for s in pv.decs}
W = {**pv.P1, "XAU": Decimal(20)}
print("\n=== Donchian portföy ===")
for start in ("2018-11-01", "2021-05-01", "2022-07-01", "2024-08-01"):
    a = pv.port(W, pv.decs, start); m = pv.port(W, mixed, start); x = pv.port(W, alld, start)
    print(f"{start}  A şimdiki: {pv.stats(a.equity)} işlem {len(a.trades):3} | A' altcoin Donchian: {pv.stats(m.equity)} işlem {len(m.trades):3} | hepsi Donchian: {pv.stats(x.equity)} | al-tut {pv.stats(a.hold_equity)}")
