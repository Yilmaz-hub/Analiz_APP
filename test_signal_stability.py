# -*- coding: utf-8 -*-
"""
Smoke test for the anti-whipsaw signal engine + composite backtest.
Uses synthetic OHLCV data (trend regimes + noise) so no network is needed.

Run from the project directory with the Python that has the deps (3.13):
  C:\\Users\\Yilmaz\\AppData\\Local\\Programs\\Python\\Python313\\python.exe test_signal_stability.py
"""
import sys, time, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# --- Build synthetic OHLCV with trend regimes ---
rng = np.random.default_rng(42)
n = 500
regime = np.concatenate([
    np.full(120, 0.003),   # uptrend
    np.full(80, -0.004),   # downtrend
    np.full(100, 0.0),     # chop
    np.full(120, 0.005),   # strong uptrend
    np.full(80, -0.002),   # mild downtrend
])
rets = regime + rng.normal(0, 0.02, n)
close = 100 * np.cumprod(1 + rets)
high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
open_ = np.roll(close, 1); open_[0] = close[0]
vol = rng.uniform(1e6, 5e6, n) * (1 + np.abs(regime) * 100)

idx = pd.date_range("2024-01-01", periods=n, freq="D")
df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)

# --- Run through the app's own indicator pipeline ---
from data_fetchers import process_data
df, src = process_data(df, "test")
assert df is not None, "process_data failed"
print(f"[1] process_data OK — {len(df)} bars, columns include RSI/ADX/ATR: "
      f"{all(c in df.columns for c in ['RSI','ADX','ATR','EMA_20','EMA_50'])}")

from signal_engine import generate_stable_signal, generate_composite_signal, _compute_bar_score, SignalStateMachine
from technical_analysis import run_strategy_backtest

# --- [2] Stable signal: basic output + determinism across reruns ---
s1 = generate_stable_signal(df, "1d", include_ml=False)
s2 = generate_stable_signal(df, "1d", include_ml=False)
print(f"[2] stable verdict={s1.verdict} raw={s1.raw_verdict} score={s1.final_score:.1f} "
      f"conf={s1.confidence:.0f} bars_held={s1.bars_held}")
assert s1.verdict == s2.verdict and s1.final_score == s2.final_score, "not deterministic"
assert s1.verdict in ("GÜÇLÜ AL", "AL", "BEKLE", "SAT", "GÜÇLÜ SAT")
print("    determinism OK, top reason:", s1.reasons[0] if s1.reasons else "-")

# --- [3] Flip-count: stable vs raw over a rolling window ---
raw_flips, stable_flips = 0, 0
prev_raw, prev_stable = None, None
machine = SignalStateMachine()
t0 = time.time()
for i in range(150, len(df)):
    sl = df.iloc[:i]
    b = _compute_bar_score(sl, "1d", include_ml=False)
    if b is None:
        continue
    state, raw = machine.update(b["score"], b["confidence"], b["rsi"], b["adx"])
    if prev_raw is not None and raw != prev_raw:
        raw_flips += 1
    if prev_stable is not None and state != prev_stable:
        stable_flips += 1
    prev_raw, prev_stable = raw, state
elapsed = time.time() - t0
per_bar = elapsed / (len(df) - 150)
print(f"[3] over {len(df)-150} bars: raw direction flips={raw_flips}, "
      f"confirmed (stable) flips={stable_flips}  ({per_bar*1000:.0f} ms/bar)")
assert stable_flips < raw_flips, "state machine did not reduce flips"

# --- [4] Backtest on composite engine ---
t0 = time.time()
bt = run_strategy_backtest(df, initial_balance=10000, timeframe="1d")
elapsed = time.time() - t0
assert bt is not None, "backtest returned None (no trades)"
print(f"[4] backtest OK in {elapsed:.1f}s — trades={bt['total_trades']} "
      f"return={bt['total_return']:.1f}% win_rate={bt['win_rate']:.0f}% pf={bt['profit_factor']:.2f}")
reasons = {}
for t in bt["trades"]:
    reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
print(f"    exit reasons: {reasons}")

# --- [5] Backtest determinism ---
bt2 = run_strategy_backtest(df, initial_balance=10000, timeframe="1d")
assert bt2["total_return"] == bt["total_return"], "backtest not deterministic"
print("[5] backtest determinism OK")

# --- [6] Weekly + 4h paths, short data guard ---
def fmt(bt_r):
    if bt_r is None:
        return "None (no trades)"
    return "{} trades, {:.1f}%".format(bt_r["total_trades"], bt_r["total_return"])

bt_w = run_strategy_backtest(df, timeframe="1wk")
bt_4h = run_strategy_backtest(df, timeframe="4h")
print("[6] 1wk: " + fmt(bt_w) + " | 4h: " + fmt(bt_4h))
short = generate_stable_signal(df.iloc[:40], "1d")
assert short.verdict == "BEKLE" and "Yetersiz" in short.reasons[0]
print("    short-data guard OK")

# --- [7] Unclosed-candle immunity: mutate the LAST bar, verdict must not change ---
df_mut = df.copy()
df_mut.iloc[-1, df_mut.columns.get_loc("Close")] *= 1.10  # +10% fake intrabar pump
s_mut = generate_stable_signal(df_mut, "1d", include_ml=False)
assert s_mut.verdict == s1.verdict, f"verdict flipped from unclosed candle! {s1.verdict} -> {s_mut.verdict}"
print(f"[7] unclosed-candle immunity OK (verdict stayed {s1.verdict} despite +10% pump on forming bar)")

print("\nALL SMOKE TESTS PASSED")
