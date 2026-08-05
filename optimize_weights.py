"""
Offline per-asset-class signal weight optimizer.

Finds better DecisionEngineConfig dimension weights (trend/momentum/volume/
pattern — ml is fixed, advanced is derived; see the design doc for why) per
asset class (crypto/BIST/commodity/forex) by backtesting candidate weight
vectors with scipy.optimize.differential_evolution, validated on a held-out
test split. Writes results to weight_profiles.json.

Usage:
  python optimize_weights.py                       # all 4 classes
  python optimize_weights.py --class crypto          # just one
  python optimize_weights.py --maxiter 100 --popsize 25
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

from config import DEFAULT_COIN_MAP, FileConfig, DecisionEngineConfig
from data_fetchers import get_market_data
from technical_analysis import run_strategy_backtest
from weight_profiles import ASSET_CLASSES, WEIGHT_KEYS, classify_asset, load_profiles, save_profiles
from logger import logger

TRAIN_FRACTION = 0.7
MIN_TRADES = 5
REJECT_SCORE = -1e6


def _load_asset_map():
    cmap = DEFAULT_COIN_MAP.copy()
    if os.path.exists(FileConfig.ASSETS_FILE):
        try:
            with open(FileConfig.ASSETS_FILE, 'r', encoding='utf-8') as fh:
                cmap = json.load(fh)
        except Exception:
            pass
    return cmap


def _assets_by_class(coin_map):
    by_class = {c: [] for c in ASSET_CLASSES}
    for name, symbol in coin_map.items():
        cls = classify_asset(symbol)
        if cls:
            by_class[cls].append((name, symbol))
    return by_class


def _weights_from_vector(x):
    """x: 4 free weights [trend, momentum, volume, pattern]. ML_WEIGHT is
    fixed (see design doc); advanced is derived. Returns a full 6-key
    weights dict, or None if infeasible (advanced would be negative)."""
    trend, momentum, volume, pattern = x
    ml = DecisionEngineConfig.ML_WEIGHT
    advanced = 1.0 - ml - (trend + momentum + volume + pattern)
    if advanced < 0:
        return None
    return {
        "trend": float(trend), "momentum": float(momentum), "volume": float(volume),
        "pattern": float(pattern), "ml": float(ml), "advanced": float(advanced),
    }


def _asset_score(bt_result):
    if bt_result is None or bt_result["total_trades"] < MIN_TRADES:
        return REJECT_SCORE
    pf_term = min(bt_result["profit_factor"], 5) / 5
    return bt_result["total_return"] * (0.5 + 0.3 * pf_term + 0.2 * bt_result["win_rate"] / 100)


def _split_train_test(df):
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    return df.iloc[:split], df.iloc[split:]


def _class_fitness(x, train_dfs):
    """differential_evolution MINIMIZES, so this returns the negative of
    the class's median score (maximizing score = minimizing -score)."""
    weights = _weights_from_vector(x)
    if weights is None:
        return -REJECT_SCORE
    scores = [_asset_score(run_strategy_backtest(df, initial_balance=10000, timeframe="1d", weights=weights))
              for df in train_dfs]
    return -float(np.median(scores))


def optimize_class(asset_class, assets, maxiter, popsize, source_pref="Binance"):
    """assets: list of (name, symbol) tuples. Returns a profile dict, or
    None if there wasn't enough usable data to run."""
    train_dfs, test_dfs, used_assets = [], [], []
    for name, symbol in assets:
        df, _ = get_market_data(source_pref, symbol, "1d")
        if df is None or len(df) < 200:
            logger.warning(f"[{asset_class}] skipping {name} ({symbol}): insufficient data")
            continue
        train_df, test_df = _split_train_test(df)
        if len(train_df) < 100 or len(test_df) < 30:
            logger.warning(f"[{asset_class}] skipping {name} ({symbol}): split too small")
            continue
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        used_assets.append(symbol)

    if not train_dfs:
        logger.warning(f"[{asset_class}] no usable assets, skipping class")
        return None

    print(f"[{asset_class}] optimizing over {len(train_dfs)} assets: {used_assets}")

    def progress(xk, convergence):
        print(f"  ... candidate: {_weights_from_vector(xk)}, convergence={convergence:.4f}")

    t0 = time.time()
    result = differential_evolution(
        _class_fitness, bounds=[(0, 1)] * 4, args=(train_dfs,),
        seed=42, maxiter=maxiter, popsize=popsize, tol=0.01,
        callback=progress, polish=False,
    )
    elapsed = time.time() - t0

    best_weights = _weights_from_vector(result.x) or {k: (1 - DecisionEngineConfig.ML_WEIGHT) / 5 if k != "ml" else DecisionEngineConfig.ML_WEIGHT for k in WEIGHT_KEYS}
    train_score = float(-result.fun)

    test_scores = [_asset_score(run_strategy_backtest(df, initial_balance=10000, timeframe="1d", weights=best_weights))
                   for df in test_dfs]
    test_score = float(np.median(test_scores))

    print(f"[{asset_class}] done in {elapsed:.0f}s — train_score={train_score:.2f} test_score={test_score:.2f}")
    print(f"[{asset_class}] weights: {best_weights}")

    return {
        "weights": best_weights,
        "tuned_date": time.strftime("%Y-%m-%d"),
        "train_assets": used_assets,
        "train_score": round(train_score, 2),
        "test_score": round(test_score, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Tune signal weights per asset class")
    parser.add_argument("--class", dest="asset_class", choices=ASSET_CLASSES, default=None,
                         help="Only optimize this class (default: all)")
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--source", default="Binance")
    args = parser.parse_args()

    coin_map = _load_asset_map()
    by_class = _assets_by_class(coin_map)

    classes_to_run = [args.asset_class] if args.asset_class else list(ASSET_CLASSES)
    profiles = load_profiles()

    for asset_class in classes_to_run:
        assets = by_class.get(asset_class, [])
        if not assets:
            print(f"[{asset_class}] no classified assets in your asset list, skipping")
            continue
        entry = optimize_class(asset_class, assets, args.maxiter, args.popsize, args.source)
        if entry is not None:
            profiles[asset_class] = entry
            save_profiles(profiles)  # save after each class so partial runs aren't lost

    print("Done. Profiles written to", FileConfig.WEIGHT_PROFILES_FILE)


if __name__ == "__main__":
    main()
