"""
Offline per-asset-class signal weight optimizer.

Finds better DecisionEngineConfig dimension weights (trend/momentum/volume/
pattern — ml is fixed, advanced is derived; see the design doc for why) per
asset class (crypto/BIST/commodity/forex) by backtesting candidate weight
vectors with scipy.optimize.differential_evolution (searched with
include_ml=False for speed), then validates them on multiple chronological
walk-forward windows with include_ml=True -- matching what the live signal
actually scores, so a profile can't pass validation on a strategy variant
users never see. Writes results to weight_profiles.json.

Fitness/validation pools every asset's trades into one combined sample
(_pool_score) rather than scoring each asset independently and requiring
each to individually clear MIN_TRADES: on real daily data this strategy
is selective enough (confirmation bars + entry-score threshold + a
100-bar regime warmup) that a single asset's ~30% test split often has
too few trades to judge alone, even under the shipped default weights.
Pooling gives the class's combined evidence enough samples to evaluate.

Runs serially. Parallel workers (differential_evolution's workers=-1) were
tried and gave a real speedup at small scale, but hung indefinitely at the
default search size on this machine (cause not identified) -- not worth
the risk versus a slower, proven-reliable serial run. See the
differential_evolution call in optimize_class for details.

Usage:
  python optimize_weights.py                       # all 4 classes
  python optimize_weights.py --class crypto          # just one
  python optimize_weights.py --maxiter 20 --popsize 15   # more thorough (much slower)
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
WALK_FORWARD_FOLDS = 3
MIN_WALK_FORWARD_TRAIN_BARS = 120
MIN_WALK_FORWARD_TEST_BARS = 60


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
    """x: 4 free weights [trend, momentum, volume, pattern], each searched
    over [0, 1]. ML_WEIGHT is fixed (see design doc); advanced is derived.

    The 4 free weights must sum to at most (1 - ML_WEIGHT) for `advanced`
    to stay non-negative -- but that feasible region is a small corner of
    the [0,1]^4 search box (~2.7% of its volume at ML_WEIGHT=0.1), so
    treating an over-budget sample as simply infeasible (the original
    design, returning None) meant the vast majority of randomly-sampled
    candidates were instantly discarded, and a small population could
    easily contain zero feasible members at all -- observed in practice: a
    real run's entire initial population was infeasible, collapsing the
    search to its uniform-weights fallback immediately. Proportionally
    rescaling an over-budget sample down to the budget instead preserves
    the RELATIVE weighting the candidate expressed (its proportions across
    the 4 dimensions) while guaranteeing every sampled point is usable --
    no wasted evaluations, and no more empty-population collapses."""
    trend, momentum, volume, pattern = x
    ml = DecisionEngineConfig.ML_WEIGHT
    budget = 1.0 - ml
    raw_sum = trend + momentum + volume + pattern
    if raw_sum > budget:
        scale = budget / raw_sum
        trend, momentum, volume, pattern = trend * scale, momentum * scale, volume * scale, pattern * scale
    # After rescaling, advanced should land at >= 0 by construction, but
    # float rounding can leave a ~1e-16 residual on the wrong side of zero.
    advanced = max(0.0, budget - (trend + momentum + volume + pattern))
    return {
        "trend": float(trend), "momentum": float(momentum), "volume": float(volume),
        "pattern": float(pattern), "ml": float(ml), "advanced": float(advanced),
    }


def _pool_score(bt_results):
    """Aggregate score across every asset in a class, pooling all their
    trades into one combined sample.

    Earlier version scored each asset independently (min 5 trades each)
    then took the median. On real daily crypto data that never worked: this
    strategy is deliberately selective (confirmation bars + entry-score
    threshold + a 100-bar regime-MA warmup), so a ~300-bar test split
    produces 0-3 trades per asset — even under the shipped default
    weights. Requiring 5 individually was unreachable regardless of which
    weights got tried, which is why hours of search never moved the score
    off REJECT_SCORE. Pooling gives the class's combined evidence enough
    samples to actually evaluate (e.g. 7 assets x 1-3 trades each still
    clears a single MIN_TRADES floor applied to the total).

    bt_results: list of run_strategy_backtest() return values (each may be
    None, e.g. if that asset had zero trades)."""
    trades = []
    for bt in bt_results:
        if bt:
            trades.extend(bt["trades"])
    if len(trades) < MIN_TRADES:
        return REJECT_SCORE

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = 100 * len(wins) / len(trades)
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (5.0 if gross_win > 0 else 0.0)
    avg_return_pct = sum(t["pnl_pct"] for t in trades) / len(trades)

    pf_term = min(profit_factor, 5) / 5
    return avg_return_pct * (0.5 + 0.3 * pf_term + 0.2 * win_rate / 100)


def _split_train_test(df):
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    return df.iloc[:split], df.iloc[split:]


def _walk_forward_splits(df, n_splits=WALK_FORWARD_FOLDS,
                         min_train_bars=MIN_WALK_FORWARD_TRAIN_BARS,
                         min_test_bars=MIN_WALK_FORWARD_TEST_BARS):
    """Return expanding-train, strictly later validation windows."""
    if n_splits < 2 or len(df) < min_train_bars + n_splits * min_test_bars:
        return []
    test_size = (len(df) - min_train_bars) // n_splits
    if test_size < min_test_bars:
        return []

    windows = []
    train_end = min_train_bars
    for fold in range(n_splits):
        test_end = len(df) if fold == n_splits - 1 else train_end + test_size
        windows.append((df.iloc[:train_end], df.iloc[train_end:test_end]))
        train_end = test_end
    return windows


def _class_fitness(x, train_dfs):
    """differential_evolution MINIMIZES, so this returns the negative of
    the class's pooled score (maximizing score = minimizing -score)."""
    weights = _weights_from_vector(x)
    if weights is None:
        return -REJECT_SCORE
    bt_results = [run_strategy_backtest(df, initial_balance=10000, timeframe="1d", weights=weights)
                  for df in train_dfs]
    return -_pool_score(bt_results)


def optimize_class(asset_class, assets, maxiter, popsize, source_pref="Binance"):
    """assets: list of (name, symbol) tuples. Returns a profile dict, or
    None if there wasn't enough usable data to run."""
    train_dfs, walk_forward_dfs, used_assets = [], [], []
    for name, symbol in assets:
        df, _ = get_market_data(source_pref, symbol, "1d")
        if df is None or len(df) < 200:
            logger.warning(f"[{asset_class}] skipping {name} ({symbol}): insufficient data")
            continue
        windows = _walk_forward_splits(df)
        if not windows:
            logger.warning(f"[{asset_class}] skipping {name} ({symbol}): insufficient walk-forward history")
            continue
        train_dfs.append(windows[0][0])
        walk_forward_dfs.append((df, windows))
        used_assets.append(symbol)

    if not train_dfs:
        logger.warning(f"[{asset_class}] no usable assets, skipping class")
        return None

    print(f"[{asset_class}] optimizing over {len(train_dfs)} assets: {used_assets}")

    # Calibrate against real data before committing to a maxiter/popsize
    # budget -- a hardcoded time estimate would be wrong the moment the
    # asset list, history length, or machine changes.
    calib_t0 = time.time()
    run_strategy_backtest(train_dfs[0], initial_balance=10000, timeframe="1d")
    per_backtest_s = time.time() - calib_t0
    population = popsize * 4
    generations = maxiter + 1
    # Serial execution (see the differential_evolution call below for why
    # this isn't parallelized) -- no worker-count division.
    est_s = generations * population * len(train_dfs) * per_backtest_s
    print(f"[{asset_class}] calibration: ~{per_backtest_s:.1f}s/backtest (serial) -> "
          f"est. {est_s/60:.0f} min for this class (rough; actual varies with backtest cost per candidate)")

    def progress(xk, convergence):
        print(f"  ... candidate: {_weights_from_vector(xk)}, convergence={convergence:.4f}")

    # workers=-1 (all-core multiprocessing) was tried here: it worked and
    # gave a real ~8x speedup in isolated tests (single asset, all 7 crypto
    # assets, tiny maxiter/popsize) -- but the FULL default-size search
    # (population 40) hung indefinitely on this machine: one process pegged
    # at ~100% CPU, zero progress-callback output, for 10+ minutes straight,
    # with no error. Root cause not identified (suspect Windows
    # multiprocessing.Pool degrading under sustained load at this scale, not
    # reproduced at smaller scale) and not worth chasing further under time
    # pressure -- serial execution is slower but proven reliable at every
    # scale tested. Revisit parallelism later with a capped worker count if
    # the serial runtime becomes the bottleneck again.
    t0 = time.time()
    result = differential_evolution(
        _class_fitness, bounds=[(0, 1)] * 4, args=(train_dfs,),
        seed=42, maxiter=maxiter, popsize=popsize, tol=0.01,
        callback=progress, polish=False,
    )
    elapsed = time.time() - t0

    best_weights = _weights_from_vector(result.x) or {k: (1 - DecisionEngineConfig.ML_WEIGHT) / 5 if k != "ml" else DecisionEngineConfig.ML_WEIGHT for k in WEIGHT_KEYS}
    train_score = float(-result.fun)

    # Validation folds run with include_ml=True: this is what the user's
    # live signal actually scores (ML runs on the actionable bar), so a
    # profile that only clears a technical-only backtest wouldn't validate
    # what's shipped. The search above stays include_ml=False for speed --
    # the ML weight is fixed, not searched (see module docstring), so
    # scoring it during the search wouldn't change which candidate wins.
    fold_scores = []
    for fold_idx in range(WALK_FORWARD_FOLDS):
        fold_bt_results = []
        for full_df, windows in walk_forward_dfs:
            train_df, test_df = windows[fold_idx]
            end = len(train_df) + len(test_df)
            fold_bt_results.append(run_strategy_backtest(
                full_df.iloc[:end], initial_balance=10000, timeframe="1d",
                weights=best_weights, start_index=len(train_df), include_ml=True,
            ))
        fold_scores.append(float(_pool_score(fold_bt_results)))

    # Store the median so a single exceptional regime cannot dominate.
    test_score = float(np.median(fold_scores))

    print(f"[{asset_class}] done in {elapsed:.0f}s — train_score={train_score:.2f} test_score={test_score:.2f}")
    print(f"[{asset_class}] weights: {best_weights}")

    return {
        "weights": best_weights,
        "tuned_date": time.strftime("%Y-%m-%d"),
        "train_assets": used_assets,
        "train_score": round(train_score, 2),
        "test_score": round(test_score, 2),
        "walk_forward_scores": [round(score, 2) for score in fold_scores],
    }


def main():
    parser = argparse.ArgumentParser(description="Tune signal weights per asset class")
    parser.add_argument("--class", dest="asset_class", choices=ASSET_CLASSES, default=None,
                         help="Only optimize this class (default: all)")
    # A single real backtest costs ~2-4s and this runs serially (see the
    # differential_evolution call in optimize_class for why). Defaults sized
    # to roughly 30-45 min for the largest (7-asset) class -- override for a
    # more thorough (slower) search. The calibration line printed at the
    # start of each class gives an actual estimate for your machine/data.
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--popsize", type=int, default=5)
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
