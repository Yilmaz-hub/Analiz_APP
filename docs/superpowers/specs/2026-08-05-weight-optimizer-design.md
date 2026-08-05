# Per-Asset-Class Signal Weight Optimizer — Design

## Goal

The composite decision engine (`signal_engine._compute_bar_score`) blends 6
dimensions — Trend, Momentum, Volume, Pattern, ML, Advanced — into one score
using fixed weights from `DecisionEngineConfig` (`TREND_WEIGHT=0.25`,
`MOMENTUM_WEIGHT=0.20`, etc., summing to 1.0). These weights are hand-picked
and shared across every asset. This feature finds better weights per asset
class (crypto / BIST / commodities / forex) via backtesting, and applies
them live without touching the scoring logic itself.

Explicitly out of scope for this iteration (deferred, not forgotten):
per-formation sub-weights (e.g. weighting a triangle differently from a
flag), tuning entry/exit thresholds, and true per-asset (vs per-class)
tuning. All three were considered and deferred during design discussion —
see "Rejected/deferred alternatives" at the end.

## Asset classification

New module `weight_profiles.py`, function `classify_asset(symbol: str) ->
str | None`:

| Pattern | Class |
|---|---|
| ends with `.IS` | `"bist"` |
| ends with `=X` | `"forex"` |
| `== "XAU_GOLD"` or `== "GRAM_TRY"` | `"commodity"` |
| contains `-USD` or `-USDT` | `"crypto"` |
| anything else | `None` (no class — caller uses shipped defaults) |

This mirrors the pattern-matching already used in
`data_fetchers.get_live_price_for_portfolio`. `None` is a normal, expected
return value (e.g. a custom stock ticker you add later) — callers must
treat it as "use `DecisionEngineConfig` defaults," never as an error.

## Weight storage: `weight_profiles.json`

```json
{
  "crypto": {
    "weights": {"trend": 0.30, "momentum": 0.15, "volume": 0.25, "pattern": 0.05, "ml": 0.10, "advanced": 0.15},
    "tuned_date": "2026-08-05",
    "train_assets": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "PEPE-USD"],
    "train_score": 41.2,
    "test_score": 33.8
  },
  "bist": { ... },
  "commodity": { ... },
  "forex": { ... }
}
```

`weight_profiles.py` provides:
- `load_profiles() -> dict` (atomic-write-safe reader, matching
  `portfolio.py`'s pattern; returns `{}` if the file doesn't exist yet)
- `save_profiles(data: dict)` (atomic write: temp file + `os.replace`)
- `get_weights_for_symbol(symbol: str) -> dict | None` — classifies the
  symbol, looks up its class in the loaded profiles, validates the 6
  weights are present and sum to ~1.0 (tolerance 1e-6), and returns the
  weights dict or `None` (malformed profile, missing class, or
  unclassifiable symbol all fall back to `None` → live code uses
  `DecisionEngineConfig` defaults, exactly like today. A malformed profile
  logs a warning but never crashes the app).

## Objective / score function

Per-asset score, computed from a `run_strategy_backtest` result:

```python
def _asset_score(bt_result):
    if bt_result is None or bt_result["total_trades"] < MIN_TRADES:  # MIN_TRADES = 5
        return REJECT_SCORE  # -1e6 — thin/lucky results can't win
    pf_term = min(bt_result["profit_factor"], 5) / 5
    return bt_result["total_return"] * (0.5 + 0.3 * pf_term + 0.2 * bt_result["win_rate"] / 100)
```

Class-level fitness for a candidate weight vector = **median** of
`_asset_score` across all assets currently classified into that class
(median, not mean, so one outlier asset can't dominate the tuned result).

## Search algorithm

`scipy.optimize.differential_evolution` (already a dependency — no new
package needed). Searches 5 free weights (`trend, momentum, volume,
pattern, ml`) each bounded `[0, 1]`; the 6th (`advanced`) =
`1 - sum(others)`. If that's negative, the candidate is infeasible and
gets `REJECT_SCORE` (simple rejection, not clipping/renormalizing — avoids
misleading the optimizer with a flattened gradient at the boundary).

`differential_evolution(objective, bounds=[(0,1)]*5, seed=42, maxiter=40,
popsize=15, tol=0.01)` — seeded for reproducibility (matching the
project's existing `random_state=42` convention). `maxiter=40` /
`popsize=15` is a starting point balancing search quality against runtime;
exposed as CLI flags so a longer search can be run if the default proves
too coarse.

Timeframe: **daily (`1d`) only** for this iteration, matching the existing
validated precedent in `config.py` ("Daily: entry 25 / SL 2.5 ATR
validated on 5 assets"). 4h/1wk tuning is a natural follow-up once daily
is proven out, but weekly especially may not generate enough trades per
asset to tune meaningfully.

## Train/test split

Per asset, chronological 70/30 split (no shuffling — this is time-series
data). The search (`differential_evolution` objective calls) only ever
sees the train slice. Once the search converges, the winning weights are
backtested **once** on the test slice of every asset in the class — that
result (never seen during search) is what gets written to
`weight_profiles.json` as `test_score`, alongside `train_score` for
comparison. A large train/test gap is the visible overfitting signal.

## CLI script: `optimize_weights.py`

Root-level script, same style as `paper_trading.py`'s `__main__` block:

```
python optimize_weights.py                  # runs all 4 classes
python optimize_weights.py --class crypto    # just one class
python optimize_weights.py --maxiter 100 --popsize 25   # longer search
```

For each class: load its assets (from `DEFAULT_COIN_MAP` merged with
`varliklar.json` if present, same source `app.py` already uses), fetch
daily data via `data_fetchers.get_market_data`, run the search, print
progress (generation number + best-so-far score) to stdout, write results
into `weight_profiles.json` (merging into the existing file — a run of
one class doesn't erase the others' profiles).

Skips a class entirely (with a warning) if it has zero classified assets.

## Live integration

**`signal_engine.py`**:
- `_compute_bar_score(df, timeframe="1d", include_ml=True, weights=None)` —
  when `weights` is provided, use it instead of reading
  `DecisionEngineConfig.*_WEIGHT` directly. `None` (the default) preserves
  exactly today's behavior.
- `generate_composite_signal(..., weights=None)` and
  `generate_stable_signal(..., weights=None)` thread it through.
- Cache keys (`_stable_cache`, `_bar_score_cache`) must incorporate the
  weights, since two different weight vectors produce different scores for
  the identical bar — a `_weights_cache_key(weights)` helper turns the
  dict into a hashable tuple (or the literal `"default"` when `None`) for
  inclusion in the existing cache-key tuples.

**`technical_analysis.py`**: `run_strategy_backtest(..., weights=None)`
passes `weights` through to its internal `_compute_bar_score` calls —
this is also what the optimizer's own search loop calls, so tuning and
live signal generation always go through the identical code path.

**`app.py` / `scanner.py` / `paper_trading.py`**: each already knows the
selected symbol; call `weight_profiles.get_weights_for_symbol(symbol)` and
pass the result straight into `generate_stable_signal(...)` /
`run_strategy_backtest(...)`. `None` is a valid, expected value (falls
back to defaults) — no branching needed at call sites.

**UI**: in `app.py`'s existing "Backtest" expander, when a profile exists
for the selected asset's class, run the backtest twice (shipped defaults
vs. tuned weights) and show both side by side, so you can visually confirm
the tuned version is actually better on that specific asset before
trusting it. Nothing is auto-applied to the live verdict silently — the
live verdict panel uses the tuned weights automatically once a profile
exists (per the "class-level, applies immediately" decision), but the
backtest comparison is what lets you sanity-check that decision.

## Testing plan

- `weight_profiles.py`: classification rules (all 4 classes + unmatched →
  `None`), load/save round-trip, malformed-profile fallback to `None`.
- `signal_engine.py`: `_compute_bar_score` with a custom `weights` dict
  produces a different score than `weights=None` on the same bar (proves
  it's actually wired in); cache keys differ between two different weight
  vectors on the same bar (proves no cross-contamination between
  profiles).
- `technical_analysis.py`: `run_strategy_backtest` with an explicit
  `weights` dict runs without error and produces a well-formed result on
  synthetic data.
- `optimize_weights.py`: the objective function scoring
  (`_asset_score` reject-on-thin-trades, infeasible-vector rejection,
  median aggregation) tested directly with fabricated backtest results —
  not the full `differential_evolution` search itself (too slow for a
  unit test suite; that's what a manual CLI run + eyeballing the written
  `weight_profiles.json` is for).

## Rollout / safety

- Nothing changes for any asset until `optimize_weights.py` is actually
  run — no auto-tuning on app startup, no silent background job.
- **Tuned weights apply to the `1d` live signal only.** The search only
  ever validates against daily data, so applying those same weights to 4h
  or weekly signals would mean using weights tuned for one timeframe's
  dynamics in a context they were never tested against. `app.py`'s
  per-timeframe loop passes `weights=get_weights_for_symbol(symbol)` only
  when `tf == "1d"`; the 4h and 1wk calls always pass `weights=None`
  (shipped defaults), regardless of whether a profile exists. Per-timeframe
  tuning is a natural follow-up once daily is proven out.
- `weight_profiles.json` should be **committed** to the repo (like
  `paper_trading.json` already is), not gitignored — Streamlit Cloud
  deploys from git, so an un-committed profile file would silently vanish
  on every redeploy and every asset would quietly fall back to defaults.
- A profile can be deleted/reset by simply removing its key from
  `weight_profiles.json` — the live code falls straight back to
  `DecisionEngineConfig` defaults with no code change needed.

## Rejected/deferred alternatives (for the record)

- **Per-formation sub-weights** — deferred: rare patterns (Butterfly,
  ABCD, H&S) fire too infrequently per asset to tune sub-weights against
  without overfitting to noise. Revisit once we can see per-formation
  trade counts from real usage.
- **True per-asset tuning** — deferred: each class already has few enough
  assets that per-asset tuning would have very little trade history to
  work with per asset. Class-level tuning was chosen explicitly over this
  during design discussion.
- **Full grid search** — rejected: a 6-weight simplex grid is
  combinatorially far too large to evaluate at any reasonable resolution
  given each point costs a full multi-asset backtest;
  `differential_evolution` gets a comparable-quality answer in far fewer
  evaluations.
- **Walk-forward validation** — deferred in favor of a single train/test
  split: more robust but substantially slower (many backtests per asset
  instead of one) and more code; a reasonable v2 if the simple split
  proves the concept but the walk-forward robustness gap becomes a
  concern in practice.
