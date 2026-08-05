# Per-Asset-Class Signal Weight Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune the composite signal engine's 4 backtestable dimension weights (trend, momentum, volume, pattern) separately per asset class (crypto/BIST/commodity/forex) via offline backtesting, and apply the tuned weights live for daily-timeframe signals only.

**Architecture:** A new `weight_profiles.py` module handles ticker→class classification and JSON storage/lookup. A new offline CLI script `optimize_weights.py` searches weight-vectors with `scipy.optimize.differential_evolution`, validated on a held-out test split, and writes results to `weight_profiles.json`. `signal_engine.py` and `technical_analysis.py` get an optional `weights` parameter threaded through their scoring functions (default `None` = today's hardcoded `DecisionEngineConfig` behavior, fully backward compatible). Live call sites (`app.py`, `scanner.py`, `paper_trading.py`, `portfolio.py`) look up a symbol's profile and pass it in — but only for `timeframe == "1d"`.

**Tech Stack:** Python, `scipy.optimize.differential_evolution` (already a dependency — no new packages), pytest.

## Global Constraints

- Weight dict schema: exactly the keys `trend, momentum, volume, pattern, ml, advanced`, must sum to `1.0` within `1e-6` tolerance.
- Asset classes: exactly `crypto, bist, commodity, forex` (see `docs/superpowers/specs/2026-08-05-weight-optimizer-design.md` for the classification table).
- `ML_WEIGHT` is never searched — it stays fixed at `DecisionEngineConfig.ML_WEIGHT` in every candidate the optimizer proposes, because `run_strategy_backtest` always runs with `include_ml=False`, so the ML dimension's own score never contributes during backtesting (see design doc "Search algorithm" section for the full reasoning).
- Search is over exactly 4 free weights: `trend, momentum, volume, pattern`. `advanced = 1 - ML_WEIGHT - sum(the 4 free ones)`; negative → infeasible → `REJECT_SCORE = -1e6`.
- Per-asset score: `0 → REJECT_SCORE` if `total_trades < 5`, else `total_return * (0.5 + 0.3*min(profit_factor,5)/5 + 0.2*win_rate/100)`.
- Class fitness = `median` (not mean) of per-asset scores.
- Train/test split: chronological 70/30 per asset, no shuffling.
- `differential_evolution(seed=42, maxiter=40, popsize=15, tol=0.01)` as defaults, overridable via CLI flags.
- Tuned weights apply **only** when `timeframe == "1d"`. Every other timeframe always passes `weights=None`.
- `weight_profiles.json` is committed to the repo (not gitignored) — same treatment as `paper_trading.json`.
- All file writes use the existing atomic-write pattern (temp file + `os.replace`), matching `portfolio.py`/`paper_trading.py`.

---

### Task 1: `weight_profiles.py` — classification, storage, lookup

**Files:**
- Modify: `config.py:323-327` (add `WEIGHT_PROFILES_FILE` to `FileConfig`)
- Create: `weight_profiles.py`
- Test: `tests/test_weight_profiles.py`

**Interfaces:**
- Produces: `weight_profiles.WEIGHT_KEYS` (tuple of 6 strings), `weight_profiles.ASSET_CLASSES` (tuple of 4 strings: `"crypto", "bist", "commodity", "forex"`), `weight_profiles.classify_asset(symbol: str) -> str | None`, `weight_profiles.load_profiles() -> dict`, `weight_profiles.save_profiles(data: dict) -> None`, `weight_profiles.get_weights_for_symbol(symbol: str) -> dict | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_weight_profiles.py`:

```python
import weight_profiles as wp


def test_classify_asset_crypto():
    assert wp.classify_asset("BTC-USD") == "crypto"
    assert wp.classify_asset("DOGE-USDT") == "crypto"


def test_classify_asset_bist():
    assert wp.classify_asset("THYAO.IS") == "bist"


def test_classify_asset_forex():
    assert wp.classify_asset("EURUSD=X") == "forex"


def test_classify_asset_commodity():
    assert wp.classify_asset("XAU_GOLD") == "commodity"
    assert wp.classify_asset("GRAM_TRY") == "commodity"


def test_classify_asset_unmatched_returns_none():
    assert wp.classify_asset("AAPL") is None
    assert wp.classify_asset("") is None
    assert wp.classify_asset(None) is None


def test_load_profiles_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(missing))
    assert wp.load_profiles() == {}


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    data = {"crypto": {"weights": {k: 1 / 6 for k in wp.WEIGHT_KEYS}}}
    wp.save_profiles(data)
    assert target.exists()
    assert wp.load_profiles() == data


def test_get_weights_for_symbol_returns_valid_profile(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    weights = {"trend": 0.3, "momentum": 0.2, "volume": 0.15, "pattern": 0.1, "ml": 0.1, "advanced": 0.15}
    wp.save_profiles({"crypto": {"weights": weights}})
    assert wp.get_weights_for_symbol("BTC-USD") == weights


def test_get_weights_for_symbol_unmatched_class_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    wp.save_profiles({})
    assert wp.get_weights_for_symbol("AAPL") is None


def test_get_weights_for_symbol_no_profile_for_class_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    wp.save_profiles({})
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_malformed_profile_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    bad_weights = {"trend": 0.5, "momentum": 0.5, "volume": 0.5, "pattern": 0.5, "ml": 0.5, "advanced": 0.5}
    wp.save_profiles({"crypto": {"weights": bad_weights}})  # sums to 3.0, not 1.0
    assert wp.get_weights_for_symbol("BTC-USD") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_weight_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'weight_profiles'`

- [ ] **Step 3: Add the storage file path to config.py**

In `config.py`, modify the `FileConfig` class (currently lines 323-327):

```python
#FILES
class FileConfig:
    PORTFOLIO_FILE = 'portfolio.json'
    ASSETS_FILE = 'varliklar.json'
    PAPER_FILE = 'paper_trading.json'
    WEIGHT_PROFILES_FILE = 'weight_profiles.json'
```

- [ ] **Step 4: Create `weight_profiles.py`**

```python
"""
Per-asset-class signal weight profiles: classification, storage, and
lookup. Written/updated offline by optimize_weights.py; read live by
app.py, scanner.py, paper_trading.py, and portfolio.py.
"""
import json
import os

from config import FileConfig
from logger import logger

WEIGHT_KEYS = ("trend", "momentum", "volume", "pattern", "ml", "advanced")
ASSET_CLASSES = ("crypto", "bist", "commodity", "forex")


def classify_asset(symbol):
    """Map a ticker symbol to one of ASSET_CLASSES, or None if it doesn't
    match any known pattern (caller falls back to shipped defaults)."""
    if not symbol:
        return None
    if symbol.endswith(".IS"):
        return "bist"
    if symbol.endswith("=X"):
        return "forex"
    if symbol in ("XAU_GOLD", "GRAM_TRY"):
        return "commodity"
    if "-USD" in symbol or "-USDT" in symbol:
        return "crypto"
    return None


def load_profiles():
    f = FileConfig.WEIGHT_PROFILES_FILE
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception as e:
            logger.error(f"Weight profiles load error: {e}")
    return {}


def save_profiles(data):
    target = FileConfig.WEIGHT_PROFILES_FILE
    tmp = f"{target}.tmp"
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def _valid_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_KEYS):
        return False
    try:
        total = sum(float(weights[k]) for k in WEIGHT_KEYS)
    except (TypeError, ValueError):
        return False
    return abs(total - 1.0) < 1e-6


def get_weights_for_symbol(symbol):
    """Return the tuned weight dict for symbol's asset class, or None if
    no class matches / no profile exists / the stored profile is
    malformed (caller falls back to DecisionEngineConfig defaults)."""
    asset_class = classify_asset(symbol)
    if asset_class is None:
        return None
    entry = load_profiles().get(asset_class)
    if not entry or "weights" not in entry:
        return None
    weights = entry["weights"]
    if not _valid_weights(weights):
        logger.warning(f"Malformed weight profile for class '{asset_class}', using defaults")
        return None
    return {k: float(weights[k]) for k in WEIGHT_KEYS}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_weight_profiles.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add config.py weight_profiles.py tests/test_weight_profiles.py
git commit -m "Add weight_profiles module: asset classification + JSON storage"
```

---

### Task 2: Thread `weights` through `signal_engine.py`

**Files:**
- Modify: `signal_engine.py` (`_compute_bar_score`, `generate_composite_signal`, `generate_stable_signal`, `_cached_bar_score`; add `_weights_fingerprint`)
- Test: `tests/test_signal_engine_weights.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly — accepts a plain 6-key dict, decoupled from `weight_profiles.py`.
- Produces: `_compute_bar_score(df, timeframe="1d", include_ml=True, weights=None)`, `generate_composite_signal(df, timeframe="1d", supports=None, resistances=None, include_ml=True, weights=None)`, `generate_stable_signal(df, timeframe="1d", supports=None, resistances=None, include_ml=True, weights=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_signal_engine_weights.py`:

```python
from signal_engine import _compute_bar_score, generate_stable_signal, _weights_fingerprint, _bar_score_cache

CUSTOM_WEIGHTS = {"trend": 0.05, "momentum": 0.05, "volume": 0.60, "pattern": 0.05, "ml": 0.05, "advanced": 0.20}


def test_compute_bar_score_with_custom_weights_differs_from_default(processed_df):
    default_bar = _compute_bar_score(processed_df, "1d", include_ml=False, weights=None)
    custom_bar = _compute_bar_score(processed_df, "1d", include_ml=False, weights=CUSTOM_WEIGHTS)
    assert default_bar is not None and custom_bar is not None
    assert default_bar["score"] != custom_bar["score"]


def test_weights_fingerprint_differs_for_different_weights():
    fp_default = _weights_fingerprint(None)
    fp_a = _weights_fingerprint(CUSTOM_WEIGHTS)
    fp_b = _weights_fingerprint({**CUSTOM_WEIGHTS, "trend": 0.06, "advanced": 0.19})
    assert fp_default is None
    assert fp_a != fp_b


def test_generate_stable_signal_caches_per_weights_not_shared(processed_df):
    """Two calls with different weights on the identical bar must not
    collide in the cache and return each other's signal."""
    _bar_score_cache._data.clear()
    s_default = generate_stable_signal(processed_df, "1d", include_ml=False, weights=None)
    s_custom = generate_stable_signal(processed_df, "1d", include_ml=False, weights=CUSTOM_WEIGHTS)
    assert s_default.final_score != s_custom.final_score
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_signal_engine_weights.py -v`
Expected: FAIL — `TypeError: _compute_bar_score() got an unexpected keyword argument 'weights'` (and `ImportError` for `_weights_fingerprint`, which doesn't exist yet)

- [ ] **Step 3: Add a `_weights_fingerprint` helper**

In `signal_engine.py`, add this function right after the `_BoundedCache` class definition (before the `CompositeSignal` dataclass):

```python
def _weights_fingerprint(weights):
    """Turn a weights dict into a hashable tuple for cache keys, or None
    when weights is None (the default-config case) — so a cache entry
    computed under one weight vector is never returned for another."""
    if weights is None:
        return None
    return tuple(round(float(weights[k]), 6) for k in
                 ("trend", "momentum", "volume", "pattern", "ml", "advanced"))
```

- [ ] **Step 4: Add `weights=None` to `_compute_bar_score`**

In `signal_engine.py`, modify `_compute_bar_score` (currently starting at line 344). Change the signature and the `dim_weights` construction:

```python
def _compute_bar_score(df, timeframe="1d", include_ml=True, weights=None):
    """
    Score the LAST bar of `df` across all 6 dimensions and combine them into
    a weighted composite score + confidence. Pure computation — no verdict
    mapping, no state. Returns None when there is not enough data.

    include_ml=False skips the RandomForest dimension (its weight is
    redistributed to the other dimensions) — required for bar-by-bar
    backtesting where training a model per bar would be far too slow.

    weights: optional dict with keys trend/momentum/volume/pattern/ml/advanced
    overriding DecisionEngineConfig's fixed weights (e.g. a per-asset-class
    profile from weight_profiles.py). None (default) uses the config values,
    unchanged from today's behavior.
    """
    if df is None or len(df) < 50:
        return None

    last = df.iloc[-1]
    price = last['Close']
    atr = last.get('ATR', price * 0.02)
    if pd.isna(atr) or atr == 0:
        atr = price * 0.02

    # === SCORE ALL 6 DIMENSIONS ===
    trend_score, trend_reasons = _score_trend(df)
    momentum_score, momentum_reasons = _score_momentum(df)
    volume_score, volume_reasons = _score_volume(df)
    pattern_score, pattern_reasons = _score_patterns(df)
    if include_ml:
        ml_score, ml_reasons = _score_ml(df)
    else:
        ml_score, ml_reasons = 0, ["ML: Devre dışı"]
    advanced_score, advanced_reasons = _score_advanced(df, timeframe)

    dimension_scores = {
        "trend": trend_score,
        "momentum": momentum_score,
        "volume": volume_score,
        "pattern": pattern_score,
        "ml": ml_score,
        "advanced": advanced_score
    }

    # === ADAPTIVE WEIGHTED COMPOSITE ===
    cfg = DecisionEngineConfig
    w = weights or {
        "trend": cfg.TREND_WEIGHT, "momentum": cfg.MOMENTUM_WEIGHT,
        "volume": cfg.VOLUME_WEIGHT, "pattern": cfg.PATTERN_WEIGHT,
        "ml": cfg.ML_WEIGHT, "advanced": cfg.ADVANCED_WEIGHT,
    }

    dim_weights = {
        "trend": (trend_score, w["trend"]),
        "momentum": (momentum_score, w["momentum"]),
        "volume": (volume_score, w["volume"]),
        "pattern": (pattern_score, w["pattern"]),
        "ml": (ml_score, w["ml"]),
        "advanced": (advanced_score, w["advanced"]),
    }

    # Separate active (has meaningful data) vs inactive dimensions.
    # A dimension is "inactive" only if it returns exactly 0 AND its reasons
    # indicate data issues (not a genuine neutral score).
    active_dims = {}
    inactive_weight = 0.0

    for key, (score, weight) in dim_weights.items():
        if key == "ml" and score == 0 and any("Yetersiz" in r or "Hesaplama" in r or "hesaplanamadı" in r or "Devre dışı" in r for r in ml_reasons):
            inactive_weight += weight
        elif key == "volume" and score == 0 and any("Hacim verisi yok" in r for r in volume_reasons):
            inactive_weight += weight
        else:
            active_dims[key] = (score, weight)

    # Redistribute inactive weight proportionally
    total_active_weight = sum(w for _, w in active_dims.values())
    if total_active_weight > 0:
        redistribution_factor = (total_active_weight + inactive_weight) / total_active_weight
    else:
        redistribution_factor = 1.0

    final_score = sum(score * weight * redistribution_factor for score, weight in active_dims.values())

    # === CONFIDENCE ===
    # Only count dimensions that have data (active dimensions)
    active_scores = [score for score, _ in active_dims.values()]
    non_zero = [s for s in active_scores if abs(s) > 5]

    if len(non_zero) > 0:
        same_direction = sum(1 for s in non_zero if (s > 0) == (final_score > 0))
        agreement_ratio = same_direction / len(non_zero)
    else:
        agreement_ratio = 0.5  # No strong signals = moderate confidence

    # Average magnitude of active dimensions only
    avg_magnitude = np.mean([abs(s) for s in active_scores]) if active_scores else 0

    confidence = (agreement_ratio * 60) + (min(avg_magnitude, 60) / 60 * 40)
    confidence = max(0, min(100, confidence))

    all_reasons = trend_reasons + momentum_reasons + volume_reasons + pattern_reasons + ml_reasons + advanced_reasons

    rsi_val = last.get('RSI', 50)
    if pd.isna(rsi_val): rsi_val = 50
    adx_val = last.get('ADX', 25)
    if pd.isna(adx_val): adx_val = 25

    # Regime: +1 above the long-term SMA, -1 below, 0 unknown/disabled.
    # The state machine only allows long entries at +1 and shorts at -1.
    regime = 0
    ma_period = getattr(cfg, "REGIME_MA_PERIOD", 0)
    if ma_period and len(df) >= ma_period:
        regime_ma = df['Close'].iloc[-ma_period:].mean()
        if not pd.isna(regime_ma):
            regime = 1 if price > regime_ma else -1

    return {
        "score": final_score,
        "confidence": confidence,
        "rsi": rsi_val,
        "adx": adx_val,
        "price": price,
        "atr": atr,
        "regime": regime,
        "dimension_scores": dimension_scores,
        "reasons": all_reasons,
    }
```

Note: `w = weights or {...}` relies on a real weights dict always being truthy (non-empty dict) — fine since `_valid_weights` in `weight_profiles.py` guarantees all 6 keys are present whenever a dict is returned.

- [ ] **Step 5: Add `weights=None` to `generate_composite_signal`**

In `signal_engine.py`, modify the signature and the `_compute_bar_score` call (currently around line 586):

```python
def generate_composite_signal(df, timeframe="1d", supports=None, resistances=None, include_ml=True, weights=None):
    """
    Raw single-bar verdict — scores the latest bar with no persistence.
    Kept for compatibility/diagnostics; the UI should prefer
    generate_stable_signal(), which adds the anti-whipsaw layer.
    (supports/resistances are accepted for backward compatibility but the
    dimension scores never used them.)
    """
    signal = CompositeSignal(timeframe=timeframe)
    cfg = DecisionEngineConfig

    bar = _compute_bar_score(df, timeframe, include_ml, weights=weights)
```

(Everything else in this function is unchanged.)

- [ ] **Step 6: Add `weights=None` to `_cached_bar_score`, fold it into the cache key**

In `signal_engine.py`, replace the current `_cached_bar_score` (around line 692):

```python
def _cached_bar_score(df_slice, timeframe, include_ml, weights=None):
    try:
        key = (timeframe, str(df_slice.index[-1]),
               round(float(df_slice['Close'].iloc[-1]), 8), len(df_slice), include_ml,
               _weights_fingerprint(weights))
    except Exception:
        key = None

    if key is not None:
        cached = _bar_score_cache.get(key)
        if cached is not None:
            return cached

    result = _compute_bar_score(df_slice, timeframe, include_ml=include_ml, weights=weights)
    if key is not None and result is not None:
        _bar_score_cache.set(key, result)
    return result
```

- [ ] **Step 7: Add `weights=None` to `generate_stable_signal`, thread it through**

In `signal_engine.py`, modify `generate_stable_signal` (currently around line 710). Change the signature, the `cache_key` tuple, and the replay-loop call:

```python
def generate_stable_signal(df, timeframe="1d", supports=None, resistances=None, include_ml=True, weights=None):
    """
    Whipsaw-resistant signal — this is what the UI should display and what
    run_strategy_backtest() trades, so backtest numbers match live behavior.

      1. Scores CLOSED candles only (drops the still-forming last bar), so
         the verdict cannot flip intraday from a half-formed candle.
      2. Replays the last STABILITY_LOOKBACK closed bars through
         SignalStateMachine: a direction change needs CONFIRMATION_BARS of
         agreement, and exits have score hysteresis.

    Deterministic between candle closes — Streamlit reruns always reproduce
    the same verdict until a new candle closes.

    ML runs only on the final bar (replay bars skip it for speed; its 10%
    weight is redistributed there).

    weights: optional per-asset-class dimension weights (see
    _compute_bar_score). None uses DecisionEngineConfig defaults.
    """
    cfg = DecisionEngineConfig
    signal = CompositeSignal(timeframe=timeframe)

    work = df
    if cfg.DROP_UNCLOSED_CANDLE and df is not None and len(df) > 1:
        work = df.iloc[:-1]

    if work is None or len(work) < 55:
        signal.verdict = "BEKLE"
        signal.reasons = ["Yetersiz veri"]
        return signal

    # Output only changes when a new candle closes — cache on the last bar.
    try:
        cache_key = (timeframe, str(work.index[-1]),
                     round(float(work['Close'].iloc[-1]), 8), len(work), include_ml,
                     _weights_fingerprint(weights))
    except Exception:
        cache_key = None
    if cache_key is not None:
        cached_signal = _stable_cache.get(cache_key)
        if cached_signal is not None:
            return cached_signal

    machine = SignalStateMachine(cfg)
    lookback = min(cfg.STABILITY_LOOKBACK, len(work) - 50)
    bar = None
    raw_dir = 0

    for k in range(lookback - 1, -1, -1):
        s = work.iloc[:len(work) - k]
        is_final = (k == 0)
        b = _cached_bar_score(s, timeframe, include_ml=(include_ml and is_final), weights=weights)
        if b is None:
            continue
        _, raw_dir = machine.update(b["score"], b["confidence"], b["rsi"], b["adx"], b.get("regime", 0))
        if is_final:
            bar = b

    if bar is None:
        signal.verdict = "BEKLE"
        signal.reasons = ["Yetersiz veri"]
        return signal
```

(Everything from `signal.final_score = bar["score"]` to the end of the function is unchanged — leave it exactly as-is.)

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest tests/test_signal_engine_weights.py -v`
Expected: 3 passed

- [ ] **Step 9: Run the full existing test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: all previously-passing tests still pass (40 from before this task, +9 from Task 1, +3 from this task)

- [ ] **Step 10: Run the smoke test**

Run: `python test_signal_stability.py`
Expected: `ALL SMOKE TESTS PASSED` with identical output to before this task (it never passes `weights`, so it exercises the `weights=None` default path — must be byte-identical to pre-change behavior)

- [ ] **Step 11: Commit**

```bash
git add signal_engine.py tests/test_signal_engine_weights.py
git commit -m "Thread optional per-asset weights through signal_engine scoring"
```

---

### Task 3: Thread `weights` through `technical_analysis.run_strategy_backtest`

**Files:**
- Modify: `technical_analysis.py` (`run_strategy_backtest` signature + its two `_compute_bar_score` calls)
- Test: `tests/test_backtest_weights.py`

**Interfaces:**
- Consumes: `signal_engine._compute_bar_score(..., weights=...)` from Task 2.
- Produces: `run_strategy_backtest(df, initial_balance=10000, timeframe="1d", progress_callback=None, sl_mult=None, tp_mult=None, entry_score=None, regime_ma_period=None, fee_rate=None, entry_mode=None, pullback_tol=0.005, pullback_min_score=10, rsi_cap=None, direction="long", weights=None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backtest_weights.py`:

```python
from technical_analysis import run_strategy_backtest

CUSTOM_WEIGHTS = {"trend": 0.05, "momentum": 0.05, "volume": 0.60, "pattern": 0.05, "ml": 0.05, "advanced": 0.20}


def test_backtest_with_custom_weights_runs_and_can_differ_from_default(processed_df):
    default_bt = run_strategy_backtest(processed_df, initial_balance=10000, timeframe="1d", weights=None)
    custom_bt = run_strategy_backtest(processed_df, initial_balance=10000, timeframe="1d", weights=CUSTOM_WEIGHTS)
    # Both must at least run cleanly; a heavily volume-weighted strategy
    # trading the same data as the balanced default is expected (not
    # guaranteed on every possible dataset, but true for this fixture) to
    # produce a different trade count.
    assert default_bt is not None
    assert custom_bt is None or custom_bt["total_trades"] != default_bt["total_trades"] or custom_bt["total_return"] != default_bt["total_return"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_backtest_weights.py -v`
Expected: FAIL — `TypeError: run_strategy_backtest() got an unexpected keyword argument 'weights'`

- [ ] **Step 3: Modify `run_strategy_backtest`'s signature and its two `_compute_bar_score` calls**

In `technical_analysis.py`, modify the function signature (currently starting at line 580):

```python
def run_strategy_backtest(df, initial_balance=10000, timeframe="1d", progress_callback=None,
                          sl_mult=None, tp_mult=None, entry_score=None, regime_ma_period=None,
                          fee_rate=None, entry_mode=None, pullback_tol=0.005,
                          pullback_min_score=10, rsi_cap=None, direction="long", weights=None):
```

Then find this line inside the function's main loop (currently around line 664):

```python
        bar = _compute_bar_score(current_slice, timeframe, include_ml=False)
```

Replace it with:

```python
        bar = _compute_bar_score(current_slice, timeframe, include_ml=False, weights=weights)
```

That is the only call to `_compute_bar_score` inside `run_strategy_backtest` — confirm with:

Run: `grep -n "_compute_bar_score" technical_analysis.py`
Expected: exactly one match, on the line just modified.

Also add one line to the function's docstring (right after the `direction="short"` paragraph, before the closing `"""`) documenting the new parameter:

```
    weights: optional dict overriding DecisionEngineConfig's dimension
    weights (trend/momentum/volume/pattern/ml/advanced), e.g. a tuned
    per-asset-class profile from weight_profiles.py. None (default)
    reproduces today's shipped behavior exactly.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_backtest_weights.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the full test suite and smoke test to check for regressions**

Run: `pytest tests/ -v`
Expected: all tests pass (53 total: 40 + 9 + 3 + 1)

Run: `python test_signal_stability.py`
Expected: `ALL SMOKE TESTS PASSED`, identical numbers to before (`weights` defaults to `None` everywhere it's not explicitly passed)

- [ ] **Step 6: Commit**

```bash
git add technical_analysis.py tests/test_backtest_weights.py
git commit -m "Thread optional weights parameter through run_strategy_backtest"
```

---

### Task 4: Wire live call sites to use tuned weights (1d only)

**Files:**
- Modify: `app.py:6, 17, 107, 172` (import + two call sites)
- Modify: `scanner.py:5, 78` (import + call site)
- Modify: `paper_trading.py:119, 149` (import + call site)
- Modify: `portfolio.py:110, 116` (import + call site)
- Test: `tests/test_weight_profiles.py` already covers `get_weights_for_symbol`; this task is wiring, verified by the existing suite + a manual check (Step 6).

**Interfaces:**
- Consumes: `weight_profiles.get_weights_for_symbol(symbol: str) -> dict | None` from Task 1; `generate_stable_signal(..., weights=...)` from Task 2.

- [ ] **Step 1: `app.py` — import and wire the per-timeframe loop**

In `app.py`, add the import next to the existing `signal_engine` import (currently line 17):

```python
from signal_engine import generate_stable_signal, CompositeSignal
from weight_profiles import get_weights_for_symbol
```

Then modify the main data loop (currently lines 100-107):

```python
for tf, label in intervals.items():
    df, src = get_market_data(src_pref, symbol, tf)
    if tf == "1d": active_src = src
    results[tf] = df

    if df is not None:
        # Tuned weights only apply to the daily signal — the optimizer
        # only ever validates against daily data (see weight_profiles.py).
        sig_weights = get_weights_for_symbol(symbol) if tf == "1d" else None
        # Stable (whipsaw-filtered) composite signal — closed candles only
        comp_sig = generate_stable_signal(df, tf, weights=sig_weights)
        composite_signals[tf] = comp_sig
```

- [ ] **Step 2: `app.py` — wire the `active_signal` fallback**

Modify the fallback call (currently lines 170-172):

```python
    # Get the composite signal for the current view timeframe
    active_signal = composite_signals.get(view_tf)
    if active_signal is None:
        fallback_weights = get_weights_for_symbol(symbol) if view_tf == "1d" else None
        active_signal = generate_stable_signal(df_view, view_tf, weights=fallback_weights)
```

- [ ] **Step 3: `scanner.py` — import and wire the scan loop**

In `scanner.py`, add the import next to the existing `signal_engine` import (currently line 5):

```python
from signal_engine import generate_stable_signal
from weight_profiles import get_weights_for_symbol
```

Then modify the scan loop's signal call (currently line 78, inside `for tf, signal_col in [("1d", "Günlük Sinyal"), ("1wk", "Haftalık Sinyal")]:`):

```python
                    if isinstance(d_scan, pd.DataFrame) and not getattr(d_scan, 'empty', True) and len(d_scan) > 20:
                        # Stable (whipsaw-filtered) composite signal — same as dashboard
                        scan_weights = get_weights_for_symbol(sym) if tf == "1d" else None
                        comp_signal = generate_stable_signal(d_scan, tf, weights=scan_weights)
```

- [ ] **Step 4: `paper_trading.py` — import and wire the daily journal**

In `paper_trading.py`, `run_paper_update` currently does a local import (line 119):

```python
    from data_fetchers import get_market_data
    from signal_engine import generate_stable_signal
```

Change to:

```python
    from data_fetchers import get_market_data
    from signal_engine import generate_stable_signal
    from weight_profiles import get_weights_for_symbol
```

Then modify the signal call (currently line 149, inside `for k in todo:`, where `sym` is in scope from the outer `for idx, (name, sym) in enumerate(items):` loop):

```python
            for k in todo:
                # Live-signal input as of bar k's close: bars 0..k+1, where
                # k+1 was the then-forming candle the engine drops itself.
                slice_df = df.iloc[:k + 2]
                sig = generate_stable_signal(slice_df, "1d", weights=get_weights_for_symbol(sym))
```

- [ ] **Step 5: `portfolio.py` — import and wire multi-timeframe confirmation**

In `portfolio.py`, `multi_timeframe_confirmation` currently does a local import (line 110):

```python
    from signal_engine import generate_stable_signal  # lazy: avoids import cycle via technical_analysis
```

Change to:

```python
    from signal_engine import generate_stable_signal  # lazy: avoids import cycle via technical_analysis
    from weight_profiles import get_weights_for_symbol
```

Then modify the signal call (currently line 116, inside `for tf in ["4h", "1d", "1wk"]:`):

```python
            if isinstance(df, pd.DataFrame) and not getattr(df, 'empty', True) and len(df) > 50:  # type: ignore
                tf_weights = get_weights_for_symbol(symbol) if tf == "1d" else None
                status = generate_stable_signal(df, tf, weights=tf_weights).verdict
```

- [ ] **Step 6: Manually verify the wiring with no profile present (safe no-op check)**

Since no `weight_profiles.json` exists yet at this point in the plan, every `get_weights_for_symbol(...)` call returns `None`, so this step must produce **zero behavior change** — that's the regression check.

Run: `pytest tests/ -v`
Expected: all tests pass, same count as after Task 3

Run: `python test_signal_stability.py`
Expected: `ALL SMOKE TESTS PASSED`, numbers identical to Task 3's run

- [ ] **Step 7: Commit**

```bash
git add app.py scanner.py paper_trading.py portfolio.py
git commit -m "Wire live call sites to use tuned weight profiles for daily signals"
```

---

### Task 5: Backtest comparison UI (tuned vs default)

**Files:**
- Modify: `app.py:269-303` (the "Backtest" expander)

**Interfaces:**
- Consumes: `get_weights_for_symbol` (Task 4 import), `run_strategy_backtest(..., weights=...)` (Task 3).

- [ ] **Step 1: Modify the backtest button handler to compare tuned vs default when a profile exists**

In `app.py`, replace the backtest expander body (currently lines 272-303, `with st.expander("📊 Backtest: Strateji Performansı", expanded=False):` through the end of that block's `st.plotly_chart(fig_eq, ...)` call and the trades-table expander):

```python
        with st.expander("📊 Backtest: Strateji Performansı", expanded=False):
            st.info("Ekranda gördüğünüz Karar Motoru sinyalini (onay + histerezis filtreli) geçmiş veride test eder. Not: ML boyutu hız nedeniyle backtestte devre dışıdır, ağırlığı diğer boyutlara dağıtılır.")
            if st.button("🚀 Backtest Başlat"):
                with st.spinner("Backtest çalışıyor..."):
                    import plotly.graph_objects as go
                    tuned_weights = get_weights_for_symbol(symbol) if view_tf == "1d" else None

                    bt_progress = st.progress(0.0)
                    bt_results = run_strategy_backtest(df_view, initial_balance=10000, timeframe=view_tf,
                                                       weights=tuned_weights,
                                                       progress_callback=lambda p: bt_progress.progress(min(1.0, p)))
                    bt_progress.empty()

                    if bt_results is None:
                        st.warning("Yeterli işlem oluşmadı. Daha uzun veri gerekebilir.")
                    else:
                        if tuned_weights:
                            st.caption("🎯 Bu varlık sınıfı için ayarlanmış ağırlıklar kullanılıyor.")
                            bt_default = run_strategy_backtest(df_view, initial_balance=10000, timeframe=view_tf, weights=None)
                            if bt_default is not None:
                                comp_col1, comp_col2 = st.columns(2)
                                with comp_col1:
                                    st.markdown("**Varsayılan Ağırlıklar**")
                                    st.metric("Toplam Getiri", f"%{bt_default['total_return']:.2f}")
                                    st.metric("Kazanma Oranı", f"%{bt_default['win_rate']:.1f}")
                                with comp_col2:
                                    st.markdown("**Ayarlanmış Ağırlıklar**")
                                    st.metric("Toplam Getiri", f"%{bt_results['total_return']:.2f}",
                                               delta=f"{bt_results['total_return'] - bt_default['total_return']:+.2f}")
                                    st.metric("Kazanma Oranı", f"%{bt_results['win_rate']:.1f}",
                                               delta=f"{bt_results['win_rate'] - bt_default['win_rate']:+.1f}")
                                st.divider()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Toplam Getiri", f"%{bt_results['total_return']:.2f}")
                        col2.metric("Kazanma Oranı", f"%{bt_results['win_rate']:.1f}")
                        col3.metric("Toplam İşlem", bt_results['total_trades'])
                        col4.metric("Profit Factor", f"{bt_results['profit_factor']:.2f}")
                        st.divider()
                        col_det1, col_det2 = st.columns(2)
                        col_det1.write(f"✅ Kazanan İşlem: {bt_results['winning_trades']}")
                        col_det1.write(f"💰 Ort. Kazanç: ${bt_results['avg_win']:.2f}")
                        col_det2.write(f"❌ Kaybeden İşlem: {bt_results['losing_trades']}")
                        col_det2.write(f"💸 Ort. Kayıp: ${bt_results['avg_loss']:.2f}")
                        st.divider()
                        st.subheader("📈 Sermaye Eğrisi")
                        eq_df = pd.DataFrame(bt_results['equity_curve'])
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['equity'], mode='lines', name='Sermaye', line=dict(color='cyan', width=2)))
                        fig_eq.add_hline(y=10000, line_dash="dot", line_color="gray", annotation_text="Başlangıç")
                        fig_eq.update_layout(height=400, template="plotly_dark", hovermode='x unified', yaxis_title="Bakiye ($)", xaxis_title="Tarih")
                        st.plotly_chart(fig_eq, use_container_width=True)
                        with st.expander("📋 Tüm İşlemler"):
                            trades_df = pd.DataFrame(bt_results['trades'])
                            st.dataframe(trades_df, width="stretch")
```

Note: the trailing `with st.expander("📋 Tüm İşlemler"):` block's body (`trades_df = ...`, `st.dataframe(...)`) is unchanged from the original — only re-indented consistently since it's still inside the same `else:` branch. Verify against the original file that no lines were dropped: the original block ends with those exact two lines.

- [ ] **Step 2: Syntax-check the file**

Run: `python -m py_compile app.py`
Expected: no output (success)

- [ ] **Step 3: Run the full test suite (no test directly covers Streamlit UI code, this confirms nothing else broke)**

Run: `pytest tests/ -v`
Expected: all tests still pass, same count as after Task 4

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add tuned-vs-default backtest comparison to the Backtest panel"
```

---

### Task 6: `optimize_weights.py` — the offline search script

**Files:**
- Create: `optimize_weights.py`
- Test: `tests/test_optimize_weights.py`

**Interfaces:**
- Consumes: `weight_profiles.{WEIGHT_KEYS, ASSET_CLASSES, classify_asset, load_profiles, save_profiles}` (Task 1), `technical_analysis.run_strategy_backtest(..., weights=...)` (Task 3), `data_fetchers.get_market_data`, `config.{DEFAULT_COIN_MAP, FileConfig, DecisionEngineConfig}`.
- Produces: `optimize_weights.py`'s `_weights_from_vector(x)`, `_asset_score(bt_result)`, `_class_fitness(x, train_dfs)`, `_split_train_test(df)`, `optimize_class(asset_class, assets, maxiter, popsize, source_pref="Binance")`, `main()` — these are unit-testable pure functions plus one CLI entry point.

This task only unit-tests the pure scoring/splitting functions, **not** the full `differential_evolution` search (too slow for a test suite — that's what the manual end-to-end run in Task 7 is for).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_optimize_weights.py`:

```python
import pandas as pd

from config import DecisionEngineConfig
from optimize_weights import _weights_from_vector, _asset_score, _split_train_test, MIN_TRADES, REJECT_SCORE


def test_weights_from_vector_fills_ml_and_derives_advanced():
    weights = _weights_from_vector([0.30, 0.15, 0.25, 0.05])
    assert weights is not None
    assert weights["trend"] == 0.30
    assert weights["momentum"] == 0.15
    assert weights["volume"] == 0.25
    assert weights["pattern"] == 0.05
    assert weights["ml"] == DecisionEngineConfig.ML_WEIGHT
    expected_advanced = 1.0 - DecisionEngineConfig.ML_WEIGHT - (0.30 + 0.15 + 0.25 + 0.05)
    assert abs(weights["advanced"] - expected_advanced) < 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_from_vector_rejects_infeasible_combination():
    # trend+momentum+volume+pattern alone already exceed what's left after
    # ML_WEIGHT is reserved -> advanced would be negative -> infeasible.
    assert _weights_from_vector([0.9, 0.9, 0.9, 0.9]) is None


def test_asset_score_rejects_thin_trade_count():
    bt = {"total_trades": MIN_TRADES - 1, "total_return": 50.0, "profit_factor": 3.0, "win_rate": 90.0}
    assert _asset_score(bt) == REJECT_SCORE


def test_asset_score_rejects_none_result():
    assert _asset_score(None) == REJECT_SCORE


def test_asset_score_positive_for_good_backtest():
    bt = {"total_trades": MIN_TRADES, "total_return": 20.0, "profit_factor": 2.0, "win_rate": 60.0}
    score = _asset_score(bt)
    assert score > 0
    # formula: 20.0 * (0.5 + 0.3*min(2,5)/5 + 0.2*60/100) = 20.0 * (0.5+0.12+0.12) = 20.0*0.74
    assert abs(score - 14.8) < 1e-9


def test_split_train_test_is_chronological_and_70_30():
    df = pd.DataFrame({"Close": range(100)})
    train, test = _split_train_test(df)
    assert len(train) == 70
    assert len(test) == 30
    assert train.iloc[-1]["Close"] < test.iloc[0]["Close"]  # train is strictly earlier
    assert list(train["Close"]) + list(test["Close"]) == list(df["Close"])  # no gaps/overlap
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_optimize_weights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'optimize_weights'`

- [ ] **Step 3: Create `optimize_weights.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_optimize_weights.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass (59 total: 53 from before + 6 new)

- [ ] **Step 6: Commit**

```bash
git add optimize_weights.py tests/test_optimize_weights.py
git commit -m "Add optimize_weights.py: offline per-asset-class weight search"
```

---

### Task 7: End-to-end manual verification

**Files:** none modified — this task runs the real script against real (network-fetched) data to prove the whole pipeline works, and confirms the live app picks up the result. No new automated tests (by design — a full `differential_evolution` run is too slow for the test suite; that's exactly what Task 6 deliberately left uncovered).

- [ ] **Step 1: Run a fast, reduced-effort search for one class**

Run: `python optimize_weights.py --class crypto --maxiter 5 --popsize 6`
Expected: prints progress lines (`[crypto] optimizing over N assets: [...]`, several `... candidate: {...}` lines, then `[crypto] done in Xs — train_score=... test_score=...`), and creates/updates `weight_profiles.json` in the project root.

- [ ] **Step 2: Inspect the written profile**

Run: `python -c "import json; print(json.dumps(json.load(open('weight_profiles.json')), indent=2))"`
Expected: valid JSON with a `"crypto"` key containing `weights` (6 keys summing to ~1.0, `ml` equal to `DecisionEngineConfig.ML_WEIGHT`), `tuned_date`, `train_assets`, `train_score`, `test_score`.

- [ ] **Step 3: Confirm the live app picks it up**

Run: `python -c "from weight_profiles import get_weights_for_symbol; print(get_weights_for_symbol('BTC-USD'))"`
Expected: prints the same weights dict just written for the crypto class (not `None`).

Run: `python -c "from weight_profiles import get_weights_for_symbol; print(get_weights_for_symbol('THYAO.IS'))"`
Expected: prints `None` (no BIST profile was generated in Step 1, since only `--class crypto` was run).

- [ ] **Step 4: Confirm generate_stable_signal actually uses it end-to-end**

Run:
```bash
python -c "
from data_fetchers import get_market_data, process_data
from signal_engine import generate_stable_signal
from weight_profiles import get_weights_for_symbol

df, _ = get_market_data('Binance', 'BTC-USD', '1d')
w = get_weights_for_symbol('BTC-USD')
s_default = generate_stable_signal(df, '1d', weights=None)
s_tuned = generate_stable_signal(df, '1d', weights=w)
print('default score:', s_default.final_score)
print('tuned score:  ', s_tuned.final_score)
"
```
Expected: two different score values printed (proves the tuned weights actually change the live signal for a real, currently-classified asset).

- [ ] **Step 5: Run the full test suite one final time**

Run: `pytest tests/ -v`
Expected: all 59 tests still pass

Run: `python test_signal_stability.py`
Expected: `ALL SMOKE TESTS PASSED` (this script never passes `weights` or looks up a profile, so it's unaffected by `weight_profiles.json` existing)

- [ ] **Step 6: Commit the generated profile**

```bash
git add weight_profiles.json
git commit -m "Add initial crypto weight profile from optimize_weights.py"
```

(Note: this commits whatever the `--maxiter 5 --popsize 6` quick run produced — a real tuning pass with the default `--maxiter 40 --popsize 15` for all 4 classes should be run afterward, whenever there's time for it to run in the background; that's a follow-up, not part of this plan.)
