# Analiz_APP — Trading Signal & Portfolio Dashboard

A Streamlit app (Turkish UI) that pulls OHLCV data for crypto/FX/gold/BIST tickers, runs technical/ML/pattern analysis, and renders a composite BUY/SELL/WAIT verdict per timeframe, plus a manual portfolio tracker and an opportunity scanner across a configurable asset list.

Run: `streamlit run app.py` (deps in `requirements.txt`: streamlit, pandas, pandas_ta, plotly, requests, yfinance, scipy, numpy, scikit-learn). No test suite exists.

**Python environment (Windows):** the deps are installed under Python 3.13 at `C:\Users\Yilmaz\AppData\Local\Programs\Python\Python313\python.exe`. The `python` on PATH is 3.14, which does NOT have pandas/streamlit — use the 3.13 path explicitly for scripts and tests.

`backup/Analiz_APP-main/` is a full snapshot of the previous version of every module — use it to diff what changed, not as a place to develop. `data_fetchers.py`, `signal_engine.py`, `technical_analysis.py`, `app.py`, `config.py`, and `scanner.py` have diverged from backup; everything else is currently identical.

## The core problem & the anti-whipsaw architecture (built July 2026)

The original complaint: the live verdict (GÜÇLÜ AL / AL / BEKLE / SAT / GÜÇLÜ SAT) flipped day-to-day or even hourly, unusable without watching the screen. Root causes were: (1) stateless verdict mapping with no persistence, (2) narrow thresholds (±15 on a ±100 scale) relative to score noise, (3) re-scoring the still-forming candle on every Streamlit rerun, and (4) the backtest testing a *different* signal (`calculate_oracle_signal_v2`) than the one displayed.

All four are now addressed in `signal_engine.py`:

- **`_compute_bar_score(df, timeframe, include_ml)`** — pure per-bar scoring (6 weighted dimensions → composite score + confidence), shared by live signal and backtest. `include_ml=False` skips the RandomForest dimension and redistributes its weight (mandatory in bar-by-bar loops — training a model per bar is infeasible).
- **`SignalStateMachine`** — the anti-whipsaw core, shared by live and backtest. State: +1/0/-1. Direction changes require `CONFIRMATION_BARS` consecutive closed bars of agreement (at the ±15 `BUY/SELL_THRESHOLD` level); exits have score hysteresis (`EXIT_SCORE_BUFFER` — once long, the signal holds while score stays above `BUY_THRESHOLD - buffer`). On top of the direction state sits an **arming latch** (`.signal` property): a confirmed direction only becomes an actionable AL/SAT once one bar clears `ENTRY_SCORE` (±25) with the regime filter agreeing (price above/below the `REGIME_MA_PERIOD`=100 SMA); once armed it stays armed until the direction dies. Entry guards (entry score, regime, chop/ADX, RSI overextension, low confidence) only block *new* signals, never force exits. **Do not "simplify" the arming latch into raising BUY_THRESHOLD** — requiring ±25 for the confirmation bars themselves was tested and destroys returns (avg −17.8% vs +22.6%), because it delays exits and misses one-bar entry triggers.
- **`generate_stable_signal()`** — the public entry point used by `app.py`, `scanner.py`, and `portfolio.multi_timeframe_confirmation`. Drops the unclosed last candle (`DROP_UNCLOSED_CANDLE`), replays the last `STABILITY_LOOKBACK` closed bars through the state machine, caches on the last closed bar so Streamlit reruns are free and the verdict is deterministic between candle closes. Exposes `raw_verdict` (unfiltered) and `bars_held` so the UI can show "onay bekliyor" / "N bardır stabil".
- **`generate_composite_signal()`** — the raw single-bar verdict, kept for diagnostics/compat; UI should not display it directly.
- **`run_strategy_backtest()`** (`technical_analysis.py`) now trades `_compute_bar_score` + `SignalStateMachine` — the same logic the dashboard shows (minus ML). Signal decided on bar close, executed next bar; long-only; ATR-based SL/TP with breakeven + profit-lock trailing; loss cooldown; timeframe-adaptive params; `sl_mult`/`tp_mult`/`entry_score` overrides for tuning. ~5ms/bar → ~5s per 1000-bar run.

All stability/quality knobs live in `DecisionEngineConfig` (`config.py`): `CONFIRMATION_BARS`, `STABILITY_LOOKBACK`, `EXIT_SCORE_BUFFER`, `DROP_UNCLOSED_CANDLE`, `CHOP_ADX_LIMIT`, `CHOP_SCORE_OVERRIDE`, `ENTRY_SCORE`, `REGIME_MA_PERIOD`.

## Backtest results with the shipped config (2026-07-15, Binance 1d, 1000 bars, ML off, 0.1%/side fees)

The backtest charges `BacktestConfig.FEE_RATE` (0.001 = 0.1% Binance spot taker) per side; `fee_rate=0` override reproduces the old no-fee numbers (which averaged +22.6%, i.e. fees cost ~3.2pp and flipped the strategy from ≈beating to trailing buy&hold).

| Asset | Strategy | Buy&Hold | Win% | Trades | PF |
|---|---|---|---|---|---|
| BTC | +37.4% | +54.3% | 50 | 18 | 1.78 |
| ETH | −23.4% | −11.8% | 29 | 17 | 0.56 |
| SOL | −47.7% | +7.1% | 22 | 18 | 0.58 |
| XRP | +158.0% | +85.2% | 50 | 10 | 3.45 |
| DOGE | −27.1% | −17.2% | 40 | 15 | 0.84 |
| **avg** | **+19.4%** | **+23.5%** | | | |

## Trend-regime scanner ranking (built 2026-07-15)

`calculate_regime_score(df)` (`technical_analysis.py`, knobs in `RegimeConfig`) scores trend quality 0–100 from four components: % of last 50 bars above the 100-SMA, 100-SMA slope, ADX (counted only above the MA), EMA20>EMA50>SMA100 alignment. The scanner shows it as "Trend Rejimi", ranks results by "Fırsat Puanı" (0.6·decision score + 0.4·regime mapped to ±100), and recommends allocation only when regime ≥ `MIN_TRADEABLE_SCORE`.

**Per-trade validation (72 trades, 5 assets):** regime<35 at entry netted ~0% combined (9 trades) → gating there is free. But the score LAGS new trends — trades entered at regime 35–55 summed +41.6%, and a gate at 55 would have dropped +42% of profit (early entries into what became the big XRP/BTC runs). So `MIN_TRADEABLE_SCORE=35`; do not raise it without re-running the per-trade analysis (`scratchpad validate_regime_trades.py` pattern: regime score at each backtest trade's entry bar, bucketed by threshold). The regime column is primarily *comparative* (which asset is healthiest now) — it is not a validated per-trade entry filter.

Tuning history that led here (don't re-derive): defaults before tuning averaged −38.1% on BTC/ETH/SOL. Entry 15→25 helped every asset; SL 1.8→2.5 ATR helped consistently; MA100 regime filter helped every loser without hurting winners; MA150/200 were no better than MA100. Raising the machine's confirmation thresholds to ±25 instead of using the arming latch was much worse (see above).

Whipsaw verification: direction-state flips over last 365 daily bars — BTC raw 28 → confirmed 21; the *displayed* verdict is further stabilized by the arming latch. Verdict immune to ±10% mutation of the forming candle; backtest deterministic. Regression test: `test_signal_stability.py` (synthetic data, no network) covers determinism, flip reduction, unclosed-candle immunity, short-data guards — run it after any change to `signal_engine.py`, `technical_analysis.py`, or `DecisionEngineConfig`.

## Where this is headed

With honest fees the strategy now *trails* buy&hold on average (+19.4% vs +23.5%): it wins big on assets that trended (BTC, XRP) and bleeds on choppy/declining ones (ETH, SOL, DOGE — profit factors 0.56–0.84). The scanner's regime ranking (above) steers capital toward trending assets but is comparative, not a validated entry filter. Known open directions, in rough order of expected value: (1) entry-quality research — current entries buy strength and get stopped in mean reversion (SL is the dominant exit); test pullback-in-uptrend entries. (2) Short side is scored but the backtest is long-only — validate before trusting SAT signals for actual shorts. (3) A less-lagging regime measure (the current one misses early trends by construction) could make per-trade gating viable — re-run the per-trade bucket analysis before believing any variant. Rules: tune only via `run_strategy_backtest` overrides across ≥5 assets, compare to buy&hold, and ship to `config.py` only what wins broadly — single-asset wins are noise.

## Module map

| File | Role |
|---|---|
| `app.py` | Streamlit entrypoint. Sidebar controls, per-timeframe signal loop, decision dashboard, backtest panel, portfolio UI, Telegram auto-alert loop (`time.sleep(14400)` when "Otomatik Bot" is on). |
| `config.py` | All tunable constants/thresholds, grouped by concern (`DataFetchConfig`, `IndicatorConfig`, `RiskConfig`, `MLConfig`, `SignalConfig` [legacy], `DecisionEngineConfig` [composite engine], `BacktestConfig` [fees], `RegimeConfig` [scanner trend ranking], `AdvancedAnalysisConfig`, `SRConfig`, `Constants`, `FileConfig`, `TelegramConfig`, `UIConfig`). Change thresholds here, not inline. |
| `data_fetchers.py` | OHLCV fetching from Binance, OKX, Yahoo Finance, plus synthetic gram-gold conversion and Fear & Greed index. `get_market_data()` is the main entry, dispatches by `source_pref`. |
| `technical_analysis.py` | Indicator/S-R/trendline/pattern calculations, legacy `calculate_oracle_signal_v2` signal (no longer used by UI), and `run_strategy_backtest` (trades the composite engine via lazy import — see above). |
| `signal_engine.py` | Composite decision engine — 6 weighted dimensions (trend, momentum, volume, pattern, ML, advanced). `generate_stable_signal()` (whipsaw-filtered, closed candles) is what the UI/scanner display; `_compute_bar_score` + `SignalStateMachine` are the shared internals. |
| `advanced_analysis.py` | Elliott Wave, Ichimoku, Wyckoff phase, market structure (BOS/CHoCH) — feeds the "advanced" dimension of the composite engine and its own expander panel in the UI. |
| `ml_models.py` | RF/LR ensemble price-direction model (`calculate_ml_direction_signal`, feeds the "ml" dimension) and a separate price-path forecaster (`calculate_smart_prediction_FIXED`, feeds the AI Tahmin chart overlay). |
| `scanner.py` | Opportunity scanner — runs the signal pipeline across the whole asset map, ranks by "Fırsat Puanı" (signal + trend regime blend), gates allocation advice on `RegimeConfig.MIN_TRADEABLE_SCORE`. |
| `portfolio.py` | JSON-backed manual portfolio (`portfolio.json`), risk validation, auto-close of active positions against live price, multi-timeframe confirmation helper. |
| `ui_components.py` | Sidebar widgets and the main Plotly chart renderer. |
| `exceptions.py` / `logger.py` | Shared error types and logger setup. |

## Conventions
- All magic numbers live in `config.py` as class attributes — don't inline new thresholds in the module files.
- UI-facing strings/labels are Turkish; keep new user-facing text consistent with that.
- Assets are keyed by a display name → ticker map (`varliklar.json`, falls back to `DEFAULT_COIN_MAP` in `config.py`).
