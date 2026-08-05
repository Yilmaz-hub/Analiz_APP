"""
Paper-trading verification (built 2026-07-15).

Purpose: record the LIVE signal (ML dimension included — the one component
backtests can't reach) once per closed daily bar, per asset, and simulate
positions with the exact daily-backtest rules. After a few weeks the journal
answers: does live behavior match the backtest's statistical profile?

Design:
- Idempotent per (asset, closed-bar date): safe to run from the daily
  scheduled task, the app button, or both. Missed days are backfilled by
  replaying historical slices — legitimate because the whole pipeline incl.
  the ML models (random_state=42) is deterministic — and marked
  "backfilled": true so forward-recorded and reconstructed rows stay
  distinguishable.
- Each asset runs an independent paper book (PER_ASSET_BALANCE), mirroring
  the per-asset backtests: entry on AL at the closed bar's close, ATR
  SL/TP + breakeven/profit-lock trailing, exit on signal loss, fees per side,
  loss cooldown — the same daily parameters as run_strategy_backtest.

State lives in FileConfig.PAPER_FILE. CLI: run this file directly to update
and print the report (this is what the Windows scheduled task calls).
"""
import json
import os
import pandas as pd
from config import FileConfig, BacktestConfig, DecisionEngineConfig
from logger import logger

PER_ASSET_BALANCE = 1000.0
# Daily-timeframe execution params — keep in sync with run_strategy_backtest
SL_MULT = 2.5
TP_MULT = 2.8
TRAIL_BREAKEVEN = 1.0
TRAIL_LOCK_PCT = 0.5
COOLDOWN_BARS = 2
MAX_BACKFILL = 10  # ML makes replay slow; cap catch-up bars per asset


def _load_state():
    f = FileConfig.PAPER_FILE
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception as e:
            logger.error(f"Paper state load error: {e}")
    return {"created": None, "assets": {}, "journal": []}


def _save_state(state):
    # Atomic write (temp file + os.replace) so the live app and the
    # scheduled task writing concurrently can't leave paper_trading.json
    # half-written or corrupted.
    target = FileConfig.PAPER_FILE
    tmp = f"{target}.tmp"
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, target)


def _blank_book():
    return {"balance": PER_ASSET_BALANCE, "position": None, "cooldown": 0,
            "trades": [], "last_date": None, "first_price": None}


def _step_book(book, verdict, price, atr, date_str):
    """Advance one asset's paper book by one closed bar (backtest rules)."""
    fee = BacktestConfig.FEE_RATE
    pos = book["position"]
    if pos is not None:
        if price > pos["highest"]:
            pos["highest"] = price
        profit_distance = pos["highest"] - pos["entry"]
        if profit_distance > atr * 2.0:
            pos["sl"] = max(pos["sl"], pos["entry"] + profit_distance * TRAIL_LOCK_PCT)
        elif profit_distance > atr * TRAIL_BREAKEVEN:
            pos["sl"] = max(pos["sl"], pos["entry"])

        reason = None
        if price <= pos["sl"]:
            reason = "SL"
        elif price >= pos["tp"]:
            reason = "TP"
        elif "AL" not in verdict:
            reason = "Sinyal"
        if reason:
            proceeds = pos["qty"] * price * (1 - fee)
            pnl = proceeds - pos["cost"]
            book["balance"] += proceeds
            book["trades"].append({
                "entry": pos["entry"], "exit": price, "entry_date": pos["entry_date"],
                "exit_date": date_str, "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / pos["cost"] * 100, 2), "reason": reason,
            })
            if pnl < 0:
                book["cooldown"] = COOLDOWN_BARS
            book["position"] = None
    elif book["cooldown"] > 0:
        book["cooldown"] -= 1
    elif "AL" in verdict and book["balance"] > 0 and atr > 0:
        qty = (book["balance"] * 0.95) / price
        cost = qty * price * (1 + fee)
        book["position"] = {
            "entry": price, "entry_date": date_str, "qty": qty, "cost": cost,
            "highest": price,
            "sl": price - atr * SL_MULT, "tp": price + atr * TP_MULT,
        }
        book["balance"] -= cost


def run_paper_update(coin_map, source_pref="Binance", progress_callback=None):
    """Record any unrecorded closed daily bars for every asset. Returns a
    short status dict: {"new_rows": int, "assets": int, "errors": [names]}."""
    from data_fetchers import get_market_data
    from signal_engine import generate_stable_signal

    state = _load_state()
    if state["created"] is None:
        state["created"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    new_rows, errors = 0, []
    items = list(coin_map.items())
    for idx, (name, sym) in enumerate(items):
        if progress_callback:
            progress_callback(idx / max(len(items), 1), name)
        try:
            df, _ = get_market_data(source_pref, sym, "1d")
            if df is None or len(df) < 130:
                errors.append(name); continue

            book = state["assets"].setdefault(name, _blank_book())
            closed = df.iloc[:-1] if DecisionEngineConfig.DROP_UNCLOSED_CANDLE else df

            # Which closed bars still need recording?
            if book["last_date"] is None:
                todo = [len(closed) - 1]                     # first run: today only
            else:
                todo = [k for k in range(len(closed))
                        if str(closed.index[k].date()) > book["last_date"]][-MAX_BACKFILL:]

            for k in todo:
                # Live-signal input as of bar k's close: bars 0..k+1, where
                # k+1 was the then-forming candle the engine drops itself.
                slice_df = df.iloc[:k + 2]
                sig = generate_stable_signal(slice_df, "1d")
                price = float(closed['Close'].iloc[k])
                atr = float(closed['ATR'].iloc[k]) if 'ATR' in closed.columns else price * 0.02
                date_str = str(closed.index[k].date())
                is_backfill = k < len(closed) - 1

                _step_book(book, sig.verdict, price, atr, date_str)
                if book["first_price"] is None:
                    book["first_price"] = price
                state["journal"].append({
                    "date": date_str, "asset": name, "verdict": sig.verdict,
                    "raw_verdict": sig.raw_verdict, "score": round(sig.final_score, 1),
                    "confidence": round(sig.confidence, 0), "bars_held": sig.bars_held,
                    "price": price, "backfilled": is_backfill,
                })
                book["last_date"] = date_str
                new_rows += 1
        except Exception as e:
            logger.error(f"Paper update error {name}: {e}")
            errors.append(name)

    _save_state(state)
    return {"new_rows": new_rows, "assets": len(state["assets"]), "errors": errors}


def paper_report():
    """Per-asset paper performance vs hold-since-start. Returns (DataFrame, totals dict)."""
    state = _load_state()
    rows = []
    last_price = {}
    for j in state["journal"]:
        last_price[j["asset"]] = j["price"]
    for name, book in state["assets"].items():
        lp = last_price.get(name, 0.0)
        pos = book["position"]
        pos_value = pos["qty"] * lp if pos else 0.0
        equity = book["balance"] + pos_value
        ret = (equity / PER_ASSET_BALANCE - 1) * 100
        bh = (lp / book["first_price"] - 1) * 100 if book.get("first_price") else 0.0
        days = sum(1 for j in state["journal"] if j["asset"] == name)
        rows.append({
            "Varlık": name, "Gün": days, "İşlem": len(book["trades"]),
            "Pozisyon": "LONG" if pos else "-",
            "Bakiye ($)": round(equity, 2), "Getiri (%)": round(ret, 2),
            "Al&Tut (%)": round(bh, 2),
        })
    df = pd.DataFrame(rows)
    totals = {
        "başlangıç": state.get("created"),
        "toplam_getiri_pct": round(df["Getiri (%)"].mean(), 2) if len(df) else 0.0,
        "al_tut_pct": round(df["Al&Tut (%)"].mean(), 2) if len(df) else 0.0,
        "kayıt": len(state["journal"]),
    }
    return df, totals


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from config import DEFAULT_COIN_MAP

    cmap = DEFAULT_COIN_MAP.copy()
    if os.path.exists(FileConfig.ASSETS_FILE):
        try:
            with open(FileConfig.ASSETS_FILE, 'r', encoding='utf-8') as fh:
                cmap = json.load(fh)
        except Exception:
            pass

    status = run_paper_update(cmap)
    print(f"paper update: {status['new_rows']} yeni kayıt, {status['assets']} varlık, hatalar: {status['errors'] or 'yok'}")
    df, totals = paper_report()
    if len(df):
        print(df.to_string(index=False))
    print(f"başlangıç: {totals['başlangıç']} | ort. getiri %{totals['toplam_getiri_pct']} | al&tut %{totals['al_tut_pct']} | {totals['kayıt']} kayıt")
