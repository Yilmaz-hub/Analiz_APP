"""
AI TAHMİN KARNESİ (prediction journal)

Records every AI forecast the UI displays, evaluates each one when its
horizon elapses, and reports a per-(symbol, timeframe) track record — so the
user can see what the AI predicted N bars ago, what actually happened, and
how much the current prediction deserves to be trusted.

Storage: FileConfig.PREDICTIONS_LOG_FILE, atomic tmp+replace writes (same
pattern as weight_profiles.py). One record per (symbol, timeframe, anchor
bar): Streamlit reruns and cache refreshes within the same bar do not
duplicate; the FIRST forecast seen for a bar is the one kept (that is the
one the user acted on).
"""
import json
import os

import pandas as pd

from config import FileConfig
from logger import logger

# Keep the journal bounded per (symbol, timeframe) key.
MAX_RECORDS_PER_KEY = 100


def _load():
    f = FileConfig.PREDICTIONS_LOG_FILE
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception as e:
            logger.error(f"Prediction log load error: {e}")
    return {}


def _save(data):
    target = FileConfig.PREDICTIONS_LOG_FILE
    tmp = f"{target}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, target)
    except Exception as e:
        logger.error(f"Prediction log save error: {e}")


def _key(symbol, timeframe):
    return f"{symbol}|{timeframe}"


def record_prediction(symbol, timeframe, df, f_dates, f_prices, ai_score):
    """Persist the forecast anchored to the bar it was seeded on (df's last
    row — same row calculate_smart_prediction_FIXED walks forward from)."""
    if df is None or len(df) < 2 or not f_prices:
        return
    try:
        anchor_time = str(df.index[-1])
        anchor_price = float(df['Close'].iloc[-1])
        if anchor_price <= 0:
            return

        data = _load()
        recs = data.setdefault(_key(symbol, timeframe), [])
        if any(r.get("anchor") == anchor_time for r in recs):
            return  # already journaled for this bar

        end_price = float(f_prices[-1])
        recs.append({
            "anchor": anchor_time,
            "anchor_price": round(anchor_price, 8),
            "horizon": len(f_prices),
            "pred_end_price": round(end_price, 8),
            "pred_change_pct": round((end_price / anchor_price - 1) * 100, 3),
            "ai_score": round(float(ai_score), 2),
            "outcome": None,    # filled by evaluate_predictions at maturity
            "progress": None,   # refreshed by evaluate_predictions until then
        })
        del recs[:-MAX_RECORDS_PER_KEY]
        _save(data)
    except Exception as e:
        logger.debug(f"record_prediction failed: {e}")


def evaluate_predictions(symbol, timeframe, df):
    """Score every journaled prediction for (symbol, timeframe) against the
    realized closes in df. Predictions whose horizon has fully elapsed get a
    final `outcome` (direction hit + error); younger ones get a `progress`
    snapshot (bars elapsed, realized change so far). Returns the record list,
    oldest first."""
    if df is None or len(df) == 0:
        return []
    data = _load()
    recs = data.get(_key(symbol, timeframe), [])
    changed = False
    for r in recs:
        try:
            if r.get("outcome") is not None:
                continue
            anchor = pd.Timestamp(r["anchor"])
            future = df[df.index > anchor]
            if len(future) == 0:
                continue
            horizon = int(r["horizon"])
            n = min(len(future), horizon)
            realized = float(future['Close'].iloc[n - 1])
            real_pct = (realized / r["anchor_price"] - 1) * 100
            if n >= horizon:
                pred_pct = r["pred_change_pct"]
                r["outcome"] = {
                    "realized_change_pct": round(real_pct, 3),
                    "direction_hit": bool((pred_pct >= 0) == (real_pct >= 0)),
                    "abs_error_pct": round(abs(pred_pct - real_pct), 3),
                }
                r["progress"] = None
                changed = True
            else:
                r["progress"] = {
                    "bars_elapsed": n,
                    "realized_change_pct": round(real_pct, 3),
                }
                changed = True
        except Exception as e:
            logger.debug(f"evaluate_predictions record skipped: {e}")
    if changed:
        _save(data)
    return recs


def get_track_record(recs):
    """Summarize a record list from evaluate_predictions: how many matured,
    direction hit rate, average absolute error. hit_rate/avg_err are None
    until at least one prediction has matured."""
    done = [r for r in recs if r.get("outcome")]
    if not done:
        return {"n": 0, "hit_rate": None, "avg_err": None}
    hits = sum(1 for r in done if r["outcome"]["direction_hit"])
    return {
        "n": len(done),
        "hit_rate": hits / len(done) * 100,
        "avg_err": sum(r["outcome"]["abs_error_pct"] for r in done) / len(done),
    }
