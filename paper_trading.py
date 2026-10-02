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
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import pandas as pd
from config import FileConfig, BacktestConfig, DecisionEngineConfig
from logger import logger
from trade_decisions import LOSS_COOLDOWN_BARS
from trade_execution import (
    Bar, BookState, CostAssumptions, OpenPosition, TradeSettings, advance_daily_bar,
)

PER_ASSET_BALANCE = 10000.0
# Daily-timeframe execution params — read from the same shared source as
# run_strategy_backtest (BacktestConfig.TIMEFRAME_PARAMS) so this journal
# can never silently drift from what the backtest actually validated.
_DAILY_PARAMS = BacktestConfig.TIMEFRAME_PARAMS["1d"]
SL_MULT = _DAILY_PARAMS["sl_mult"]
TP_MULT = _DAILY_PARAMS["tp_mult"]
TRAIL_BREAKEVEN = _DAILY_PARAMS["trail_breakeven"]
TRAIL_LOCK_PCT = _DAILY_PARAMS["trail_lock_pct"]
COOLDOWN_BARS = LOSS_COOLDOWN_BARS  # ortak kural: trade_decisions.LOSS_COOLDOWN_BARS
BILESEN_YOK = "BILESEN_YOK"  # zorunlu bileşen hesaplanamadı: yeni AL/SAT yok
MAX_BACKFILL = 10  # ML makes replay slow; cap catch-up bars per asset


def advance_v1_book(book, verdict, bar):
    """Advance a V1 paper position without targets or automatic trailing."""
    from trade_execution import evaluate_stop

    result = deepcopy(book)
    position = result.get("position")
    if position is None:
        return result
    stop_fill = evaluate_stop(bar, Decimal(position["stop"]), bar.at)
    if stop_fill is not None:
        exit_price, reason = stop_fill.base_price, "STOP"
    elif "SAT" in verdict:
        exit_price, reason = bar.open, "SAT"
    else:
        return result
    quantity = Decimal(position["quantity"])
    result["cash"] = str(Decimal(result.get("cash", "0")) + quantity * exit_price)
    result.setdefault("trades", []).append({
        "exit": str(exit_price), "at": bar.at.isoformat(), "reason": reason,
    })
    result["position"] = None
    return result


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


def _blank_book(capital=PER_ASSET_BALANCE):
    return {"balance": str(Decimal(str(capital))), "initial_balance": str(Decimal(str(capital))),
            "position": None, "cooldown": 0,
            "trades": [], "last_date": None, "first_price": None,
            "pending": None, "quantity_step": None,
            "trade_notional": 1000.0, "currency": "USD/USDT",
            "costs": {"spread_bps": None, "slippage_bps": None, "commission_pct": None},
            "last_block": None}


def _book_settings(book):
    """Sanal defterin işlem varsayımları; eksik/bilinmeyen alan None kalır."""
    def optional(value):
        return None if value in (None, "") else Decimal(str(value))

    costs = book.get("costs") or {}
    return TradeSettings(
        Decimal(str(book.get("trade_notional", 1000.0))),
        optional(book.get("quantity_step")),
        CostAssumptions(optional(costs.get("spread_bps")), optional(costs.get("slippage_bps")),
                        optional(costs.get("commission_pct"))),
    )


def _bar_time(day):
    return datetime.fromisoformat(str(day))


def advance_pending_daily_decision(book, bar):
    """Önceki kapanmış mumun kararını bu mumun açılışında uygular, stopunu sınar.

    Geçmiş testle aynı ortak motoru kullanır (`trade_execution.advance_daily_bar`);
    komisyon, makas, kayma, miktar adımı ve zarar sonrası bekleme kullanıcı
    varsayımlarından gelir, gizli bir oran yoktur (Q1/Q7).
    """
    pending = book.get("pending")
    if not pending or str(bar["date"]) <= str(pending["known_at"]):
        return False

    settings = _book_settings(book)
    raw = book.get("position")
    position = None
    if raw is not None:
        position = OpenPosition(
            _bar_time(raw["entry_date"]), Decimal(str(raw["entry"])), Decimal(str(raw["qty"])),
            Decimal(str(raw["cost"])), Decimal(str(raw.get("entry_fee", "0"))),
            Decimal(str(raw["sl"])),
        )
    state = BookState(Decimal(str(book["balance"])), position, int(book.get("cooldown", 0)))
    at = _bar_time(bar["date"])
    trade_bar = Bar(at, Decimal(str(bar["open"])), Decimal(str(bar["high"])),
                    Decimal(str(bar["low"])), Decimal(str(bar["close"])))
    atr = Decimal(str(pending["atr"]))
    step = advance_daily_bar(state, trade_bar, str(pending["verdict"]),
                             atr if atr.is_finite() else None, settings)

    for item in step.trades:
        invested = item.entry * item.quantity
        book["trades"].append({
            "entry": str(item.entry), "exit": str(item.exit),
            "entry_date": item.entry_at.date().isoformat(), "exit_date": str(bar["date"]),
            "qty": str(item.quantity), "pnl": str(item.pnl),
            "pnl_pct": str((item.pnl / invested * Decimal("100")).quantize(Decimal("0.01")))
            if invested else "0.00",
            "reason": item.reason, "net_verified": item.net_verified,
            "provisional": item.provisional,
        })
    book["balance"] = str(state.cash)
    book["cooldown"] = state.cooldown
    open_position = state.position
    book["position"] = None if open_position is None else {
        "entry": str(open_position.entry), "entry_date": open_position.entry_at.date().isoformat(),
        "qty": str(open_position.quantity), "cost": str(open_position.cost),
        "entry_fee": str(open_position.entry_fee), "sl": str(open_position.stop), "tp": None,
    }
    if step.blocked:
        book["last_block"] = step.blocked      # alım neden yapılmadı (AC102 görünürlüğü)
    elif step.opened:
        book["last_block"] = None
    book["pending"] = None
    return bool(step.trades or step.opened)


def _step_book(book, verdict, price, atr, date_str):
    """Legacy storage adapter for the approved fixed-stop V1 behavior."""
    fee = BacktestConfig.FEE_RATE
    pos = book["position"]
    if pos is not None:
        reason = None
        if price <= pos["sl"]:
            reason = "STOP"
        elif "SAT" in verdict:
            reason = "SAT"
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
            "sl": price - atr * 2.5, "tp": None,
        }
        book["balance"] -= cost


def run_paper_update(coin_map, source_pref="Binance", progress_callback=None, paper_settings=None):
    """Record any unrecorded closed daily bars for every asset. Returns a
    short status dict: {"new_rows": int, "assets": int, "errors": [names]}."""
    from data_fetchers import get_market_data
    from signal_engine import generate_stable_signal
    from market_validation import policy_for_symbol, validate_market_data
    from weight_profiles import get_weights_for_symbol

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
            if df is None:
                errors.append(name); continue

            evaluation_time = pd.Timestamp.now(tz="UTC").to_pydatetime()
            policy = policy_for_symbol(sym, evaluation_time)
            validation = validate_market_data(
                df, evaluation_time, policy,
                provider_open=df.attrs.get("provider_open", set()),
                require_components=True,
            )
            if not validation.is_valid:
                detail = ", ".join(validation.missing_components)
                errors.append(f"{name}: {validation.reason}" + (f" ({detail})" if detail else ""))
                continue

            asset_settings = (paper_settings or {}).get(name, {})
            capital = Decimal(str(asset_settings.get("capital", PER_ASSET_BALANCE)))
            book = state["assets"].setdefault(name, _blank_book(capital))
            for key, value in _blank_book().items():
                book.setdefault(key, deepcopy(value))
            if asset_settings:
                book["trade_notional"] = str(asset_settings.get("trade_notional", book["trade_notional"]))
                book["currency"] = str(asset_settings.get("currency", book["currency"]))
                quantity_step = asset_settings.get("quantity_step")
                book["quantity_step"] = None if quantity_step in (None, "") else str(quantity_step)
                # Maliyet varsayımları geçmiş testle aynı kullanıcı ayarından gelir.
                # Boş = bilinmiyor; 0 = açıkça sıfır (Q7).
                book["costs"] = {
                    key: None if asset_settings.get(key) in (None, "") else str(asset_settings[key])
                    for key in ("spread_bps", "slippage_bps", "commission_pct")
                }
            closed = validation.usable

            # Which closed bars still need recording?
            if book["last_date"] is None:
                todo = [len(closed) - 1]                     # first run: today only
            else:
                todo = [k for k in range(len(closed))
                        if str(closed.index[k].date()) > book["last_date"]][-MAX_BACKFILL:]

            for k in todo:
                slice_df = closed.iloc[:k + 1]
                sig = generate_stable_signal(
                    slice_df, "1d", weights=get_weights_for_symbol(sym), data_is_closed=True,
                    strict_components=True,
                )
                # Hesaplanamayan bileşen: yeni AL/SAT yazılmaz (spec 0003 AC120).
                verdict = BILESEN_YOK if sig.unavailable_components else sig.verdict
                price = float(closed['Close'].iloc[k])
                atr = float(closed['ATR'].iloc[k]) if 'ATR' in closed.columns else price * 0.02
                date_str = str(closed.index[k].date())
                is_backfill = k < len(closed) - 1

                advance_pending_daily_decision(book, {
                    "date": date_str,
                    "open": float(closed['Open'].iloc[k]),
                    "high": float(closed['High'].iloc[k]),
                    "low": float(closed['Low'].iloc[k]),
                    "close": price,
                })
                book["pending"] = {
                    "verdict": verdict, "atr": atr, "known_at": date_str,
                }
                if book["first_price"] is None:
                    book["first_price"] = price
                state["journal"].append({
                    "date": date_str, "asset": name, "verdict": verdict,
                    "unavailable_components": list(sig.unavailable_components),
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
        pos_value = Decimal(str(pos["qty"])) * Decimal(str(lp)) if pos else Decimal("0")
        equity = Decimal(str(book["balance"])) + pos_value
        initial_balance = Decimal(str(book.get("initial_balance", PER_ASSET_BALANCE)))
        ret = float((equity / initial_balance - 1) * 100)
        net_verified = (book.get("costs") or {}).get("commission_pct") is not None
        bh = (lp / book["first_price"] - 1) * 100 if book.get("first_price") else 0.0
        days = sum(1 for j in state["journal"] if j["asset"] == name)
        rows.append({
            "Varlık": name, "Gün": days, "İşlem": len(book["trades"]),
            "Pozisyon": "LONG" if pos else "-",
            "Bakiye": round(float(equity), 2), "Para Birimi": book.get("currency", "USD/USDT"),
            "Net Doğrulandı": "Evet" if net_verified else "Hayır (komisyon bilinmiyor)",
            "Getiri (%)": round(ret, 2),
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

    # Varlık listesi artık belge deposundan okunur (spec 0004); kayıt yoksa
    # yerleşik liste kullanılır.
    from assets import load_assets

    try:
        cmap = load_assets() or DEFAULT_COIN_MAP.copy()
    except Exception as exc:
        print(f"Varlik listesi okunamadi, yerlesik liste kullanilacak: {exc}")
        cmap = DEFAULT_COIN_MAP.copy()

    status = run_paper_update(cmap)
    print(f"paper update: {status['new_rows']} yeni kayıt, {status['assets']} varlık, hatalar: {status['errors'] or 'yok'}")
    df, totals = paper_report()
    if len(df):
        print(df.to_string(index=False))
    print(f"başlangıç: {totals['başlangıç']} | ort. getiri %{totals['toplam_getiri_pct']} | al&tut %{totals['al_tut_pct']} | {totals['kayıt']} kayıt")
