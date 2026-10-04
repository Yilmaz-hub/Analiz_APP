"""Ekrandan bağımsız ileri dönem takip koşucusu (spec 0005, Adım 7 / S6, Q06a).

Çalışan bir Streamlit sunucusu/oturumu gerekmez; ancak canlı veri yolu `data_fetchers`'ı
içe aktarır ve o modül `st.cache_data` kullanır (oturumsuz çalışır, önbellek bellekte kalır).

GitHub Actions zamanlanmış görevi her gün çalıştırır:

    python -m forward_runner

Varlık listesi ve işlem varsayımları uygulamanın kullandığı aynı depodan
(`ANALIZ_APP_DB_URL`, Neon Postgres) okunur; kararlar `forward_tracker` tablolarına
yazılır. Kaçan günler bir sonraki çalışmada en fazla
`MAX_BACKFILL_DAYS` kadar tamamlanır ve "sonradan oluşturuldu" işaretlenir.

Test / tatbikat seçenekleri: `--now` saati sabitler (bu durumda gözlemler
"hızlandırılmış" sayılır ve yeterlilik sayacına girmez, AC87), `--fixture DIR`
ağ yerine `DIR/<SEMBOL>.csv` okur, `--no-ml` ML bileşenini kapatır (sürüm adına
`/ML-YOK` eklenir; ML'li kararlarla karışmaz).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import forward_tracker as ft
import market_map
from config import ForwardConfig


@dataclass
class RunReport:
    recorded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provider_seconds: float = 0.0
    compute_seconds: float = 0.0


def _fingerprint(asset_name: str) -> str:
    """Karar kuralı (`DecisionEngineConfig`) ve varlığın işlem varsayımlarının kısa parmak izi (B16, AC103)."""
    from config import DecisionEngineConfig
    from trade_settings import load_raw

    rules = {key: getattr(DecisionEngineConfig, key) for key in sorted(vars(DecisionEngineConfig))
             if key.isupper()}
    body = json.dumps({"rules": rules, "assumptions": load_raw(asset_name)},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]


def _version(asset_name: str, include_ml: bool, real_clock: bool) -> tuple[str, object]:
    import strategy_candidates as sc

    active = sc.active_strategy()
    candidate = None if active == sc.REFERENCE_STRATEGY else sc.find_by_strategy(active)
    label = sc.REFERENCE_STRATEGY if candidate is None else f"ADAY-{active[:12]}"
    label = f"{label}~{_fingerprint(asset_name)}"
    if not include_ml:
        label += "/ML-YOK"
    if not real_clock:
        label += "/TATBIKAT"   # hızlandırılmış koşu gerçek kayıtla karışmaz (AC87, AC92)
    return label, candidate


def _fixture_fetch(directory: Path):
    import pandas as pd

    from data_fetchers import process_data

    def fetch(symbol: str):
        raw = pd.read_csv(directory / f"{symbol}.csv", index_col=0, parse_dates=True)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        return process_data(raw, "Fixture")

    return fetch


def _live_fetch(symbol: str):
    from data_fetchers import get_market_data

    return get_market_data("Binance", symbol, "1d")


def _day(ts):
    return ts.date() if hasattr(ts, "date") else ts


def _candle(row) -> dict:
    return {key: str(Decimal(str(row[column]))) for key, column in
            (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"))}


def _assumptions(asset_name: str) -> dict:
    """Gerçekleşme varsayımları: bilinmeyen alan "bilinmiyor" yazar, sıfır sayılmaz (R16)."""
    from trade_settings import load_raw

    raw = load_raw(asset_name)

    def known(name):
        value = str(raw.get(name, "")).strip()
        return value if value else "bilinmiyor"

    return {"dolum": "ertesi gün açılışı", "stop": "2,5 ATR başlangıç stopu",
            "komisyon": known("commission_pct"), "makas": known("spread_bps"),
            "kayma": known("slippage_bps"), "miktar adımı": known("quantity_step"),
            "sermaye": known("capital"), "işlem tutarı": known("notional")}


def process_asset(name: str, symbol: str, frame, source: str, *, now: datetime, real_clock: bool,
                  include_ml: bool, report: RunReport, clock=None) -> None:
    import time

    from regime_classifier import classify_frame, classify_latest
    from technical_analysis import build_v1_decisions, run_v1_strategy_backtest
    from trade_settings import load_raw, parse_settings

    clock = clock or (lambda: now)
    info = market_map.market_of(symbol)
    if info is None:
        report.notes.append(f"{symbol}: piyasası tanınmıyor, atlandı.")
        return
    market = info[0]
    started = time.perf_counter()
    version, candidate = _version(name, include_ml, real_clock)

    closed = [(pos, ts) for pos, ts in enumerate(frame.index)
              if ft.available_at(market, _day(ts)) <= now]
    if not closed:
        report.notes.append(f"{symbol}: kapanmış günlük mum yok.")
        return
    known = ft.decisions(symbol, version)
    last = known[-1].candle_day if known else None
    todo = [(pos, ts) for pos, ts in closed if last is None or _day(ts) > last]
    todo = todo[-ForwardConfig.MAX_BACKFILL_DAYS:] if last is not None else todo[-1:]

    if todo:
        # Yalnız yeni günlerin kararı hesaplanır; kayıtlı günler yeniden üretilmez (AC90).
        v1 = build_v1_decisions(frame, include_ml=include_ml, only_last=len(frame) - todo[0][0])
        if candidate is None:
            decisions = v1
        else:
            import strategy_candidates as sc
            decisions = sc.candidate_decisions(frame, v1, candidate, classify_frame(frame))
        assumptions = _assumptions(name)
        for pos, ts in todo:
            label, regime_version = classify_latest(frame.iloc[:pos + 1])
            day = _day(ts)
            evaluated_at = clock()
            inserted = ft.record(ft.ForwardDecision(
                asset=symbol, strategy_version=version, candle_day=day,
                decision=decisions.get(ts, "BEKLE"), source=source, evaluated_at=evaluated_at,
                candle=_candle(frame.iloc[pos]), assumptions=assumptions,
                on_time=ft.is_on_time(market, day, evaluated_at), real_clock=real_clock,
                regime=label, regime_version=regime_version))
            if inserted:
                report.recorded.append(f"{symbol} {day}")

    # Kayıtlı son günlerin mumu sağlayıcıda sonradan değiştiyse karar korunur, revizyon yazılır (AC91).
    by_day = {_day(ts): pos for pos, ts in closed}
    for earlier in ft.decisions(symbol, version)[-ForwardConfig.MAX_BACKFILL_DAYS:]:
        pos = by_day.get(earlier.candle_day)
        if pos is None:
            continue
        current = _candle(frame.iloc[pos])
        if current != dict(earlier.candle):
            ft.record(replace(earlier, candle=current))
            report.notes.append(f"{symbol} {earlier.candle_day}: mum sağlayıcıda revize edildi; karar değişmedi.")

    tracked = ft.decisions(symbol, version)
    parsed = parse_settings(load_raw(name))
    if not parsed.ok or parsed.settings.quantity_step is None:
        report.notes.append(f"{symbol}: miktar adımı/varsayımlar eksik; sanal işlem hesaplanmadı.")
    elif tracked:
        first = tracked[0].candle_day
        window = frame[[_day(ts) >= first for ts in frame.index]]
        tracked_days = {d.candle_day: d.decision for d in tracked}
        decision_map = {ts: tracked_days[_day(ts)] for ts in window.index if _day(ts) in tracked_days}
        result = run_v1_strategy_backtest(
            window, decision_map, initial_cash=parsed.capital,
            trade_notional=parsed.settings.notional, quantity_step=parsed.settings.quantity_step,
            costs=parsed.settings.costs)
        ft.save_trades(symbol, version, [
            {"entry_day": _day(t["entry_at"]), "exit_day": _day(t["exit_at"]), "pnl": t["pnl"]}
            for t in result["trades"]])
    report.compute_seconds += time.perf_counter() - started


def run(now: datetime, *, real_clock: bool, fetch, include_ml: bool,
        symbols: list[str] | None = None, clock=None) -> RunReport:
    import time

    from assets import load_assets

    clock = clock or ((lambda: datetime.now(timezone.utc)) if real_clock else (lambda: now))
    report = RunReport()
    assets = {s: s for s in symbols} if symbols else load_assets()
    for name, symbol in assets.items():
        started = time.perf_counter()
        try:
            frame, source = fetch(symbol)
        except Exception as exc:  # sağlayıcı hatası görünür kalır, diğer varlıklar sürer
            report.notes.append(f"{symbol}: veri alınamadı ({type(exc).__name__}).")
            continue
        finally:
            report.provider_seconds += time.perf_counter() - started
        if frame is None or len(frame) == 0:
            report.notes.append(f"{symbol}: veri alınamadı ({source}).")
            continue
        process_asset(name, symbol, frame, source, now=now, real_clock=real_clock,
                      include_ml=include_ml, report=report, clock=clock)
    ft.record_run(clock(), not any("alınamadı" in n for n in report.notes),
                  "; ".join(report.notes) or "tamam")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="İleri dönem sanal takip koşucusu")
    parser.add_argument("--now", help="Sabit saat (ISO 8601). Verilirse gözlem hızlandırılmış sayılır.")
    parser.add_argument("--fixture", type=Path, help="Ağ yerine <DIR>/<SEMBOL>.csv kullan.")
    parser.add_argument("--no-ml", action="store_true", help="ML bileşenini kapat.")
    parser.add_argument("--symbols", help="Virgülle ayrılmış semboller (varsayılan: varlık listesi).")
    args = parser.parse_args(argv)
    now = (datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now
           else datetime.now(timezone.utc))
    report = run(now, real_clock=args.now is None,
                 fetch=_fixture_fetch(args.fixture) if args.fixture else _live_fetch,
                 include_ml=not args.no_ml,
                 symbols=args.symbols.split(",") if args.symbols else None)
    for line in report.recorded:
        print(f"kaydedildi: {line}")
    for line in report.notes:
        print(f"not: {line}")
    print(f"sağlayıcı bekleme: {report.provider_seconds:.2f} sn · hesaplama: "
          f"{report.compute_seconds:.2f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
