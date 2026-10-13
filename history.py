"""Uzun geçmiş ve V1 karar önbelleği (spec 0026).

Backtest ve portföy testi kripto için 2018-01-01'den başlayan günlük veriyi kullanır (Binance, sayfalı). Grafik ve
sinyal 1000 günlük veride kalır. Uzun veride V1 kararlarının hesabı yavaştır (~3200 gün ≈ 84 sn/varlık); kararlar
depoda saklanır ve sonraki çalışmalarda yalnız yeni günler hesaplanır (`build_v1_decisions(only_last=N)`, tam hesapla
birebir aynı kararlar, spec 0005 AC90).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

import market_map

logger = logging.getLogger("ProTrader")

LONG_START = date(2018, 1, 1)
LONG = "LONG"            # dönem seçimi: 2018'den
RECENT = "RECENT"        # dönem seçimi: uygulamanın günlük verisi (~1000 gün)
PERIOD_TEXT = {LONG: "2018'den (uzun geçmiş)", RECENT: "Son ~2 yıl (uygulamanın günlük verisi)"}
_CACHE_PREFIX = "v1_cache:"
_KLINES_URLS = ("https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines")
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class History:
    frame: object            # işlenmiş günlük tablo (göstergeler dahil) ya da None
    source: str
    note: str | None         # uzun veri alınamadıysa açıklama


def fetch_binance_daily(symbol: str, start: date = LONG_START, get=None):
    """Binance günlük mumları `start`'tan bugüne (1000'lik sayfalar). Kripto değilse ya da alınamazsa None.
    Sağlayıcının henüz kapanmamış mumu `attrs["provider_open"]` içinde işaretlenir (fetch_binance_simple ile aynı)."""
    pair = market_map.binance_symbol(symbol)
    if pair is None:
        return None
    if get is None:
        import requests
        from config import DataFetchConfig

        def get(url, params):
            return requests.get(url, params=params, headers=DataFetchConfig.HEADERS, timeout=10).json()
    for url in _KLINES_URLS:
        rows, cursor = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        try:
            while True:
                batch = get(url, {"symbol": pair, "interval": "1d", "startTime": cursor, "limit": 1000})
                if not isinstance(batch, list):
                    raise ValueError(f"beklenmeyen yanıt: {str(batch)[:80]}")
                if not batch:
                    break
                rows += batch
                cursor = int(batch[-1][0]) + _DAY_MS
                if len(batch) < 1000:
                    break
        except Exception as exc:  # sağlayıcı hatası: sıradaki adres, sonra 1000 günlük veriye dönüş
            logger.debug(f"Uzun geçmiş alınamadı ({url}): {exc}")
            continue
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime",
                                            "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"])
        frame.index = pd.to_datetime(pd.to_numeric(frame["OpenTime"]), unit="ms")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        provider_open = set(frame.index[pd.to_numeric(frame["CloseTime"]) >= now_ms])
        result = frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        result.attrs["provider_open"] = provider_open
        return result
    return None


#: Yahoo varlıklarında 2018 ortasına hazır olmak için (200 işlem günü ısınma) 2017 başından veri çekilir.
YAHOO_LONG_START = date(2017, 1, 1)
OUNCE_TO_GRAMS = 31.1035


def fetch_yahoo_daily(ticker: str, start: date = YAHOO_LONG_START):
    """Yahoo günlük mumlar `start`'tan bugüne (ayarlanmış fiyat). Alınamazsa None."""
    try:
        import yfinance as yf

        frame = yf.download(ticker, start=start.isoformat(), interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:  # sağlayıcı hatası: 1000 günlük veriye dönüş
        logger.debug(f"Yahoo uzun geçmiş alınamadı ({ticker}): {exc}")
        return None
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    if "Volume" not in frame.columns:
        frame["Volume"] = 0
    return frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])


def _long_raw(symbol: str):
    """Uzun ham günlük veri: kripto Binance 2018'den; ons altın GC=F, gram altın GC=F × USD/TRY, hisse ve kur Yahoo
    2017'den. Desteklenmeyen sembolde None."""
    if market_map.binance_symbol(symbol) is not None:
        return fetch_binance_daily(symbol)
    if symbol == "XAU_GOLD":
        return fetch_yahoo_daily("GC=F")
    if symbol == "GRAM_TRY":
        ons, usd = fetch_yahoo_daily("GC=F"), fetch_yahoo_daily("USDTRY=X")
        if ons is None or usd is None:
            return None
        rate = usd["Close"].reindex(ons.index).ffill()
        gram = ons[["Open", "High", "Low", "Close"]].mul(rate, axis=0) / OUNCE_TO_GRAMS
        gram["Volume"] = 0
        return gram.dropna()
    info = market_map.market_of(symbol)
    if symbol.endswith("=X") or (info is not None and info[0] in ("ABD", "BIST")):
        return fetch_yahoo_daily(symbol)
    return None


def load_history(source_pref: str, symbol: str, period: str = LONG) -> History:
    """Backtest verisi: `LONG` seçiliyse uzun geçmiş (kripto 2018'den, Yahoo varlıkları 2017'den; alınamazsa
    uygulamanın verisi + not); `RECENT` ya da desteklenmeyen sembolde uygulamanın günlük verisi (`get_market_data`)."""
    from data_fetchers import get_market_data

    symbol = market_map.canonical_symbol(symbol)
    if period == LONG:
        long_frame = _cached_long(symbol)
        if long_frame is not None:
            return History(long_frame, "Uzun geçmiş", None)
        frame, source = get_market_data(source_pref, symbol, "1d")
        note = f"{symbol}: uzun geçmiş alınamadı; uygulamanın son ~1000 günlük verisi kullanıldı."
        return History(frame, source, note if frame is not None else None)
    frame, source = get_market_data(source_pref, symbol, "1d")
    return History(frame, source, None)


def _long_processed(symbol: str):
    from data_fetchers import process_data

    raw = _long_raw(symbol)
    if raw is None or len(raw) < 260:
        return None
    frame, _ = process_data(raw, "Uzun geçmiş")
    return frame


try:  # Streamlit varsa uzun veri saatlik önbelleklenir (her yeniden çalıştırmada yeniden indirilmez)
    import streamlit as _st

    _cached_long = _st.cache_data(ttl=3600, show_spinner=False)(_long_processed)
except Exception:  # pragma: no cover - streamlit yoksa önbelleksiz
    _cached_long = _long_processed


def _fingerprint(weights, include_ml) -> str:
    from config import DecisionEngineConfig as cfg

    body = {"weights": sorted((str(k), str(v)) for k, v in (weights or {}).items()), "ml": bool(include_ml),
            "lookback": cfg.STABILITY_LOOKBACK, "v": 1}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _day(ts) -> str:
    return (ts.date() if hasattr(ts, "date") else ts).isoformat()


def cached_v1_decisions(symbol: str, frame, *, weights=None, include_ml=None, progress=None, variant="") -> dict:
    """V1 kararları; depodaki önbellekten, yalnız yeni günler hesaplanarak.

    Önbellek anahtarı sembol + `variant` (ör. dolara çevrilmiş veri "USD") + parmak izi (ağırlıklar, ML,
    kararlılık penceresi); farklı kullanımlar birbirinin kaydını ezmez. Verinin ilk günü değişmişse tam hesaplanır. Son mum (sağlayıcıda kapanmamış olabilir) önbelleğe yazılmaz, her seferinde
    yeniden hesaplanır. Depo okunamaz/yazılamazsa sessizce önbelleksiz hesaplanır."""
    from config import DecisionEngineConfig
    from storage import StorageAccessError, read_doc, write_doc
    from technical_analysis import build_v1_decisions

    if include_ml is None:
        include_ml = DecisionEngineConfig.V1_INCLUDE_ML
    if frame is None or len(frame) == 0:
        return {}
    fingerprint = _fingerprint(weights, include_ml)
    key = f"{_CACHE_PREFIX}{market_map.canonical_symbol(symbol)}:{variant}:{fingerprint}"
    first = _day(frame.index[0])
    try:
        stored = read_doc(key) or {}
    except StorageAccessError:
        stored = {}
    known = {}
    if stored.get("first") == first and stored.get("fp") == fingerprint:
        known = dict(stored.get("decisions") or {})
    days = [_day(ts) for ts in frame.index]
    last_known = max(known) if known else None
    todo = len(frame) if last_known is None else sum(1 for d in days if d > last_known)
    if progress is not None and todo > 400:
        progress(f"{symbol}: {todo} günün kararı hesaplanıyor (ilk seferde ~1–2 dakika; sonra saklanır)…")
    fresh = build_v1_decisions(frame, weights=weights, include_ml=include_ml,
                               only_last=None if last_known is None else max(todo, 1))
    by_day = {_day(ts): verdict for ts, verdict in fresh.items()}
    present = set(days)
    merged = {d: v for d, v in known.items() if d in present}
    merged.update(by_day)
    to_store = {d: v for d, v in merged.items() if d < days[-1]}     # son mum yazılmaz
    if to_store != known:
        try:
            write_doc(key, {"first": first, "fp": fingerprint, "decisions": to_store})
        except StorageAccessError:
            pass
    return {ts: merged[_day(ts)] for ts in frame.index if _day(ts) in merged}
