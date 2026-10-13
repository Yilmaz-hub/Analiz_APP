"""Spec 0026 — uzun geçmiş (kripto 2018'den, Yahoo varlıkları 2017'den) ve V1 karar önbelleği."""
from datetime import date

import pandas as pd
import pytest

import history

_DAY_MS = 86_400_000


def _klines(start_ms, count):
    return [[start_ms + i * _DAY_MS, "1", "2", "0.5", "1.5", "10", start_ms + i * _DAY_MS + _DAY_MS - 1,
             "0", 0, "0", "0", "0"] for i in range(count)]


def test_ac01_binance_history_is_paged_from_2018_and_falls_back_to_the_next_address():
    """AC01 — Binance günlük veri 2018-01-01'den 1000'lik sayfalarla birleştirilir; ilk adres hata verirse sıradaki
    denenir; kripto olmayan sembolde None."""
    calls = []
    first_ms = int(pd.Timestamp(date(2018, 1, 1), tz="UTC").timestamp() * 1000)

    def get(url, params):
        calls.append((url, params["startTime"]))
        if "data-api" in url:
            raise ConnectionError("engelli")
        offset = (params["startTime"] - first_ms) // _DAY_MS
        return _klines(params["startTime"], min(1000, 2500 - offset))

    frame = history.fetch_binance_daily("BTC-USD", get=get)
    assert len(frame) == 2500 and frame.index[0] == pd.Timestamp("2018-01-01") and frame.index.is_unique
    assert [start for url, start in calls if "data-api" not in url] == [first_ms, first_ms + 1000 * _DAY_MS,
                                                                        first_ms + 2000 * _DAY_MS]
    assert history.fetch_binance_daily("AAPL", get=get) is None


def test_ac02_long_period_uses_long_history_and_falls_back_with_a_note(monkeypatch):
    """AC02 — "2018'den" seçiliyse uzun geçmiş kullanılır; alınamazsa uygulamanın verisi kullanılır ve not yazılır;
    "son ~2 yıl" uygulamanın verisidir."""
    import data_fetchers

    long_frame, short_frame = pd.DataFrame({"Close": [1.0] * 3000}), pd.DataFrame({"Close": [2.0] * 1000})
    monkeypatch.setattr(data_fetchers, "get_market_data", lambda *a, **k: (short_frame, "Binance"))
    monkeypatch.setattr(history, "_cached_long", lambda symbol: long_frame)
    assert history.load_history("Binance", "BTC-USD", history.LONG).frame is long_frame
    assert history.load_history("Binance", "BTC-USD", history.RECENT).frame is short_frame
    monkeypatch.setattr(history, "_cached_long", lambda symbol: None)
    fallback = history.load_history("Binance", "BTC-USD", history.LONG)
    assert fallback.frame is short_frame and "uzun geçmiş alınamadı" in fallback.note


def test_ac03_gram_gold_long_history_is_ounce_times_rate_over_grams(monkeypatch):
    """AC03 — Uzun geçmişte gram altın = ons (GC=F) × USD/TRY ÷ 31,1035; ons altın GC=F, hisse kendi kodu."""
    idx = pd.date_range("2017-01-02", periods=3, freq="D")
    data = {"GC=F": pd.DataFrame({"Open": 1000.0, "High": 1010.0, "Low": 990.0, "Close": 1000.0, "Volume": 0}, index=idx),
            "USDTRY=X": pd.DataFrame({"Open": 3.0, "High": 3.0, "Low": 3.0, "Close": 3.11035, "Volume": 0}, index=idx),
            "AAPL": pd.DataFrame({"Open": 5.0, "High": 5.0, "Low": 5.0, "Close": 5.0, "Volume": 0}, index=idx)}
    monkeypatch.setattr(history, "fetch_yahoo_daily", lambda ticker, start=None: data[ticker])
    gram = history._long_raw("GRAM_TRY")
    assert list(gram["Close"].round(6)) == [100.0] * 3
    assert history._long_raw("XAU_GOLD") is data["GC=F"] and history._long_raw("AAPL") is data["AAPL"]


def test_ac04_v1_cache_computes_only_new_days_and_matches_a_full_computation(monkeypatch, processed_df):
    """AC04 — V1 kararları depoda saklanır: ikinci çalıştırmada yalnız yeni günler hesaplanır ve sonuç tam hesapla
    birebir aynıdır; son mum saklanmaz; farklı ağırlık ayrı kayıttır; verinin ilk günü değişirse tam hesaplanır."""
    import technical_analysis

    real = technical_analysis.build_v1_decisions
    calls = []

    def spy(df, **kwargs):
        calls.append(kwargs.get("only_last"))
        return real(df, **kwargs)

    monkeypatch.setattr(technical_analysis, "build_v1_decisions", spy)
    part, full = processed_df.iloc[:-3], processed_df
    first = history.cached_v1_decisions("BTC-USD", part)
    assert calls == [None] and first == real(part, include_ml=False)
    second = history.cached_v1_decisions("BTC-USD", full)
    assert calls[-1] == 4                                     # 3 yeni gün + saklanmayan son mum
    assert second == real(full, include_ml=False)
    from config import DecisionEngineConfig as cfg
    other = {"trend": cfg.TREND_WEIGHT * 2, "momentum": cfg.MOMENTUM_WEIGHT, "volume": cfg.VOLUME_WEIGHT,
             "pattern": cfg.PATTERN_WEIGHT, "ml": cfg.ML_WEIGHT, "advanced": cfg.ADVANCED_WEIGHT}
    history.cached_v1_decisions("BTC-USD", full, weights=other)
    assert calls[-1] is None                                  # farklı ağırlık: ayrı kayıt, tam hesap
    history.cached_v1_decisions("BTC-USD", full.iloc[1:])
    assert calls[-1] is None                                  # ilk gün değişti: tam hesap


def test_ac04_cache_failure_falls_back_to_plain_computation(monkeypatch, processed_df):
    """AC04 — Depo okunamaz/yazılamazsa kararlar önbelleksiz hesaplanır, hata yükselmez."""
    import storage
    from technical_analysis import build_v1_decisions

    def broken(*a, **k):
        raise storage.StorageAccessError("yok")

    monkeypatch.setattr(storage, "read_doc", broken)
    monkeypatch.setattr(storage, "write_doc", broken)
    assert history.cached_v1_decisions("ETH-USD", processed_df) == build_v1_decisions(processed_df, include_ml=False)


def _fresh(frame):
    from datetime import UTC, datetime, timedelta

    frame = frame.copy()
    frame["Open"] = frame["Open"].clip(lower=frame["Low"], upper=frame["High"])
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    frame.index = pd.date_range(end=pd.Timestamp(yesterday), periods=len(frame), freq="D", tz="UTC")
    return frame


@pytest.mark.parametrize("period, expected_rows", [(history.LONG, 700), (history.RECENT, 500)])
def test_ac05_screen_backtest_uses_the_selected_period(store, monkeypatch, processed_df, period, expected_rows):
    """AC05 — Günlük backtest'te "Test dönemi" seçilir (varsayılan 2018'den); uzun dönem uzun veriyi, "son ~2 yıl"
    uygulamanın verisini kullanır ve kullanılan veri aralığı ekranda yazar."""
    from app_helpers import click, make_app
    from conftest import make_ohlcv
    from data_fetchers import process_data

    raw = make_ohlcv(seed=11, segments=[(700, 0.002)])
    long_frame, _ = process_data(raw, "uzun")
    long_frame = _fresh(long_frame)
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, _fresh(processed_df))
    monkeypatch.setattr(history, "_cached_long", lambda symbol: long_frame)
    at = at.run()
    at = next(sb for sb in at.selectbox if sb.label == "Periyot:").set_value("1d").run()
    box = next(sb for sb in at.selectbox if sb.key == "bt_period")
    assert box.value == history.LONG
    at = box.set_value(period).run()
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("0.0001")
    at = click(at.run(), "🚀 Backtest Başlat")
    assert not at.exception
    shown = [c.value for c in at.caption if c.value.startswith("Test verisi:")]
    assert shown and f"({expected_rows} gün;" in shown[0]


def test_ac06_warm_fills_backtest_and_portfolio_caches_and_skips_held_assets(monkeypatch, processed_df):
    """AC06 — Önceden doldurma: zamanlamalı her varlık için tek varlık (ayarlı ağırlık) ve portföy (varsayılan ağırlık;
    TL varlıkta dolara çevrilmiş) karar kümeleri hesaplanır; al-tut yöntemli hisse ve uzun verisi olmayan atlanır."""
    import portfolio_ui

    calls = []
    monkeypatch.setattr(history, "_cached_long", lambda symbol: None if symbol == "SOL-USD" else processed_df)
    monkeypatch.setattr(history, "cached_v1_decisions",
                        lambda symbol, frame, weights=None, variant="", **k: calls.append((symbol, variant)) or {})
    monkeypatch.setattr(portfolio_ui, "to_pool_frame", lambda frame, fx: frame)
    done = history.warm(["BTC-USD", "AAPL", "GRAM_TRY", "SOL-USD"], log=lambda *_: None)
    assert done == {"BTC-USD": 2, "GRAM_TRY": 2}
    assert sorted(calls) == [("BTC-USD", ""), ("BTC-USD", ""), ("GRAM_TRY", ""), ("GRAM_TRY", "USD")]
