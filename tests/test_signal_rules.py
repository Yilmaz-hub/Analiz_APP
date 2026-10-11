"""Spec 0022 — varlığa göre sinyal kuralı (BTC/ETH/altın C, altcoin kanal, hisse al-tut)."""
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import signal_rules as sr
from trade_decisions import Signal


def _series(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", tz="UTC")
    closes = [float(c) for c in closes]
    return pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
                         "Close": closes}, index=idx)


def test_ac01_trend_rule_buys_only_above_the_200_day_average_and_keeps_v1_exit():
    """AC01 — BTC/ETH/altın: V1 AL ve kapanış > 200 günlük ortalama → AL; ortalama altında BEKLE; V1 SAT → SAT; 200 gün yoksa AL yok."""
    rising, falling = _series(range(100, 350)), _series(range(350, 100, -1))
    for symbol in ("BTC-USD", "ETH-USD", "XAU_GOLD"):
        assert sr.rule_signal(symbol, rising, Signal.BUY).signal is Signal.BUY
        assert sr.rule_signal(symbol, falling, Signal.BUY).signal is Signal.WAIT
        assert sr.rule_signal(symbol, falling, Signal.SELL).signal is Signal.SELL
    assert sr.rule_signal("BTC-USD", rising.iloc[-150:], Signal.BUY).signal is Signal.WAIT
    text = sr.rule_signal("BTC-USD", falling, Signal.BUY).text
    assert "V1 + 200 günlük ortalama → BEKLE" in text and "V1 kararı: AL" in text


def test_ac02_altcoin_rule_is_the_channel_breakout_independent_of_v1():
    """AC02 — Altcoin: kapanış önceki 50 günün en yükseğini aşınca AL, önceki 20 günün en düşüğünün altına inince SAT; V1 kararı etkilemez."""
    base = [100.0] * 80
    breakout = _series(base + [110.0])
    breakdown = _series(base + [90.0])
    assert sr.rule_signal("SOL-USD", breakout, Signal.SELL).signal is Signal.BUY
    assert sr.rule_signal("AVAX-USD", breakdown, Signal.BUY).signal is Signal.SELL
    assert sr.rule_signal("SOL-USD", _series(base), Signal.BUY).signal is Signal.WAIT
    assert "Kanal kırılımı 50/20 → AL" in sr.rule_signal("SOL-USD", breakout, Signal.SELL).text


def test_ac03_stocks_get_no_timing_signal_and_unknown_markets_keep_v1():
    """AC03 — Hisse: al-tut, zamanlama sinyali yok (BEKLE); kuralı olmayan varlık (döviz) V1 kararını aynen kullanır."""
    frame = _series(range(100, 350))
    held = sr.rule_signal("AAPL", frame, Signal.BUY)
    assert held.signal is Signal.WAIT and held.method == "TUT" and "Al-tut" in held.text
    other = sr.rule_signal("EURUSD=X", frame, Signal.BUY)
    assert other.signal is Signal.BUY and other.method is None and other.text is None


def test_ac04_signal_mode_defaults_to_by_asset_and_can_be_switched():
    """AC04 — Kural seçimi varsayılan "varlığa göre"dir; V1'e çevrilince kaydedilir; geçersiz değer reddedilir."""
    assert sr.signal_mode() == sr.VARLIK
    sr.choose_signal_mode(sr.V1)
    assert sr.signal_mode() == sr.V1
    with pytest.raises(ValueError):
        sr.choose_signal_mode("BAŞKA")


def _fresh(frame):
    frame = frame.copy()
    frame["Open"] = frame["Open"].clip(lower=frame["Low"], upper=frame["High"])
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    frame.index = pd.date_range(end=pd.Timestamp(yesterday), periods=len(frame), freq="D", tz="UTC")
    return frame


def _daily(at):
    return next(sb for sb in at.selectbox if sb.label == "Periyot:").set_value("1d").run()


def test_ac05_screen_uses_the_asset_rule_and_v1_mode_restores_the_old_signal(store, monkeypatch, processed_df):
    """AC05 — Ekranda günlük görünümde karar paneli ve yan panel varlığın kuralını yazar; "V1 (eski)" seçilince kural notu kalkar ve seçim kaydedilir."""
    from app_helpers import make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = _daily(make_app(monkeypatch, _fresh(processed_df)).run())
    assert not at.exception
    panel = [c.value for c in at.caption if c.value.startswith("Sinyal kuralı: V1 + 200 günlük ortalama")]
    assert panel and any("📐 Sinyal kuralı" in c.value for c in at.sidebar.caption)
    at = next(r for r in at.radio if r.key == "signal_mode_choice").set_value(sr.V1).run()
    assert not at.exception and sr.signal_mode() == sr.V1
    assert not any(c.value.startswith("Sinyal kuralı:") for c in at.caption)
