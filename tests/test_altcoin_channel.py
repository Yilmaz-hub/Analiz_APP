"""Spec 0020 — altcoinlerde kanal kırılımı (Donchian 50/20)."""
from datetime import timedelta

import pandas as pd

import forward_runner
import forward_tracker as ft
import strategy_candidates as sc
from conftest import make_ohlcv
from data_fetchers import process_data


def _frame(closes, highs=None, lows=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", tz="UTC")
    highs = highs or [c + 1 for c in closes]
    lows = lows or [c - 1 for c in closes]
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes}, index=idx)


def test_ac01_channel_rule_buys_above_the_prior_high_and_sells_below_the_prior_low():
    """AC01 — Kapanış önceki N günün en yükseğini aşınca AL, önceki M günün en düşüğünün altına inince SAT; arada ve pencere dolmadan BEKLE; V1 kararı etkilemez."""
    closes = [10, 10, 10, 10, 12, 11, 10, 7, 10]
    frame = _frame(closes)
    idx = frame.index
    cand = sc.Candidate("t", "KANAL", "KANAL", "KRIPTO", {"entry": "3", "exit": "2"})
    v1 = {day: "SAT" for day in idx}                                   # V1 kararları yok sayılır
    out = sc.candidate_decisions(frame, v1, cand, None)
    assert [out[d] for d in idx[:3]] == ["BEKLE"] * 3                 # 3 günlük pencere dolmadı
    assert out[idx[3]] == "BEKLE"                                      # 10, önceki en yüksek 11'i aşmıyor
    assert out[idx[4]] == "AL"                                         # 12 > 11
    assert out[idx[5]] == "BEKLE"                                      # 11: ne 13'ü aşar ne 9'un altında
    assert out[idx[7]] == "SAT"                                        # 7 < önceki 2 günün en düşüğü (9)
    wide = sc.Candidate("t", "KANAL", "KANAL", "KRIPTO", {"entry": "5", "exit": "2"})
    assert sc.candidate_decisions(frame, v1, wide, None)[idx[4]] == "BEKLE"   # ayar kararı değiştirir
    channel = sc.channel_candidate("KRIPTO")
    assert channel.settings == {"entry": "50", "exit": "20"} and sc.is_approved(channel)


def test_ac02_method_is_channel_for_altcoins_and_trend_for_others():
    """AC02 — Altcoinlerde (SOL, XRP, DOGE, PEPE, tiresiz LINKUSD) kanal; BTC, ETH ve ons altında V1 + 200 günlük ortalama (hisseler spec 0021 ile al-tut)."""
    for symbol in ("SOL-USD", "XRP-USD", "DOGE-USD", "PEPE-USD", "LINKUSD"):
        assert sc.method_candidate(symbol).entry_rule == "KANAL", symbol
    for symbol in ("BTC-USD", "ETHUSD", "XAU_GOLD"):
        assert sc.method_candidate(symbol).entry_rule == "TREND_SMA", symbol
    assert sc.method_candidate("") is None


def test_ac03_runner_follows_channel_for_altcoins_and_trend_for_btc():
    """AC03 — Koşucu altcoinde V1 + kanal, BTC'de V1 + trend sürümlerini kaydeder; aktif strateji değişmez."""
    raw = make_ohlcv(seed=5, segments=[(400, 0.003)])
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")
    day = frame.index[-1].date()
    now = ft.available_at("KRIPTO", day)
    forward_runner.run(now, real_clock=True, fetch=lambda s: (frame, "fixture"), include_ml=False,
                       symbols=["SOL-USD", "BTC-USD"], clock=lambda: now + timedelta(minutes=1))
    channel = "ADAY-" + sc.strategy_fingerprint(sc.channel_candidate("KRIPTO"))[:12]
    trend = "ADAY-" + sc.strategy_fingerprint(sc.trend_candidate("KRIPTO"))[:12]
    sol, btc = ft.versions("SOL-USD"), ft.versions("BTC-USD")
    assert len(sol) == 2 and any(v.startswith(channel) for v in sol) and not any(v.startswith(trend) for v in sol)
    assert len(btc) == 2 and any(v.startswith(trend) for v in btc)
    assert all(ft.get_decision("SOL-USD", v, day) is not None for v in sol)
    assert sc.active_strategy() == sc.REFERENCE_STRATEGY


def _methods(at):
    table = next(f.value for f in at.dataframe if "Yöntem" in f.value.columns)
    return dict(zip(table["Varlık"], table["Yöntem"]))


def test_ac04_portfolio_screen_uses_channel_for_altcoins_and_can_turn_it_off(store, monkeypatch, processed_df):
    """AC04 — Portföy ekranında seçenek açıkken altcoin kanal, BTC trend yöntemiyle; kapalıyken hepsi trend; yöntem sütunu görünür."""
    from app_helpers import click, make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Solana (SOL)": "SOL-USD"})
    at = make_app(monkeypatch, processed_df).run()
    at = click(at, "📦 Portföy testini başlat")
    assert not at.exception
    assert _methods(at) == {"Bitcoin (BTC)": "V1 + 200 gün ort.", "Solana (SOL)": "Kanal 50/20"}
    at = next(c for c in at.checkbox if c.key == "pf_channel").uncheck().run()
    at = click(at, "📦 Portföy testini başlat")
    assert set(_methods(at).values()) == {"V1 + 200 gün ort."}
