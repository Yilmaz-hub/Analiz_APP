"""Spec 0019 — trend filtreli aday (C) ve ağırlıklı portföy testi (P1)."""
from datetime import timedelta
from decimal import Decimal

import pandas as pd

import forward_runner
import forward_tracker as ft
import portfolio_backtest as pb
import strategy_candidates as sc
from conftest import make_ohlcv
from data_fetchers import process_data
from trade_execution import CostAssumptions

COSTS = CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0.1"))


def _flat(n=10, price=100.0, start="2026-01-01"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1, "Close": price, "ATR": 1.0}, index=idx)


# ---- AC01 / AC02: aday kural -------------------------------------------------------------------
def test_ac01_trend_rule_buys_only_above_the_average_and_keeps_v1_exits():
    """AC01 — TREND_SMA, AL'i yalnız V1 AL ve kapanış > SMA iken üretir; SAT V1'den geçer; yükselen koşulu aranmaz."""
    idx = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    frame = pd.DataFrame({"Close": [10, 10, 10, 20, 5, 30], "Low": 1.0}, index=idx)
    v1 = {idx[3]: "AL", idx[4]: "AL", idx[5]: "SAT"}
    cand = sc.Candidate("t", "TREND_SMA", "V1", "KRIPTO", {"sma": "3"})
    out = sc.candidate_decisions(frame, v1, cand, None)          # regimes kullanılmaz
    assert out[idx[3]] == "AL"            # 20 > ort(10,10,20)
    assert out[idx[4]] == "BEKLE"         # 5 < ort(10,20,5): AL engellenir
    assert out[idx[5]] == "SAT"           # çıkış V1
    assert out[idx[1]] == "BEKLE"         # ortalama oluşmadan giriş yok
    wide = sc.Candidate("t", "TREND_SMA", "V1", "KRIPTO", {"sma": "5"})
    assert sc.candidate_decisions(frame, v1, wide, None)[idx[3]] == "BEKLE"   # ayar kararı değiştirir


def test_ac02_trend_candidate_is_a_default_approved_candidate():
    """AC02 — Varsayılan adaylar TREND_SMA'yı (200 gün) içerir ve aday onaylıdır."""
    names = {c.entry_rule: c for c in sc.default_candidates("KRIPTO", None, None)}
    assert "TREND_SMA" in names and names["TREND_SMA"].settings == {"sma": "200"}
    assert sc.is_approved(names["TREND_SMA"])


# ---- AC03: paralel takip -----------------------------------------------------------------------
def test_ac03_runner_records_v1_and_the_trend_candidate_side_by_side():
    """AC03 — Koşucu aynı gün için hem V1 hem trend adayı kararını ayrı sürüm etiketleriyle kaydeder."""
    raw = make_ohlcv(seed=5, segments=[(400, 0.003)])
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")
    day = frame.index[-1].date()
    now = ft.available_at("KRIPTO", day)
    forward_runner.run(now, real_clock=True, fetch=lambda s: (frame, "fixture"), include_ml=False,
                       symbols=["BTC-USD"], clock=lambda: now + timedelta(minutes=1))
    versions = ft.versions("BTC-USD")
    assert len(versions) == 2
    assert any(v.startswith("V1~") for v in versions) and any(v.startswith("ADAY-") for v in versions)
    assert all(ft.get_decision("BTC-USD", v, day) is not None for v in versions)
    assert sc.active_strategy() == sc.REFERENCE_STRATEGY          # aktif strateji değişmez


# ---- AC04 / AC05: portföy motoru -----------------------------------------------------------------
def test_ac04_assets_share_one_cash_pool_by_weight():
    """AC04 — Bir varlık alımdayken diğerinin AL'i kalan nakitle girer; tutar toplam varlığın ağırlığı kadardır."""
    a, b = _flat(), _flat()
    dec = {"A": {a.index[0]: "AL"}, "B": {b.index[2]: "AL"}}
    steps = {"A": Decimal("0.0001"), "B": Decimal("0.0001")}
    result = pb.run_portfolio({"A": a, "B": b}, dec, {"A": Decimal("50"), "B": Decimal("30")},
                              capital=Decimal("10000"), costs=COSTS, quantity_steps=steps)
    end = result.equity[-1][1]
    assert Decimal("9980") < end <= Decimal("10000")                 # yalnız komisyon kaybı
    assert abs(result.holdings["A"] - Decimal("5000")) < 20           # A ≈ toplamın %50'si
    assert abs(result.holdings["B"] - Decimal("3000")) < 20           # B, A açıkken kalan nakitle ≈ %30
    assert abs(result.cash - Decimal("2000")) < 30                    # nakit ≈ %20


def test_ac04_entry_is_skipped_when_less_than_half_the_weight_is_left_in_cash():
    """AC04 — Nakit, ağırlığın yarısından azını karşılıyorsa giriş yapılmaz ve "nakit yetersiz" sayılır."""
    a, b = _flat(), _flat()
    dec = {"A": {a.index[0]: "AL"}, "B": {b.index[2]: "AL", b.index[3]: "AL"}}
    result = pb.run_portfolio({"A": a, "B": b}, dec, {"A": Decimal("80"), "B": Decimal("80")},
                              capital=Decimal("10000"), costs=COSTS,
                              quantity_steps={"A": Decimal("0.0001"), "B": Decimal("0.0001")})
    assert result.blocked.get("YETERSIZ_NAKIT", 0) >= 1
    assert result.contribution["B"] == 0 and not [t for t in result.trades if t.symbol == "B"]


def test_ac05_single_asset_portfolio_matches_the_single_asset_engine(processed_df):
    """AC05 — %100 ağırlıklı tek varlık portföyü, tek varlık motoruyla aynı giriş/çıkış günlerini üretir."""
    from technical_analysis import run_v1_strategy_backtest

    frame = processed_df
    idx = frame.index
    dec = {idx[220]: "AL", idx[260]: "SAT", idx[300]: "AL", idx[340]: "SAT", idx[400]: "AL"}
    single = run_v1_strategy_backtest(frame, dec, initial_cash=Decimal("10000"), trade_notional=Decimal("9000"),
                                      quantity_step=Decimal("0.0001"), costs=COSTS)
    multi = pb.run_portfolio({"X": frame}, {"X": dec}, {"X": Decimal("100")}, capital=Decimal("10000"),
                             costs=COSTS, quantity_steps={"X": Decimal("0.0001")})
    assert [(t["entry_at"], t["exit_at"], t["reason"]) for t in single["trades"]] == \
           [(t.entry_at, t.exit_at, t.reason) for t in multi.trades]
    assert single["trades"]


# ---- AC06 / AC07 -------------------------------------------------------------------------------
def test_ac06_default_weights():
    """AC06 — Varsayılan ağırlıklar: BTC/ETH %20, DOGE %5, diğer kriptolar %10."""
    assert pb.default_weight("BTC-USD") == 20 and pb.default_weight("ETHUSD") == 20
    assert pb.default_weight("DOGE-USD") == 5 and pb.default_weight("SOL-USD") == 10


def test_ac07_report_has_returns_drawdown_trades_hold_and_contribution():
    """AC07 — Rapor: yıllık/toplam getiri, düşüş, al-tut, varlık katkısı, işlem sayısı ve veri sınırı notu."""
    import portfolio_ui

    a = _flat(n=400)
    a["Close"] = a["Open"] = [100 + i * 0.1 for i in range(400)]
    a["High"], a["Low"] = a["Close"] + 1, a["Close"] - 1
    dec = {"A": {a.index[10]: "AL", a.index[300]: "SAT"}}
    result = pb.run_portfolio({"A": a}, dec, {"A": Decimal("20")}, capital=Decimal("10000"), costs=COSTS,
                              quantity_steps={})
    view = portfolio_ui.build_portfolio_view(result, {"A": Decimal("20")})
    assert [m[0] for m in view.metrics] == ["Yıllık getiri", "Toplam getiri", "En büyük düşüş"]
    assert view.rows[0]["İşlem"] == 1 and view.rows[0]["Kazanan"] == 1
    text = " ".join(view.notes)
    assert portfolio_ui.DATA_NOTE in text and "yaklaşıktır: A" in text     # adım yok → yaklaşık


def test_ac07_screen_runs_the_portfolio_test(store, monkeypatch, processed_df):
    """AC07 — Ekranda portföy testi çalışır; strateji ve al-tut tablosu ve veri notu görünür."""
    from app_helpers import click, make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    at = click(at, "📦 Portföy testini başlat")
    assert not at.exception
    assert any("Al-tut (aynı ağırlıklar)" in f.value.columns for f in at.dataframe)
    assert "son ~1000 gün" in texts(at)
