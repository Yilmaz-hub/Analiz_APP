"""Spec 0021 — stopsuz al-tut yöntemi, TL varlıkların dolara çevrilmesi ve 0020 QA düşük bulguları."""
from datetime import date, datetime, timedelta, timezone
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


def test_ac01_hold_asset_is_bought_once_by_weight_and_never_stopped_out():
    """AC01 — Tut yöntemli varlık ilk gün ağırlığı kadar alınır; fiyat stop seviyesinin çok altına düşse de satılmaz; Net K/Z dönem sonu değeridir; nakit diğer varlıkla paylaşılır."""
    a, b = _flat(), _flat()
    a.iloc[5:, [0, 1, 2, 3]] = [[50.0, 51.0, 49.0, 50.0]] * 5           # %50 düşüş: 2,5 ATR stopun çok altı
    dec = {"B": {b.index[2]: "AL"}}
    result = pb.run_portfolio({"A": a, "B": b}, dec, {"A": Decimal("40"), "B": Decimal("30")},
                              capital=Decimal("10000"), costs=COSTS,
                              quantity_steps={"A": Decimal("0.0001"), "B": Decimal("0.0001")}, hold={"A"})
    assert not [t for t in result.trades if t.symbol == "A"]               # hiç satış yok
    assert abs(result.holdings["A"] - Decimal("2000")) < 5                 # 4000'lik alım yarıya indi, tutuluyor
    assert abs(result.contribution["A"] + Decimal("2004")) < 5             # gerçekleşmemiş zarar + komisyon
    assert abs(result.holdings["B"] - Decimal("3000")) < 30                # B kalan nakitle ≈ %30
    assert abs(result.cash - Decimal("3000")) < 30


def test_ac02_method_by_asset_class():
    """AC02 — Hisse (ABD, BIST) al-tut; altın varsayılan V1 + 200 gün ort., seçenekle al-tut; altcoin kanal; BTC/ETH, türevleri ve sabit coinler trend; hissede aday takibi yok."""
    assert sc.asset_method("AAPL") == sc.asset_method("THYAO.IS") == sc.TUT
    assert sc.asset_method("XAU_GOLD") == sc.asset_method("GRAM_TRY") == "TREND_SMA"
    assert sc.asset_method("XAU_GOLD", gold_hold=True) == sc.asset_method("GRAM_TRY", gold_hold=True) == sc.TUT
    assert sc.asset_method("SOL-USD", gold_hold=True) == "KANAL"
    for symbol in ("BTC-USD", "ETH-USDT", "WBTC-USD", "STETH-USD", "USDC-USD", "USDT-USD"):
        assert sc.asset_method(symbol) == "TREND_SMA", symbol
    assert sc.method_candidate("AAPL") is None and sc.method_candidate("THYAO.IS") is None
    assert sc.asset_method("EURUSD=X") is None
    assert pb.default_weight("THYAO.IS") == 5 and pb.default_weight("AAPL") == 10


def test_ac03_try_prices_are_converted_to_dollars_by_calendar_day():
    """AC03 — TL fiyatlar o günün (yoksa son bilinen) USD/TRY kuruna bölünür; saat dilimi farkı eşleşmeyi bozmaz; kurdan önceki günler atılır."""
    idx = pd.date_range("2026-03-02 07:00", periods=4, freq="D", tz="UTC")       # BIST mumları sabah damgalı
    frame = pd.DataFrame({"Open": 400.0, "High": 440.0, "Low": 360.0, "Close": 400.0, "ATR": 40.0}, index=idx)
    fx_idx = pd.to_datetime(["2026-03-03", "2026-03-05"]).tz_localize("UTC")     # 2 Mart kur yok, 4 Mart eksik
    fx = pd.DataFrame({"Close": [40.0, 50.0]}, index=fx_idx)
    out = pb.to_usd(frame, fx)
    assert [ts.day for ts in out.index] == [3, 4, 5]
    assert list(out["Close"]) == [10.0, 10.0, 8.0]                               # 4 Mart: son bilinen kur (40)
    assert list(out["High"]) == [11.0, 11.0, 8.8] and list(out["ATR"]) == [1.0, 1.0, 0.8]


def test_ac04_screen_runs_crypto_with_stocks_and_gold_hold(store, monkeypatch, processed_df):
    """AC04 — Ekranda BTC + BIST hissesi + altın birlikte test edilir; hisse al-tut, altın seçenekle al-tut olur; yöntem sütunu bunu gösterir."""
    from app_helpers import click, make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Türk Hava Yolları (THYAO)": "THYAO.IS",
                                       "ONS ALTIN ($)": "XAU_GOLD"})
    at = make_app(monkeypatch, processed_df).run()
    at = next(m for m in at.multiselect if m.key == "pf_assets").select("Türk Hava Yolları (THYAO)").run()
    at = click(at, "📦 Portföy testini başlat")
    assert not at.exception

    def methods():
        table = next(f.value for f in at.dataframe if "Yöntem" in f.value.columns)
        return dict(zip(table["Varlık"], table["Yöntem"]))

    assert methods() == {"Bitcoin (BTC)": "V1 + 200 gün ort.", "ONS ALTIN ($)": "V1 + 200 gün ort.",
                         "Türk Hava Yolları (THYAO)": "Al-tut (stopsuz)"}
    at = next(c for c in at.checkbox if c.key == "pf_gold_hold").check().run()
    at = click(at, "📦 Portföy testini başlat")
    assert methods()["ONS ALTIN ($)"] == "Al-tut (stopsuz)"


def test_ac05_runner_does_not_follow_a_candidate_for_held_stocks():
    """AC05 — Koşucu hissede yalnız aktif stratejiyi (V1) kaydeder; al-tut yöntemli varlık için aday sürümü açılmaz."""
    raw = make_ohlcv(seed=5, segments=[(400, 0.003)])
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")
    day = frame.index[-1].date()
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    frame = frame[[ts.date() <= day for ts in frame.index]]
    now = ft.available_at("ABD", day)
    forward_runner.run(now, real_clock=True, fetch=lambda s: (frame, "fixture"), include_ml=False,
                       symbols=["AAPL"], clock=lambda: now + timedelta(minutes=1))
    versions = ft.versions("AAPL")
    assert len(versions) == 1 and versions[0].startswith("V1~")


def test_ac06_forward_table_marks_a_version_that_stopped_updating(monkeypatch):
    """AC06 — Aynı varlığın başka sürümü ilerlerken güncellenmeyen sürüm "Takip durdu" diye işaretlenir; güncel sürüm değil."""
    import forward_ui

    def decided(last):
        return [ft.ForwardDecision(asset="SOL-USD", strategy_version="v", candle_day=last, decision="BEKLE",
                                   source="t", evaluated_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
                                   candle={"close": "1"})]

    class _Verdict:
        sufficient, missing = False, ()

    days = {"V1": date(2026, 10, 10), "ADAY-trend": date(2026, 10, 1), "ADAY-kanal": date(2026, 10, 10)}
    monkeypatch.setattr(ft, "decisions", lambda asset, version: decided(days[version]))
    monkeypatch.setattr(ft, "assess", lambda asset, version: _Verdict())
    monkeypatch.setattr(ft, "closed_trades", lambda asset, version: 0)
    rows = forward_ui.build_summary_rows([("SOL-USD", v) for v in days])
    status = {r["_version"]: r["Durum"] for r in rows}
    assert status == {"V1": "⏳ Birikiyor", "ADAY-trend": "⏸️ Takip durdu", "ADAY-kanal": "⏳ Birikiyor"}


def test_ac07_rule_note_matches_the_methods_in_use():
    """AC07 — Kural notları yalnız kullanılan yöntemler için yazılır (0020 QA düşük bulgu)."""
    import portfolio_ui

    a = _flat(n=30)
    result = pb.run_portfolio({"A": a}, {"A": {}}, {"A": Decimal("50")}, capital=Decimal("10000"), costs=COSTS,
                              quantity_steps={"A": Decimal("0.0001")}, hold={"A"})
    text = " ".join(portfolio_ui.build_portfolio_view(result, {"A": Decimal("50")}, methods={"A": "KANAL"}).notes)
    assert "Kanal 50/20" in text and "200 günlük ortalamanın" not in text
    text = " ".join(portfolio_ui.build_portfolio_view(result, {"A": Decimal("50")}, methods={"A": "TUT"}).notes)
    assert "Al-tut (stopsuz)" in text and "Kanal 50/20" not in text and "200 günlük ortalamanın" not in text
