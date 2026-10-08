"""Spec 0015 — geçmiş test sonucunun yanında aynı dönemde al-tut karşılaştırması."""
from decimal import Decimal

import pandas as pd

import performance_ui


def _frame(first_open, last_close):
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame({"Open": [90.0, first_open, 110.0], "Close": [95.0, 105.0, last_close]}, index=idx)


def test_ac01_same_amount_hold_is_scaled_to_capital():
    """AC01 — Fiyat %50 artarsa, sermayenin %10'u ile al-tut sermayeye göre %5 eder; strateji aynı ölçekte kıyaslanır."""
    c = performance_ui.hold_comparison(_frame(100.0, 150.0), {"total_return": Decimal("-1.27")}, "10000", "1000")
    assert c.price_change_pct == Decimal("50")
    assert c.exposure_pct == Decimal("10")
    assert c.hold_same_pct == Decimal("5.0")
    assert "daha kötü" in c.verdict


def test_ac02_strategy_better_than_falling_market():
    """AC02 — Piyasa düşerken stratejinin az kaybı "daha iyi" diye yazılır."""
    c = performance_ui.hold_comparison(_frame(100.0, 60.0), {"total_return": Decimal("-1")}, 10000, 1000)
    assert c.hold_same_pct == Decimal("-4.0") and "daha iyi" in c.verdict


def test_ac03_invalid_inputs_give_none_not_an_error():
    """AC03 — Kısa veri, sıfır/NaN fiyat ya da geçersiz tutarda karşılaştırma hesaplanmaz (None), istisna yok."""
    ok = {"total_return": Decimal("1")}
    assert performance_ui.hold_comparison(_frame(100.0, 150.0).iloc[:1], ok, 10000, 1000) is None
    assert performance_ui.hold_comparison(_frame(0.0, 150.0), ok, 10000, 1000) is None
    assert performance_ui.hold_comparison(_frame(float("nan"), 150.0), ok, 10000, 1000) is None
    assert performance_ui.hold_comparison(_frame(100.0, 150.0), ok, 10000, 0) is None
    assert performance_ui.hold_comparison(_frame(100.0, 150.0), {}, 10000, 1000) is None


def test_ac04_screen_shows_the_comparison_under_backtest_results(store, monkeypatch, processed_df):
    """AC04 — Günlük geçmiş test sonrası ekranda strateji, aynı tutarla al-tut ve fiyat değişimi görünür."""
    import technical_analysis
    from app_helpers import click, make_app, texts

    monkeypatch.setattr(technical_analysis, "build_v1_decisions",
                        lambda df, **k: {df.index[199]: "AL", df.index[210]: "SAT"})
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("0.001")
    at = click(at.run(), "🚀 Backtest Başlat")
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Strateji", "Al-tut (aynı tutarla)", "Varlığın fiyat değişimi"} <= set(labels)
    assert "İşlem tutarı / sermaye oranı: %10." in texts(at)


def test_ac01_hold_starts_after_the_first_decision_not_during_warm_up():
    """AC01 — Isınma dönemi olan veride al-tut, ilk karardan sonraki mumun açılışında başlar."""
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"Open": [10.0, 50.0, 90.0, 100.0, 120.0], "Close": [10.0, 50.0, 95.0, 110.0, 150.0]},
                         index=idx)
    c = performance_ui.hold_comparison(frame, {"total_return": Decimal("1")}, 10000, 1000,
                                       decisions={idx[2]: "AL", idx[3]: "BEKLE"})
    assert c.start == idx[3].date() and c.price_change_pct == Decimal("50")     # 100 → 150
    assert performance_ui.hold_comparison(frame, {"total_return": Decimal("1")}, 10000, 1000,
                                          decisions={idx[4]: "AL"}) is None      # karar sonrası mum yok


def test_ac03_notional_above_capital_is_not_compared():
    """AC03 — İşlem tutarı sermayeden büyükse strateji alım yapamaz; karşılaştırma gösterilmez."""
    assert performance_ui.hold_comparison(_frame(100.0, 150.0), {"total_return": Decimal("0")}, 1000, 5000) is None
