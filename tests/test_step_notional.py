"""Spec 0013 — adet/lot adımı ile işlem tutarı tutarsızlığı uyarısı."""
from datetime import timedelta
from decimal import Decimal

import forward_runner
import forward_tracker as ft
import trade_settings as ts
from conftest import make_ohlcv


def _parsed(step, notional="1000"):
    return ts.parse_settings({"capital": "10000", "notional": notional, "quantity_step": step,
                              "spread_bps": "0", "slippage_bps": "0", "commission_pct": "0"})


def test_ac01_step_exceeding_notional_is_detected():
    """AC01 — Adım × fiyat işlem tutarını aşıyorsa tutar döner; aşmıyorsa ya da bilinmiyorsa None."""
    assert ts.step_exceeds_notional(_parsed("5").settings, 4000) == Decimal("20000")
    assert ts.step_exceeds_notional(_parsed("0.001").settings, 4000) is None
    assert ts.step_exceeds_notional(_parsed("").settings, 4000) is None            # adım bilinmiyor
    assert ts.step_exceeds_notional(_parsed("5").settings, None) is None           # fiyat bilinmiyor
    assert ts.step_exceeds_notional(None, 4000) is None


def test_ac02_panel_warns_about_the_mismatch_and_explains_the_field(store, monkeypatch, processed_df):
    """AC02 — Adım 5 + tutar 1000 → uyarı görünür; adım 0.001 → uyarı yok; etiket "en küçük alınabilir miktar" der."""
    from app_helpers import make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    store.write_doc(store.TRADE_SETTINGS_KEY, {"Bitcoin (BTC)": ts.to_raw({
        "capital": "10000", "notional": "1000", "quantity_step": "500",
        "spread_bps": "0", "slippage_bps": "0", "commission_pct": "0"})})
    at = make_app(monkeypatch, processed_df).run()
    assert not at.exception
    step = next(t for t in at.text_input if t.label.startswith("Adet/lot adımı"))
    assert "en küçük alınabilir miktar" in step.label
    assert any("hiç alım yapılamaz" in w.value for w in at.warning)
    step.set_value("0.001").run()
    assert not any("hiç alım yapılamaz" in w.value for w in at.warning)


def test_ac03_runner_notes_the_mismatch(store):
    """AC03 — Koşucu, adım × fiyat işlem tutarını aşıyorsa raporuna not düşer."""
    frame = make_ohlcv()
    store.write_doc(store.TRADE_SETTINGS_KEY, {"BTC-USD": ts.to_raw({
        "capital": "10000", "notional": "1000", "quantity_step": "500",
        "spread_bps": "0", "slippage_bps": "0", "commission_pct": "0"})})
    now = ft.available_at("KRIPTO", frame.index[-1].date()) + timedelta(hours=1)
    report = forward_runner.run(now, real_clock=False, fetch=lambda s: (frame, "Fixture"),
                                include_ml=False, symbols=["BTC-USD"])
    assert any("adet adımı × fiyat işlem tutarını aşıyor" in n for n in report.notes)
