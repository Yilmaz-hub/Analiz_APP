"""Spec 0016 — geçmiş test teşhisi: çıkış nedenleri, geri verilen kâr, dışarıda kalmanın etkisi."""
from decimal import Decimal

import pandas as pd

from trade_diagnostics import diagnose, explain


def _frame(closes, highs=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    highs = highs or closes
    return pd.DataFrame({"Open": closes, "High": highs, "Low": closes, "Close": closes}, index=idx)


def _trade(frame, i, j, entry, exit_, reason):
    return {"entry_at": frame.index[i], "exit_at": frame.index[j], "entry": str(entry),
            "exit": str(exit_), "reason": reason, "pnl": "0"}


def test_ac01_exit_reasons_are_counted():
    """AC01 — Çıkışlar nedene göre sayılır (STOP / SAT)."""
    f = _frame([100, 100, 100, 100, 100, 100])
    bt = {"trades": [_trade(f, 1, 2, 100, 95, "STOP"), _trade(f, 3, 4, 100, 104, "SAT")]}
    d = diagnose(f, bt)
    assert d.closed == 2 and d.exits == {"STOP": 1, "SAT": 1}


def test_ac02_profit_given_back_is_measured():
    """AC02 — İşlem %20 yükseğe çıkıp %-5 ile kapanırsa: en iyi %20, kapanış %-5, kâra geçip zararla kapanan 1."""
    f = _frame([100, 100, 110, 100, 95], highs=[100, 100, 120, 100, 95])
    d = diagnose(f, {"trades": [_trade(f, 1, 4, 100, 95, "STOP")]})
    assert d.avg_best_pct == Decimal("20") and d.avg_realized_pct == Decimal("-5")
    assert d.went_green_closed_red == 1
    assert any("%25.00 geri verildi" in line for line in explain(d))


def test_ac03_staying_out_is_measured():
    """AC03 — Pozisyon dışındaki günlerde varlık düştüyse "girmemek korudu" yazılır."""
    f = _frame([100, 110, 121, 60, 30])
    d = diagnose(f, {"trades": [_trade(f, 1, 2, 110, 121, "SAT")]})
    assert d.days_in == 2 and d.days_out == 2
    assert d.in_market_pct == Decimal("21")
    assert d.out_market_pct.quantize(Decimal("0.01")) == Decimal("-75.21")      # 121→60→30
    assert any("korudu" in line for line in explain(d))


def test_ac04_open_position_counts_as_in_market_and_bad_data_is_safe():
    """AC04 — Açık pozisyon günleri pozisyonda sayılır; geçersiz veri None, NaN fiyat istisna üretmez."""
    f = _frame([100, 100, 110, 120])
    d = diagnose(f, {"trades": [], "position": {"entry_at": f.index[2]}})
    assert d.days_in == 2 and d.closed == 0 and explain(d)[0].startswith("Kapanmış işlem yok")
    assert diagnose(f.iloc[:1], {"trades": []}) is None and diagnose(f, {}) is None
    g = _frame([100, float("nan"), 110, 120])
    assert diagnose(g, {"trades": [_trade(g, 1, 2, 100, 110, "SAT")]}) is not None


def test_ac05_screen_shows_the_diagnostics_after_a_daily_backtest(store, monkeypatch, processed_df):
    """AC05 — Günlük geçmiş test sonrası ekranda teşhis başlığı ve metrikleri görünür."""
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
    assert "Strateji teşhisi" in texts(at)
    assert {"Stop ile çıkış", "SAT ile çıkış", "Kâra geçip zararla kapanan"} <= {m.label for m in at.metric}
