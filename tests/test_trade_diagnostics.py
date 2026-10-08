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
    """AC02 — İşlem %20 yükseğe çıkıp %-5 ile kapanırsa: en iyi %20, kapanış %-5 (medyan); R bilinmiyorsa 1R sayılmaz."""
    f = _frame([100, 100, 110, 100, 95], highs=[100, 100, 120, 100, 95])
    d = diagnose(f, {"trades": [_trade(f, 1, 4, 100, 95, "STOP")]})
    assert d.median_best_pct == Decimal("20") and d.median_realized_pct == Decimal("-5")
    assert d.r_known == 0 and d.reached_1r_closed_red == 0


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
    assert {"Stop ile çıkış", "SAT ile çıkış", "1R kâr görüp zararla kapanan"} <= {m.label for m in at.metric}


def test_ac03_warm_up_days_are_not_counted_as_staying_out():
    """AC03 — İlk karardan önceki ısınma günleri "dışarıda kalma" sayılmaz."""
    f = _frame([100, 50, 25, 30, 33])          # ısınmada sert düşüş, sonra yükseliş
    d = diagnose(f, {"trades": []}, decisions={f.index[2]: "BEKLE"})
    assert d.days_out == 2 and d.out_market_pct == Decimal("32")          # 25 → 33, ısınma hariç
    assert any("kaçırdı" in line for line in explain(d))
    assert diagnose(f, {"trades": []}, decisions={f.index[4]: "AL"}) is None


def test_ac02_exit_day_high_after_the_exit_is_ignored():
    """AC02 — SAT ertesi açılışta 95'ten çıkar; o günün sonradan oluşan yükseği (101) "kâra geçti" sayılmaz."""
    f = _frame([100, 100, 95], highs=[100, 99, 101])
    d = diagnose(f, {"trades": [_trade(f, 1, 2, 100, 95, "SAT")]})
    assert d.reached_1r_closed_red == 0
    assert d.median_best_pct == Decimal("-1") and d.median_realized_pct == Decimal("-5")


def test_ac04_missing_high_column_and_flat_market_are_safe():
    """AC04 — High kolonu yoksa None; dışarıdayken hiç değişim yoksa "değişmedi" yazılır."""
    f = _frame([100, 100, 100])
    assert diagnose(f.drop(columns=["High"]), {"trades": []}) is None
    assert any("değişmedi" in line for line in explain(diagnose(f, {"trades": []})))



# ---- Spec 0017: dürüst teşhis ----------------------------------------------------------------------
def _atr_frame(closes, highs, atr):
    f = _frame(closes, highs)
    f["ATR"] = atr
    return f


def test_0017_ac01_median_is_not_skewed_by_one_big_winner():
    """AC01 — Bir büyük kazanç (%+100) ve üç küçük kayıp: medyan en iyi seviye büyük kazancı yansıtmaz."""
    f = _frame([100] * 10, highs=[100, 101, 100, 101, 100, 101, 200, 100, 100, 100])
    trades = [_trade(f, 1, 2, 100, 95, "STOP"), _trade(f, 3, 4, 100, 95, "STOP"),
              _trade(f, 5, 6, 100, 95, "STOP"), _trade(f, 6, 7, 100, 200, "SAT")]
    d = diagnose(f, {"trades": trades})
    assert d.median_best_pct == Decimal("1")          # ortalama %25.75 olurdu


def test_0017_ac02_green_by_a_cent_is_not_reaching_1r():
    """AC02 — Girişin hemen üstüne çıkıp stopta kapanan işlem 1R kâr görmüş sayılmaz; 1R üstü görülen sayılır."""
    # ATR 2 → başlangıç stopu 100 − 2,5×2 = 95, R = 5
    f = _atr_frame([100, 100, 100, 100, 100, 100], [100, 100.5, 100, 100, 106, 100], 2.0)
    trades = [_trade(f, 1, 2, 100, 95, "STOP"), _trade(f, 4, 5, 100, 95, "STOP")]
    d = diagnose(f, {"trades": trades})
    assert d.r_known == 2
    assert d.reached_1r_closed_red == 1               # yalnız 106'yı (1,2R) gören


def test_0017_ac03_winners_and_losers_are_split_with_entry_regime():
    """AC03 — Kazanan ve kaybedenler ayrı yazılır; her işlemin giriş koşulu (karar günü verisiyle) görünür."""
    f = _frame([100, 100, 100, 100, 100, 100])
    trades = [_trade(f, 1, 2, 100, 95, "STOP"), _trade(f, 3, 4, 100, 120, "SAT")]
    seen = []

    def classify(part):
        seen.append(len(part))
        return ("YATAY" if len(part) == 1 else "YUKSELEN", "R1")

    d = diagnose(f, {"trades": trades}, classify=classify)
    assert seen == [1, 3]                              # yalnız girişe karar verilen güne kadarki veri
    text = " ".join(explain(d))
    assert "Kaybedenler: 1 işlem" in text and "1 Yatay" in text
    assert "Kazananlar: 1 işlem" in text and "1 Yükselen" in text
    assert "desen için az" in text


def test_0017_ac04_trade_table_has_one_row_per_closed_trade():
    """AC04 — İşlem tablosu her kapanmış işlem için bir satır verir; R bilinmiyorsa "—"."""
    from trade_diagnostics import trade_table

    f = _frame([100, 100, 100, 100])
    d = diagnose(f, {"trades": [_trade(f, 1, 2, 100, 95, "STOP")]})
    (row,) = trade_table(d)
    assert row["Sonuç"] == "Kaybetti" and row["En iyi (R)"] == "—" and row["Giriş koşulu"] == "belirsiz"
