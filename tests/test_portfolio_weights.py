"""Spec 0024 — Portföy düzenleme ve ağırlık sınırı testleri.

AC01: Form yapısı düzenleme sırasında ara rerun yapmaz (st.form).
AC02: %100 aşımında ve sıfır toplamda hata uyarısı verilir ve test başlatılamaz.
AC03: Geçerli ağırlıklarda (<= %100) test çalışır, boşta nakit gösterilir.
AC04: Motor seviyesinde run_portfolio toplam ağırlık > 100 veya <= 0 olduğunda ValueError fırlatır.
AC05: Varsayılan portföy ağırlıkları toplamı %100'dür (BTC 20, ETH 20, XAU 15, SOL 10, AVAX 10, XRP 10, LINK 10, DOGE 5).
"""
from decimal import Decimal
import pytest
import pandas as pd

import portfolio_backtest as pb
from trade_execution import CostAssumptions


def _flat(n=20, price=100.0, start="2026-01-01"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1, "Close": price, "ATR": 1.0}, index=idx)


# ---- AC04: Motor seviyesinde doğrulama ----------------------------------------------------------
def test_ac04_run_portfolio_rejects_weights_exceeding_100_percent():
    """AC04 — portfolio_backtest.run_portfolio toplamı %100'ü aşan ağırlıklarda ValueError fırlatır."""
    a, b = _flat(), _flat()
    dec = {"A": {}, "B": {}}
    with pytest.raises(ValueError, match="Portföy ağırlıkları toplamı %100'ü aşamaz"):
        pb.run_portfolio({"A": a, "B": b}, dec, {"A": Decimal("60"), "B": Decimal("50")},
                          capital=Decimal("10000"),
                          costs=CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0.1")))


def test_ac04_run_portfolio_rejects_zero_or_negative_weights():
    """AC04 — portfolio_backtest.run_portfolio sıfır toplamda ve negatif tekil ağırlıkta ValueError fırlatır."""
    a = _flat()
    with pytest.raises(ValueError, match="Portföy ağırlıkları toplamı sıfırdan büyük olmalıdır"):
        pb.run_portfolio({"A": a}, {"A": {}}, {"A": Decimal("0")},
                          capital=Decimal("10000"),
                          costs=CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0.1")))

    with pytest.raises(ValueError, match="0 ile 100 arasında olmalıdır"):
        pb.run_portfolio({"A": a}, {"A": {}}, {"A": Decimal("-10")},
                          capital=Decimal("10000"),
                          costs=CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0.1")))


# ---- AC05: Varsayılan portföy geçerliliği ve XAU %15 ------------------------------------------
def test_ac05_default_weights_sum_to_exact_100_with_link_and_gold_15():
    """AC05 — Varsayılan varlıklar (BTC %20, ETH %20, XAU %15, SOL/AVAX/XRP/LINK %10, DOGE %5) toplamı %100'dür."""
    from config import DEFAULT_COIN_MAP
    import market_map

    # DEFAULT_COIN_MAP içinde LINK yer alır
    assert "Chainlink (LINK)" in DEFAULT_COIN_MAP
    assert DEFAULT_COIN_MAP["Chainlink (LINK)"] == "LINK-USD"

    defaults = [symbol for symbol in DEFAULT_COIN_MAP.values()
                if (market_map.market_of(symbol) or ("", ""))[0] == "KRIPTO"
                or market_map.market_of(symbol) == ("ALTIN", "USD")]
    
    # 8 varsayılan varlık
    assert len(defaults) == 8
    total_default = sum((pb.default_weight(s) for s in defaults), Decimal("0"))
    assert total_default == Decimal("100")

    # Tekil ağırlıklar
    assert pb.default_weight("BTC-USD") == Decimal("20")
    assert pb.default_weight("ETH-USD") == Decimal("20")
    assert pb.default_weight("XAU_GOLD") == Decimal("15")
    assert pb.default_weight("SOL-USD") == Decimal("10")
    assert pb.default_weight("AVAX-USD") == Decimal("10")
    assert pb.default_weight("XRP-USD") == Decimal("10")
    assert pb.default_weight("LINK-USD") == Decimal("10")
    assert pb.default_weight("DOGE-USD") == Decimal("5")


# ---- AC01, AC02, AC03: Arayüz ve Form Testleri (AppTest) ---------------------------------------
def test_ac02_screen_blocks_execution_when_weights_exceed_100_or_are_zero(store, monkeypatch, processed_df):
    """AC02 — Ağırlıklar toplamı %100'ü aştığında veya 0 olduğunda ekranda hata verilir ve test başlatılamaz."""
    from app_helpers import make_app, click

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"})
    at = make_app(monkeypatch, processed_df).run()

    # Form içindeki ağırlıkları 60 + 50 = 110 yap
    w_btc = next(inp for inp in at.number_input if inp.key == "pf_w:Bitcoin (BTC)")
    w_eth = next(inp for inp in at.number_input if inp.key == "pf_w:Ethereum (ETH)")
    w_btc.set_value(60.0)
    w_eth.set_value(50.0)

    # Butona basarak formu gönder
    at = click(at, "📦 Portföy testini başlat")

    # Hata mesajı görünmeli, sonuç tablosu görünmemeli
    assert any("Ağırlıklar toplamı %100'ü geçemez" in err.value for err in at.error)
    assert not any("Al-tut (aynı ağırlıklar)" in getattr(f, "value", pd.DataFrame()).columns for f in at.dataframe)

    # Sıfır ağırlık durumu
    w_btc = next(inp for inp in at.number_input if inp.key == "pf_w:Bitcoin (BTC)")
    w_eth = next(inp for inp in at.number_input if inp.key == "pf_w:Ethereum (ETH)")
    w_btc.set_value(0.0)
    w_eth.set_value(0.0)
    at = click(at, "📦 Portföy testini başlat")
    assert any("sıfırdan büyük olmalıdır" in w.value for w in at.warning)
    assert not any("Al-tut (aynı ağırlıklar)" in getattr(f, "value", pd.DataFrame()).columns for f in at.dataframe)


def test_ac03_screen_runs_with_valid_weights_and_shows_cash_reserve(store, monkeypatch, processed_df):
    """AC03 — Ağırlıklar toplamı <= %100 olduğunda test başarıyla çalışır ve boşta nakit gösterilir."""
    from app_helpers import make_app, click

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"})
    at = make_app(monkeypatch, processed_df).run()

    # 40 + 40 = 80 (%20 boşta nakit)
    w_btc = next(inp for inp in at.number_input if inp.key == "pf_w:Bitcoin (BTC)")
    w_eth = next(inp for inp in at.number_input if inp.key == "pf_w:Ethereum (ETH)")
    w_btc.set_value(40.0)
    w_eth.set_value(40.0)

    at = click(at, "📦 Portföy testini başlat")

    assert not at.exception
    assert not at.error
    assert any("Al-tut (aynı ağırlıklar)" in f.value.columns for f in at.dataframe)
    info_text = " ".join(i.value for i in at.info)
    assert "Portföy ağırlık toplamı: %80" in info_text
    assert "Boşta nakit: %20" in info_text


def test_ac01_form_batches_inputs_without_individual_rerun(store, monkeypatch, processed_df):
    """AC01 — Form içindeki number_input değerleri submit butonuna basılmadan sonuç tablosunu çalıştırmaz."""
    from app_helpers import make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()

    w_btc = next(inp for inp in at.number_input if inp.key == "pf_w:Bitcoin (BTC)")
    w_btc.set_value(50.0)
    # Form henüz submit edilmediği için sonuç dataframe'i oluşmamış olmalı
    assert not any("Al-tut (aynı ağırlıklar)" in getattr(f, "value", pd.DataFrame()).columns for f in at.dataframe)
