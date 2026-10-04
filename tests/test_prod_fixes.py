"""Spec 0007 — yayın ortamı düzeltmeleri (kayıt bağlantısı ve fiyat bekleme süresi)."""
import time

import pytest

import data_fetchers
import storage

COIN = "Bitcoin (BTC)"


# ---- AC01: bağlantı adresi biçimi ------------------------------------------------------------
@pytest.mark.parametrize("given, expected", [
    ("postgres://u:p@h/db?sslmode=require", "postgresql+psycopg://u:p@h/db?sslmode=require"),
    ("postgresql://u:p@h/db?sslmode=require", "postgresql+psycopg://u:p@h/db?sslmode=require"),
    ('  "postgresql://u:p@h/db"  ', "postgresql+psycopg://u:p@h/db"),
    ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("postgresql+psycopg2://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
    ("sqlite:///x/analiz.db", "sqlite:///x/analiz.db"),
])
def test_ac01_connection_url_is_normalised_for_the_installed_driver(monkeypatch, given, expected):
    """AC01 — Neon'dan kopyalanan adres biçimleri sürücü eklenerek kullanılır; açık sürücü ve SQLite değişmez."""
    monkeypatch.setenv(storage.DB_URL_ENV, given)
    assert storage.db_url() == expected


# ---- AC02: bağlantı hatası kullanıcıya anlaşılır ve sızıntısız gösterilir -------------------------
def test_ac02_unreachable_remote_database_says_records_are_not_deleted(monkeypatch):
    """AC02 — Uzak veritabanına bağlanılamayınca mesaj kayıtların silinmediğini söyler; parola ve teknik metin sızmaz."""
    monkeypatch.setenv(storage.DB_URL_ENV, "postgresql://kullanici:GIZLIPAROLA@127.0.0.1:1/veritabani")
    storage.reset_engine()
    ok, message = storage.check_access()
    assert ok is False
    assert "silinmedi" in message and "bağlantı adresi" in message
    assert "GIZLIPAROLA" not in message and "psycopg" not in message and "127.0.0.1" not in message


# ---- AC03 / AC04: eşzamanlı, bütçeli fiyat alma --------------------------------------------------
def _slow_price(delay_by_coin):
    def fetch(coin, coin_map):
        time.sleep(delay_by_coin.get(coin, 0))
        return 100.0 + len(coin)
    return fetch


def test_ac03_prices_are_fetched_concurrently(monkeypatch):
    """AC03 — 6 pozisyonun her biri 1 sn süren fiyat kaynağıyla toplam bekleme yaklaşık 1 sn'dir."""
    coins = [f"Coin{i}" for i in range(6)]
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", _slow_price({c: 1.0 for c in coins}))
    started = time.monotonic()
    prices = data_fetchers.fetch_prices(coins, {c: f"{c}-USD" for c in coins}, budget=5.0)
    assert time.monotonic() - started < 2.5
    assert all(prices[c] > 0 for c in coins)


def test_ac04_slow_source_is_cut_at_the_budget_and_others_keep_their_price(monkeypatch):
    """AC04 — Bütçeden uzun süren kaynak sayfayı bütçeden fazla bekletmez; yavaş olan 0 (alınamadı), diğerleri fiyatlı döner."""
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        _slow_price({"Yavas": 5.0}))
    started = time.monotonic()
    prices = data_fetchers.fetch_prices(["Hizli", "Yavas"], {"Hizli": "A-USD", "Yavas": "B-USD"}, budget=0.6)
    assert time.monotonic() - started < 2.0
    assert prices["Yavas"] == 0 and prices["Hizli"] > 0


def test_ac04_duplicate_and_empty_inputs_are_safe(monkeypatch):
    """AC04 — Aynı coin birden çok kez istenirse tek çağrı yapılır; boş liste boş sonuç verir."""
    calls = []
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda coin, coin_map: calls.append(coin) or 10.0)
    assert data_fetchers.fetch_prices([], {}) == {}
    assert data_fetchers.fetch_prices(["A", "A", "A"], {"A": "A-USD"}) == {"A": 10.0}
    assert calls == ["A"]


# ---- AC05: ekran fiyat kaynağı yavaşken donmaz ----------------------------------------------------
def test_ac05_screen_does_not_wait_for_slow_price_sources(store, monkeypatch, processed_df):
    """AC05 — Fiyat kaynağı pozisyon başına 5 sn yavaşken ana ekran bütçe sınırı içinde tamamlanır ve portföy tablosu gelir."""
    from app_helpers import make_app

    names = {COIN: "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Chainlink": "LINK-USD", "Hedera": "HBAR-USD"}
    store.write_doc(store.ASSETS_KEY, names)
    position = lambda coin: {"Coin": coin, "Giriş": 100.0, "Adet": 2.0, "Yatırım": 200.0, "Realized": 0.0,
                             "Status": "ACTIVE", "Tarih": "2026-09-01", "Stop": 90.0,
                             "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00", "V1Verified": True}
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0,
                                          "positions": [position(c) for c in list(names)[1:]]})
    at = make_app(monkeypatch, processed_df)
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", _slow_price({c: 5.0 for c in names}))
    monkeypatch.setattr(data_fetchers, "PRICE_BUDGET_SECONDS", 1.0)
    started = time.monotonic()
    at.run()
    assert not at.exception
    assert time.monotonic() - started < 4.5          # 3 pozisyon × 5 sn = 15 sn beklenmez
    tables = [f.value for f in at.dataframe if "Fiyat" in f.value.columns]
    assert tables and set(tables[0]["Fiyat"]) == {"fiyat alınamadı (maliyetle gösterildi)"}


# ---- AC06: stop teması aynı fiyat kümesini kullanır ------------------------------------------------
def test_ac06_stop_alerts_use_the_given_prices_without_new_network_calls(monkeypatch):
    """AC06 — Stop teması uyarısı verilen fiyat kümesiyle üretilir; fiyat kaynağı ek kez çağrılmaz."""
    from portfolio import check_active_positions_auto_close

    def boom(*args, **kwargs):
        raise AssertionError("fiyat kaynağı çağrılmamalıydı")

    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", boom)
    portfolio = {"positions": [{"Coin": COIN, "Status": "ACTIVE", "Stop": 95.0, "Adet": 1, "Giriş": 100}]}
    _, alerts = check_active_positions_auto_close(portfolio, {COIN: "BTC-USD"}, prices={COIN: 90.0})
    assert [a["coin"] for a in alerts] == [COIN] and alerts[0]["observed_price"] == 90.0
