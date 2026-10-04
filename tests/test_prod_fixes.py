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
    assert time.monotonic() - started < 8.0          # 3 pozisyon × 5 sn = 15 sn beklenmez
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


# ---- AC07–AC09: fiyat kaynakları ------------------------------------------------------------------
class _Reply:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


def _fake_net(monkeypatch, routes, seen=None):
    """`routes`: (url parçası) -> Reply ya da istisna; ilk eşleşen kazanır, eşleşme yoksa istisna."""
    def get(url, params=None, headers=None, timeout=None, **kwargs):
        if seen is not None:
            seen.append({"url": url, "params": params, "headers": headers})
        for part, reply in routes.items():
            if part in url:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise ConnectionError("yanıt yok")
    monkeypatch.setattr(data_fetchers.requests, "get", get)
    monkeypatch.setattr(data_fetchers, "_yahoo_price", lambda symbol: None)


def test_ac07_price_requests_carry_the_browser_user_agent(monkeypatch):
    """AC07 — Binance fiyat isteği grafik isteğiyle aynı tarayıcı kimliğini gönderir."""
    seen = []
    _fake_net(monkeypatch, {"ticker/price": _Reply(200, {"price": "18.4"})}, seen)
    assert data_fetchers._live_price("LINKUSD") == 18.4
    assert seen and all("Mozilla" in (call["headers"] or {}).get("User-Agent", "") for call in seen)


def test_ac08_price_falls_back_to_the_last_candle_when_the_ticker_endpoint_fails(monkeypatch):
    """AC08 — Fiyat uç noktası hata verip kline uç noktası yanıt verirse fiyat son mumun kapanışından alınır."""
    candle = [[1700000000000, "17", "19", "16", "18.5", "10", 1700000059999, "1", 1, "1", "1", "0"]]
    _fake_net(monkeypatch, {"ticker/price": _Reply(403, {}), "klines": _Reply(200, candle)})
    for symbol in ("LINKUSD", "HBARUSD"):
        assert data_fetchers._live_price(symbol) == 18.5


def test_ac09_price_falls_back_to_okx_when_no_binance_host_answers(monkeypatch):
    """AC09 — Binance adreslerinin hiçbiri yanıt vermezse fiyat OKX'ten, `LINK-USDT` biçimiyle alınır."""
    seen = []
    _fake_net(monkeypatch, {"okx.com": _Reply(200, {"code": "0", "data": [{"last": "18.7"}]})}, seen)
    assert data_fetchers._live_price("LINKUSD") == 18.7
    okx = [call for call in seen if "okx.com" in call["url"]]
    assert okx and okx[0]["params"]["instId"] == "LINK-USDT"


def test_ac09_all_sources_down_gives_zero_not_an_exception(monkeypatch):
    """AC09 — Hiçbir kaynak yanıt vermezse fiyat 0 (alınamadı) döner, istisna çıkmaz."""
    _fake_net(monkeypatch, {})
    assert data_fetchers._live_price("LINKUSD") == 0


# ---- AC10 / AC11: satış paneli yerel çalışır --------------------------------------------------------
def _sell_app(store, monkeypatch, processed_df):
    from app_helpers import make_app

    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": COIN, "Giriş": 100.0, "Adet": 10.0, "Yatırım": 1000.0, "Realized": 0.0, "Status": "ACTIVE",
        "Tarih": "2026-09-01", "Stop": 90.0, "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00",
        "V1Verified": True}]})
    at = make_app(monkeypatch, processed_df)
    calls = {"market": 0}
    real = data_fetchers.get_market_data

    def counting(*args, **kwargs):
        calls["market"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(data_fetchers, "get_market_data", counting)
    return at, calls


def test_ac10_sell_panel_is_a_streamlit_fragment():
    """AC10 — "Kâr Al / Satış Yap" paneli parça (fragment) olarak tanımlıdır; yazılan değer yalnız onu yeniden çalıştırır."""
    import ast
    import os

    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"),
                  encoding="utf-8").read()
    panel = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "sell_panel")
    assert any(ast.unparse(decorator) == "st.fragment" for decorator in panel.decorator_list)
    assert "sell_panel(sel_c, curr, records_writable)" in source   # panel çağrılıyor


def test_ac11_confirming_a_sale_refreshes_the_whole_page(store, monkeypatch, processed_df):
    """AC11 — Satış onaylanınca sayfa tümüyle yenilenir: portföy tablosu ve nakit güncellenir."""
    from app_helpers import click

    at, _ = _sell_app(store, monkeypatch, processed_df)
    at.run()
    at.number_input(key=f"sell_pct:{COIN}").set_value(40.0).run()
    at = click(at, "Satışı Onayla")
    assert not at.exception
    saved = store.read_doc(store.PORTFOLIO_KEY)
    assert saved["positions"][0]["Adet"] == 6.0 and saved["balance"] > 1000.0
    tables = [f.value for f in at.dataframe if "Adet" in f.value.columns]
    assert tables and tables[0]["Adet"].tolist() == [6.0]


# ---- AC12: kaynak ayrıntısı ------------------------------------------------------------------------------
def test_ac12_price_diagnostics_are_recorded_without_technical_text(monkeypatch):
    """AC12 — Fiyatı alınamayan varlık için denenen kaynaklar ve sonuçları kaydedilir; adres ve teknik metin yok."""
    _fake_net(monkeypatch, {"ticker/price": _Reply(403, {})})
    data_fetchers.PRICE_DIAGNOSTICS.clear()
    assert data_fetchers._live_price("LINKUSD") == 0
    lines = data_fetchers.price_diagnostics("LINKUSD")
    assert lines and any("Binance" in line and "reddedildi" in line for line in lines)
    joined = " ".join(lines)
    assert "http" not in joined and "Traceback" not in joined and "Error" not in joined


# ---- QA turu 2 bulguları (F1–F6) ---------------------------------------------------------------------------
def _held_portfolio():
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal

    import trade_confirmation as tc
    from position_journal import PositionJournal

    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    portfolio, journal = {"balance": 5000.0, "positions": []}, PositionJournal()
    at = now - timedelta(hours=3)
    assert tc.confirm_buy(portfolio, journal, lambda: True, coin=COIN, symbol="BTC-USD", quantity=Decimal("10"),
                          price=Decimal("100"), stop=Decimal("90"), executed_at=at, now=now).ok
    return tc, portfolio, journal, now, at


def _sell_args(position, now, at):
    from decimal import Decimal
    return dict(position=position, symbol="BTC-USD", quantity=Decimal("4"), price=Decimal("110"),
                executed_at=at + __import__("datetime").timedelta(hours=1), now=now)


def test_f1_stale_position_object_cannot_be_sold_against_a_rolled_back_portfolio():
    """AC11 — Yazma kopup portföy geri alındıktan sonra eski pozisyon nesnesiyle satış reddedilir; nakit ve pozisyon çelişmez."""
    import copy

    tc, portfolio, journal, now, at = _held_portfolio()
    snapshot = copy.deepcopy(portfolio)
    stale = portfolio["positions"][0]
    assert tc.confirm_sell(portfolio, journal, lambda: False, **_sell_args(stale, now, at)).code == "KAYIT_YAZILAMADI"
    rolled_back = copy.deepcopy(snapshot)                 # uygulamanın safe_save_portfolio geri alması
    again = tc.confirm_sell(rolled_back, journal, lambda: True, **_sell_args(stale, now, at))
    assert again.ok is False and again.code == "POZISYON_GUNCEL_DEGIL"
    assert rolled_back == snapshot                          # nakit de pozisyon da değişmedi


def test_f1_fresh_position_from_the_current_portfolio_sells_correctly_after_a_failed_attempt():
    """AC11 — Başarısız denemeden sonra güncel portföyden çözülen pozisyonla satış tam bir kez işlenir (nakit 480 ... pozisyon 6)."""
    import copy

    import positions

    tc, portfolio, journal, now, at = _held_portfolio()
    cash_before = portfolio["balance"]
    snapshot = copy.deepcopy(portfolio)
    tc.confirm_sell(portfolio, journal, lambda: False, **_sell_args(portfolio["positions"][0], now, at))
    current = copy.deepcopy(snapshot)
    fresh = positions.active_position(current, COIN)
    assert tc.confirm_sell(current, journal, lambda: True, **_sell_args(fresh, now, at)).ok
    assert current["positions"][0]["Adet"] == 6.0 and current["balance"] == cash_before + 440.0


def test_f1_sell_panel_resolves_the_position_from_session_state():
    """AC11 — Satış paneli pozisyonu fragment argümanından değil güncel portföyden çözer."""
    import os

    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"),
                  encoding="utf-8").read()
    body = source.split("def sell_panel(")[1].split("_RUN_PRICES")[0]
    assert body.startswith("sel_c, curr, records_writable)")
    assert "group_positions(st.session_state['portfolio_data']['positions'])" in body


def test_f3_more_than_eight_positions_all_get_their_price(monkeypatch):
    """AC04 — 12 pozisyonda ilk 8'i yavaş olsa bile hızlı olanlar sırada beklemez ve fiyatını alır."""
    coins = [f"Coin{i}" for i in range(12)]
    slow = {c: 3.0 for c in coins[:8]}
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio", _slow_price(slow))
    prices = data_fetchers.fetch_prices(coins, {c: f"{c}-USD" for c in coins}, budget=1.0)
    assert all(prices[c] > 0 for c in coins[8:]) and all(prices[c] == 0 for c in coins[:8])


def test_f4_source_chain_fits_inside_the_price_budget():
    """AC09 — Binance ve OKX en kötü sürelerinin toplamı `fetch_prices` bütçesine sığar (ilk yüklemede fiyat boş kalmaz)."""
    assert data_fetchers.BINANCE_PRICE_BUDGET_SECONDS + data_fetchers.OKX_PRICE_TIMEOUT_SECONDS \
        <= data_fetchers.PRICE_BUDGET_SECONDS


def test_f6_unexpected_kline_body_is_labelled_invalid_not_unreachable(monkeypatch):
    """AC08 — Beklenmedik kline gövdesi ("yanıt yok" değil) "geçersiz yanıt" olarak etiketlenir ve çökmez."""
    _fake_net(monkeypatch, {"ticker/price": _Reply(403, {}), "klines": _Reply(200, [])})
    data_fetchers.PRICE_DIAGNOSTICS.clear()
    assert data_fetchers._live_price("LINKUSD") == 0
    assert any("Binance: geçersiz yanıt" == line for line in data_fetchers.price_diagnostics("LINKUSD"))


def test_f5_screen_lists_price_sources_for_an_unpriced_position(store, monkeypatch, processed_df):
    """AC12 — Fiyatı alınamayan pozisyon için ekranda "Fiyat kaynağı ayrıntısı" bölümü kaynakları gösterir; adres ve teknik metin yok."""
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD", "Chainlink": "LINKUSD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": "Chainlink", "Giriş": 15.0, "Adet": 20.0, "Yatırım": 300.0, "Realized": 0.0, "Status": "ACTIVE",
        "Tarih": "2026-09-01", "Stop": 13.0, "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00",
        "V1Verified": True}]})
    at = make_app(monkeypatch, processed_df)
    _fake_net(monkeypatch, {"ticker/price": _Reply(403, {}), "klines": _Reply(403, {}), "okx.com": _Reply(500, {})})
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda coin, coin_map: data_fetchers._live_price(coin_map[coin]))
    at.run()
    assert not at.exception
    shown = texts(at)
    assert "Chainlink: Binance: reddedildi; OKX: reddedildi" in shown
    assert "http" not in shown.split("Chainlink: Binance")[1][:80] and "Traceback" not in shown
