"""Spec 0006 — bağımsız QA bulguları (K1–K11) ve 0005 QA notları (N2a, N2b) düzeltme testleri."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import cash_flows
import market_map
import trade_confirmation as tc
from app_helpers import click, make_app, texts
from position_journal import PositionJournal

D = Decimal
UTC = timezone.utc
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
TRADE_AT = datetime(2026, 9, 10, 9, 30, tzinfo=UTC)
COIN = "Bitcoin (BTC)"
COINS = {COIN: "BTC-USD"}


def _held():
    portfolio, journal = {"balance": 5000.0, "positions": []}, PositionJournal()
    assert tc.confirm_buy(portfolio, journal, lambda: True, coin=COIN, symbol="BTC-USD", quantity=D("10"),
                          price=D("100"), stop=D("90"), executed_at=TRADE_AT, now=NOW).ok
    return portfolio, journal


def _sell(portfolio, journal, quantity, price, hours):
    return tc.confirm_sell(portfolio, journal, lambda: True, position=portfolio["positions"][0],
                           symbol="BTC-USD", quantity=D(str(quantity)), price=D(str(price)),
                           executed_at=TRADE_AT + timedelta(hours=hours), now=NOW)


# ---- K1: kısmi satış zararı kayıp sınırına girer -----------------------------------------------
def test_k1_partial_sale_loss_counts_toward_the_loss_limit(store):
    """AC32 — Kısmi satışın zararı, pozisyon kapanmadan da dönem kayıp sınırına parça parça girer."""
    import risk_ui

    portfolio, journal = _held()
    assert _sell(portfolio, journal, 5, 50, 1).ok               # 5 adet × (50−100) = −250
    assert risk_ui.period_results(portfolio, COINS, TRADE_AT - timedelta(days=1)) == [("USD", D("-250"))]


def test_k1_only_exits_after_the_period_start_count(store):
    """AC32 — Dönem başlangıcından önceki parça sayılmaz; yalnız sonraki parça girer, son parça kapanınca toplam bozulmaz."""
    import risk_ui

    portfolio, journal = _held()
    _sell(portfolio, journal, 5, 50, 1)                          # −250, dönem öncesi
    _sell(portfolio, journal, 5, 90, 2)                          # −50, dönem içi; pozisyon kapanır
    started = TRADE_AT + timedelta(minutes=90)
    assert risk_ui.period_results(portfolio, COINS, started) == [("USD", D("-50"))]
    assert risk_ui.period_results(portfolio, COINS, TRADE_AT - timedelta(days=1)) == [
        ("USD", D("-250")), ("USD", D("-50"))]


def test_k1_legacy_single_exit_record_still_counts(store):
    """AC32 — Çıkış listesi olmayan eski kapanmış kayıt eskisi gibi tek sonuç sayılır."""
    import risk_ui

    legacy = {"Coin": COIN, "Status": "CLOSED_CONFIRMED", "Realized": -30.0,
              "Çıkış Zamanı": TRADE_AT.isoformat()}
    assert risk_ui.period_results({"positions": [legacy]}, COINS, TRADE_AT - timedelta(days=1)) == [
        ("USD", D("-30"))]


# ---- K2: bekleyen emir tablosu fiyat yokken uydurma değer göstermez -----------------------------
def test_k2_pending_order_without_price_shows_reason_not_a_fake_distance(store, monkeypatch, processed_df):
    """AC33 — Fiyat alınamayınca bekleyen emir satırı "fiyat alınamadı" der; anlamsız uzaklık yüzdesi göstermez."""
    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD", "Ethereum (ETH)": "ETH-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": "Ethereum (ETH)", "Giriş": 2000.0, "Adet": 0.5, "Yatırım": 1000.0, "Realized": 0.0,
        "Status": "PENDING", "Tarih": "2026-09-01", "Stop": 1800.0}]})
    at = make_app(monkeypatch, processed_df, live_price=0).run()
    assert not at.exception
    rows = [frame.value for frame in at.dataframe if "Hedef Giriş" in frame.value.columns]
    assert rows, "bekleyen emir tablosu görünmedi"
    cells = " ".join(str(v) for v in rows[0].to_numpy().ravel())
    assert "1000000" not in cells and "fiyat alınamadı" in cells and "hesaplanamıyor" in cells


# ---- K3: AC09 gerçek tüketicilerle -------------------------------------------------------------
def test_k3_position_counts_as_closed_only_after_the_last_part(store):
    """AC09 — Üretimdeki sınıflandırıcıya göre pozisyon ilk parçadan sonra aktif kalır, yalnız son parçadan sonra kapanmış sayılır."""
    import positions

    portfolio, journal = _held()
    _sell(portfolio, journal, 4, 110, 1)
    assert positions.classify_position(portfolio["positions"][0]) == positions.AKTIF
    _sell(portfolio, journal, 6, 110, 2)
    assert positions.classify_position(portfolio["positions"][0]) == positions.KAPALI


def test_k3_loss_cooldown_starts_only_when_the_whole_position_is_closed(store):
    """AC09 — Zararlı kısmi satış tekrar-alım bekleme süresini başlatmaz; tam kapanıştan sonra başlatır."""
    import pandas as pd

    from trading_ui import bars_since_loss_exit

    portfolio, journal = _held()
    index = pd.date_range("2026-09-09", periods=6, freq="D", tz=UTC)
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    _sell(portfolio, journal, 4, 80, 1)
    assert bars_since_loss_exit(portfolio["positions"], COIN, index, now) is None
    _sell(portfolio, journal, 6, 80, 2)
    assert bars_since_loss_exit(portfolio["positions"], COIN, index, now) is not None


# ---- K4: R08 — ekran kısmi satışı önermez --------------------------------------------------------
def test_k4_sell_screen_does_not_recommend_partial_selling(store, monkeypatch, processed_df):
    """AC18 — Satış ekranı "tamamını kapatır" bilgisini verir ama kısmi satışı önermez; kayıt ekran açılınca değişmez."""
    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD"})
    portfolio = {"balance": 1000.0, "positions": [{
        "Coin": COIN, "Giriş": 100.0, "Adet": 10.0, "Yatırım": 1000.0, "Realized": 0.0,
        "Status": "ACTIVE", "Tarih": "2026-09-01", "Stop": 90.0,
        "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00", "V1Verified": True}]}
    store.write_doc(store.PORTFOLIO_KEY, portfolio)
    at = make_app(monkeypatch, processed_df).run()
    assert not at.exception
    shown = texts(at)
    assert "tamamını kapatır" in shown
    assert "kâr almak için" not in shown and "bir bölümünü de satabilirsiniz" not in shown
    assert store.read_doc(store.PORTFOLIO_KEY) == portfolio


# ---- K5: varlık ekleme bilgisi okunabilir kalır --------------------------------------------------
def test_k5_asset_added_message_stays_visible_after_the_rerun(store, monkeypatch, processed_df):
    """AC28 — Varlık eklendikten sonra "LINK-USD, kripto olarak okunacak" bilgisi ekranda kalır, hemen silinmez."""
    at = make_app(monkeypatch, processed_df).run()
    at.text_input[next(i for i, t in enumerate(at.text_input) if t.label.startswith("Görünen"))].set_value("Chainlink")
    at.text_input[next(i for i, t in enumerate(at.text_input) if t.label.startswith("Yahoo Kodu"))].set_value("LINKUSD")
    at = click(at, "Listeye Ekle")
    assert not at.exception
    assert "LINK-USD, kripto olarak okunacak" in texts(at)
    assert store.read_doc(store.ASSETS_KEY)["Chainlink"] == "LINKUSD"


# ---- K7: grafik kaynakları tiresiz sembolü doğru borsa biçimiyle sorar ----------------------------
def test_k7_chart_fetchers_send_the_exchange_symbol(monkeypatch):
    """AC23 — Gerçek grafik kaynakları `LINKUSD` için Binance'e `LINKUSDT`, OKX'e `LINK-USDT` gönderir."""
    import data_fetchers

    seen = []

    class _Response:
        status_code = 200

        def json(self):
            return []

    def fake_get(url, params=None, **kwargs):
        seen.append(params)
        return _Response()

    monkeypatch.setattr(data_fetchers.requests, "get", fake_get)
    data_fetchers.fetch_binance_simple("LINKUSD", "1d")
    data_fetchers.fetch_okx_simple("LINKUSD", "1d")
    assert seen[0]["symbol"] == "LINKUSDT" and seen[-1]["instId"] == "LINK-USDT"


# ---- K8: sembol yorumu döviz/altın çiftlerini kripto saymaz ---------------------------------------
def test_k8_fiat_and_metal_pairs_are_not_read_as_crypto():
    """AC34 — `GBPUSD`, `EURUSD`, `XAUUSD` gibi döviz/altın çiftleri kripto olarak yorumlanmaz."""
    for symbol in ("GBPUSD", "EURUSD", "XAUUSD", "USDUSD", "TRYUSD", "MXNUSD", "ZARUSD", "BRLUSD", "INRUSD",
                   "KRWUSD", "PLNUSD", "HKDUSD", "SGDUSD", "AUDUSD", "THBUSD"):
        assert market_map.canonical_symbol(symbol) == symbol, symbol
        assert market_map.market_of(symbol) is None, symbol
    assert market_map.canonical_symbol("LINKUSD") == "LINK-USD"
    assert market_map.canonical_symbol("HBARUSDT") == "HBAR-USDT"
    for crypto in ("BTCUSD", "SOLUSD", "ADAUSD", "DOTUSD", "XRPUSD", "USDCUSDT", "PAXGUSDT", "MNTUSD", "CROUSD",
                   "SCRUSDT", "BOBUSDT", "SBDUSD", "TOPUSDT", "SOSUSDT"):
        assert market_map.market_of(crypto) == ("KRIPTO", crypto[-4:] if crypto.endswith("USDT") else "USD"), crypto


def test_k8_yahoo_fallback_converts_usdt_to_usd(monkeypatch):
    """AC34 — Binance yanıt vermezse Yahoo'ya `LINKUSDT` için `LINK-USD` biçimiyle sorulur (Yahoo USDT çiftini tanımaz)."""
    import data_fetchers

    asked = []

    class _Ticker:
        def __init__(self, symbol):
            asked.append(symbol)
            self.fast_info = {"last_price": 18.4}

    monkeypatch.setattr(data_fetchers, "_binance_price", lambda symbol: None)
    monkeypatch.setattr(data_fetchers.yf, "Ticker", _Ticker)
    assert data_fetchers._live_price("LINKUSDT") == 18.4
    assert asked == ["LINK-USD"]


# ---- K9: canlı fiyat zinciri toplam süre bütçesine uyar ------------------------------------------
def test_k9_binance_chain_stops_after_its_time_budget(monkeypatch):
    """AC35 — Binance adresleri yanıt vermezse zincir toplam süre bütçesini aşmaz; kalan adresler denenmeden Yahoo'ya geçilir."""
    import data_fetchers

    clock = [0.0]
    timeouts = []

    def slow_get(url, params=None, timeout=None, **kwargs):
        timeouts.append(timeout)
        clock[0] += sum(timeout)          # en kötü durum: bağlantı ve okuma zaman aşımları birikir
        raise TimeoutError("zaman aşımı")

    monkeypatch.setattr(data_fetchers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(data_fetchers.requests, "get", slow_get)
    assert data_fetchers._binance_price("BTC-USD") is None
    assert clock[0] <= data_fetchers.BINANCE_PRICE_BUDGET_SECONDS + 0.01   # toplam bekleme bütçeyi aşmaz
    assert timeouts and all(sum(t) <= data_fetchers.BINANCE_PRICE_BUDGET_SECONDS for t in timeouts)


# ---- K11: kayıt zamanı sabit saatle sınanır -------------------------------------------------------
def test_k11_record_time_is_the_real_clock_not_the_execution_time(store, monkeypatch):
    """AC15 — Kayıt zamanı gerçek saat, işlem zamanı kullanıcının verdiği (dünkü) zamandır; ikisi karışmaz."""
    import position_journal

    fixed = datetime(2026, 9, 11, 8, 0, tzinfo=UTC)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(tc, "datetime", _Clock)
    monkeypatch.setattr(position_journal, "datetime", _Clock, raising=False)
    portfolio, journal = _held()
    executed = TRADE_AT + timedelta(hours=1)
    _sell(portfolio, journal, 4, 110, 1)
    trade = PositionJournal().trades[-1]
    assert trade.executed_at == executed
    assert trade.recorded_at == fixed


# ---- N2a / N2b: nakit hareketi telafisi --------------------------------------------------------------
def _balance_screen(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": []})
    at = make_app(monkeypatch, processed_df).run()
    next(n for n in at.number_input if n.label == "Güncel USDT Bakiyesi").set_value(1750.0).run()
    return at


def test_n2a_failed_compensation_is_shown_to_the_user(store, monkeypatch, processed_df):
    """AC14 — Bakiye yazılamayıp telafi kaydı da yazılamazsa kullanıcı bu durumu açıkça görür."""
    from storage import StorageAccessError

    real_write, real_record = store.write_doc, cash_flows.record
    calls = []

    def write(key, payload):
        if key == store.PORTFOLIO_KEY and calls:
            raise StorageAccessError("yazılamıyor")
        return real_write(key, payload)

    def record(delta, at, note="", reverses=None):
        calls.append(delta)
        if len(calls) > 1:
            raise StorageAccessError("yazılamıyor")
        return real_record(delta, at, note, reverses)

    at = _balance_screen(store, monkeypatch, processed_df)
    monkeypatch.setattr(store, "write_doc", write)
    monkeypatch.setattr(cash_flows, "record", record)
    at = click(at, "Bakiyeyi Güncelle")
    assert not at.exception
    assert "geri alınamadı" in texts(at) and "StorageAccessError" not in texts(at)


def test_n2b_reversed_flow_pair_does_not_block_the_comparison_day(store):
    """AC14 — Telafi ile geri alınan hareket (+x ve −x) gerçek para hareketi sayılmaz; o günün karşılaştırması engellenmez."""
    at = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    original = cash_flows.record(D("750"), at, "elle bakiye güncelleme")
    cash_flows.record(D("-750"), at, "geri alma", reverses=original)
    cash_flows.record(D("100"), at + timedelta(days=2), "elle bakiye güncelleme")
    assert [d.isoformat() for d in cash_flows.flow_days()] == ["2026-10-06"]


def test_n_c_pending_price_column_is_text_only_so_the_table_converts_cleanly(store, monkeypatch, processed_df):
    """AC33 — Bekleyen emir tablosunun "Anlık Fiyat" sütunu tek türdendir (metin); karışık tür günlüğe hata düşürmez."""
    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD", "Ethereum (ETH)": "ETH-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": "Ethereum (ETH)", "Giriş": 2000.0, "Adet": 0.5, "Yatırım": 1000.0, "Realized": 0.0,
        "Status": "PENDING", "Tarih": "2026-09-01", "Stop": 1800.0}]})
    at = make_app(monkeypatch, processed_df, live_price=2100.0).run()
    frame = next(f.value for f in at.dataframe if "Hedef Giriş" in f.value.columns)
    assert all(isinstance(v, str) for v in frame["Anlık Fiyat"])
