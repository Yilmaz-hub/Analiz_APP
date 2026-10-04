"""Spec 0006 B — pozisyon listesindeki güncel fiyat (R09, R11, R12, Q07)."""
from unittest import mock

import pytest

import data_fetchers
import positions

BLOCKED = 451


class _Providers:
    """Ağ yanıtlarını taklit eder: adres → (durum kodu, fiyat) ve Yahoo sembol → fiyat."""

    def __init__(self, binance=None, yahoo=None):
        self.binance = binance or {}
        self.yahoo = yahoo or {}
        self.binance_calls, self.yahoo_calls = [], []

    def get(self, url, params=None, **kwargs):
        host = url.split("/")[2]
        self.binance_calls.append((host, (params or {}).get("symbol")))
        outcome = self.binance.get(host, ("error", None))
        if outcome[0] == "error":
            raise ConnectionError(host)
        response = mock.Mock(status_code=outcome[0])
        response.json = lambda: {"symbol": params["symbol"], "price": outcome[1]}
        return response

    def ticker(self, symbol):
        self.yahoo_calls.append(symbol)
        price = self.yahoo.get(symbol)
        if price is None:
            raise ValueError(f"Yahoo: geçersiz sembol {symbol}")
        return mock.Mock(fast_info={"last_price": price})


def _price(symbol, providers):
    data_fetchers.get_live_price_for_portfolio.clear()  # 30 sn'lik önbellek testleri birbirine karıştırmasın
    with mock.patch.object(data_fetchers.requests, "get", providers.get), \
            mock.patch.object(data_fetchers.yf, "Ticker", providers.ticker):
        return data_fetchers.get_live_price_for_portfolio("Varlık", {"Varlık": symbol})


def test_ac19_dashless_symbol_gets_price_from_second_binance_host():
    """AC19 — İlk Binance adresi yanıt vermezken ikincisi fiyat döndürünce `LINKUSD` fiyatı bulunur."""
    providers = _Providers(binance={"data-api.binance.vision": ("error", None),
                                    "api.binance.us": (200, "18.50")})
    assert _price("LINKUSD", providers) == 18.5
    assert providers.binance_calls[-1] == ("api.binance.us", "LINKUSDT")


def test_ac20_second_dashless_example_is_found():
    """AC20 — Aynı koşulda `HBARUSD` sembollü varlığın güncel fiyatı bulunur."""
    providers = _Providers(binance={"data-api.binance.vision": (BLOCKED, None),
                                    "api.binance.us": (BLOCKED, None),
                                    "api.binance.com": (200, "0.2150")})
    assert _price("HBARUSD", providers) == 0.215
    assert [host for host, _ in providers.binance_calls] == [
        "data-api.binance.vision", "api.binance.us", "api.binance.com"]


def test_ac21_failing_first_source_falls_through_silently():
    """AC21 — İlk kaynak hata verdiğinde fiyat sıradaki kaynaktan alınır ve istisna kullanıcıya çıkmaz."""
    providers = _Providers(binance={"data-api.binance.vision": ("error", None),
                                    "api.binance.us": ("error", None),
                                    "api.binance.com": (200, "100.25")})
    assert _price("BTC-USD", providers) == 100.25


def test_ac22_yahoo_is_asked_with_the_dashed_form():
    """AC22 — Hiçbir Binance adresi yanıt vermediğinde `LINKUSD` için Yahoo'ya `LINK-USD` biçimiyle sorulur."""
    providers = _Providers(binance={}, yahoo={"LINK-USD": 18.4})
    assert _price("LINKUSD", providers) == 18.4
    assert providers.yahoo_calls == ["LINK-USD"]


def test_ac23_dashed_symbol_still_works_as_before():
    """AC23 — `LINK-USD` sembollü varlığın fiyatı bugünkü gibi bulunur (gerileme yok)."""
    providers = _Providers(binance={"data-api.binance.vision": (200, "18.50")})
    assert _price("LINK-USD", providers) == 18.5
    assert providers.binance_calls == [("data-api.binance.vision", "LINKUSDT")]


def test_ac23_usdt_pair_is_not_double_suffixed():
    """AC23 — `ETH-USDT` Binance'e `ETHUSDT` olarak sorulur (`ETHUSDTT` değil)."""
    providers = _Providers(binance={"data-api.binance.vision": (200, "3000")})
    assert _price("ETH-USDT", providers) == 3000.0
    assert providers.binance_calls[0][1] == "ETHUSDT"


def test_ac23_non_crypto_symbols_still_use_yahoo_only():
    """AC23 — Hisse sembolü Binance'e sorulmaz, Yahoo'dan alınır."""
    providers = _Providers(yahoo={"AAPL": 190.0})
    assert _price("AAPL", providers) == 190.0
    assert providers.binance_calls == []


def test_ac23_chart_fetchers_use_correct_exchange_symbols():
    """AC23 — Grafik kaynakları tiresiz sembolü doğru borsa biçimine çevirir (Binance `LINKUSDT`, OKX `LINK-USDT`)."""
    import market_map

    assert market_map.binance_symbol("LINKUSD") == market_map.binance_symbol("LINK-USD") == "LINKUSDT"
    assert market_map.binance_symbol("ETH-USDT") == "ETHUSDT"
    assert market_map.okx_symbol("LINKUSD") == market_map.okx_symbol("LINK-USD") == "LINK-USDT"
    assert market_map.binance_symbol("AAPL") is None and market_map.okx_symbol("THYAO.IS") is None


def _pos(coin="Chainlink", entry=10.0, qty=10.0):
    return {"Coin": coin, "Giriş": entry, "Adet": qty, "Yatırım": entry * qty, "Status": "ACTIVE"}


def test_ac27_row_uses_live_price_for_profit_not_zero():
    """AC27 — Sonradan eklenmiş varlığın satırı kaynaktan gelen güncel fiyat ve ona göre kâr/zararla görünür; kâr 0 değildir."""
    rows, total = positions.build_active_rows([_pos()], lambda coin: 18.5)
    assert rows[0]["Değer ($)"] == 185.0 and rows[0]["Kar/Zarar ($)"] == 85.0
    assert rows[0]["Kar/Zarar (%)"] == "%85.00"
    assert total == 185.0


def test_ac31_row_with_live_price_has_no_label():
    """AC31 — Fiyat bulunduğunda satır etiket taşımaz ve bugünkü gibi hesaplanır."""
    rows, _ = positions.build_active_rows([_pos()], lambda coin: 18.5)
    assert rows[0]["Fiyat"] == "canlı"
    assert positions.unpriced_count(rows) == 0


def test_ac30_unavailable_price_never_shows_zero_profit():
    """AC30 — Tüm kaynaklar yanıt vermezse satırda kâr/zarar 0 yazmaz; nedenini belirtir."""
    rows, _ = positions.build_active_rows([_pos()], lambda coin: 0)
    assert rows[0]["Kar/Zarar ($)"] == "hesaplanamıyor"
    assert rows[0]["Kar/Zarar (%)"] == "hesaplanamıyor"
    assert rows[0]["Fiyat"] == "fiyat alınamadı (maliyetle gösterildi)"


def test_ac30_total_value_reports_unpriced_positions():
    """AC30 — Fiyatı alınamayan pozisyon varsa toplam değer bunu belirtir (sayı ve açıklama)."""
    rows, total = positions.build_active_rows(
        [_pos(), _pos(coin="Hedera", entry=1.0, qty=100.0)],
        lambda coin: 18.5 if coin == "Chainlink" else 0)
    assert positions.unpriced_count(rows) == 1
    assert total == 185.0 + 100.0
    assert "1 pozisyonun fiyatı alınamadı" in positions.total_value_note(rows)
    assert positions.total_value_note(positions.build_active_rows([_pos()], lambda c: 18.5)[0]) == ""
