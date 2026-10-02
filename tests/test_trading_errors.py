"""AC70: sağlayıcı zaman aşımı başarı sayılmaz ve eylem üretmez (gerçek veri yolu)."""
import data_fetchers

from app_helpers import make_app, texts


def test_ac70_provider_timeout_returns_no_data_instead_of_raising(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("provider late")

    monkeypatch.setattr(data_fetchers.requests, "get", timeout)
    result = data_fetchers.fetch_binance_simple("ZZTIMEOUT-USD", "1d")
    assert result is None


def test_ac70_app_shows_an_error_and_no_action_when_the_provider_gives_no_data(store, monkeypatch):
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, None).run()
    assert not at.exception
    assert any("Veri alınamadı" in e.value for e in at.error)
    assert "Pozisyonuna göre eylem" not in texts(at)
