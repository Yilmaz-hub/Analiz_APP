"""Spec 0006 B — pozisyon listesi ekranı (AC27, AC30)."""
from app_helpers import make_app, texts

ASSETS = {"Bitcoin (BTC)": "BTC-USD", "Chainlink": "LINKUSD"}


def _position(coin="Chainlink"):
    return {"Coin": coin, "Giriş": 10.0, "Adet": 10.0, "Yatırım": 100.0, "Realized": 0.0,
            "Status": "ACTIVE", "Tarih": "2026-09-01", "Stop": 8.0,
            "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00", "V1Verified": True}


def _open(store, monkeypatch, processed_df, live_price):
    store.write_doc(store.ASSETS_KEY, ASSETS)
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [_position()]})
    return make_app(monkeypatch, processed_df, live_price=live_price).run()


def _profit_cells(at):
    return [row for frame in at.dataframe if "Kar/Zarar ($)" in frame.value.columns
            for row in frame.value["Kar/Zarar ($)"].tolist()]


def test_ac27_added_asset_row_shows_real_profit_without_opening_its_chart(store, monkeypatch, processed_df):
    """AC27 — Sonradan eklenmiş `LINKUSD` varlığının satırı, grafiği açılmadan, güncel fiyata göre gerçek kârla görünür."""
    at = _open(store, monkeypatch, processed_df, live_price=18.5)
    assert not at.exception
    assert 85.0 in _profit_cells(at)          # 10 adet × 18,5 − 100 yatırım
    assert 0.0 not in _profit_cells(at)


def test_ac30_unpriced_position_is_flagged_on_screen_not_shown_as_zero(store, monkeypatch, processed_df):
    """AC30 — Hiçbir kaynaktan fiyat gelmezse ekran kâr yerine "hesaplanamıyor" ve toplam değer uyarısı gösterir."""
    at = _open(store, monkeypatch, processed_df, live_price=0)
    assert not at.exception
    assert "hesaplanamıyor" in _profit_cells(at)
    assert "1 pozisyonun fiyatı alınamadı" in texts(at)
