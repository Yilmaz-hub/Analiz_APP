"""Spec 0010 — pozisyondaki varlık adı ile varlık listesindeki ad eşleşmesi."""
import assets

MAP = {"Chainlink (LINK)": "LINK-USD", "Hbar": "HBAR-USD", "Bitcoin (BTC)": "BTC-USD"}


def test_ac01_exact_and_case_insensitive_names_resolve():
    """AC01 — Birebir ve büyük/küçük harf ya da boşluk farkıyla yazılmış ad listedeki varlığa çözülür."""
    assert assets.resolve_name(MAP, "Hbar") == "Hbar"
    assert assets.resolve_name(MAP, " hbar ") == "Hbar"


def test_ac02_short_name_resolves_to_the_unique_asset_containing_it():
    """AC02 — `link` gibi kısa ad, adında ya da sembolünde geçen tek varlığa çözülür."""
    assert assets.resolve_name(MAP, "link") == "Chainlink (LINK)"


def test_ac03_ambiguous_or_unknown_names_are_not_guessed():
    """AC03 — Birden fazla aday ya da hiç aday yoksa tahmin edilmez (None)."""
    two = {"Link A": "LINK-USD", "Link B": "LINKB-USD"}
    assert assets.resolve_name(two, "link") is None
    assert assets.resolve_name(MAP, "Silinmis") is None
    assert assets.resolve_name(MAP, "") is None and assets.resolve_name(MAP, None) is None


def test_ac04_screen_prices_a_position_whose_name_differs_from_the_asset_list(store, monkeypatch, processed_df):
    """AC04 — Adı listeden farklı yazılmış pozisyon (link ↔ Chainlink (LINK)) ekranda fiyatlanır, "bulunamadı" yazmaz."""
    import data_fetchers
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Chainlink (LINK)": "LINKUSD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": "link", "Giriş": 10.0, "Adet": 20.0, "Yatırım": 200.0, "Realized": 0.0,
        "Tarih": "2026-09-01", "Stop": 8.0}]})
    at = make_app(monkeypatch, processed_df)
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda coin, coin_map: 18.0 if coin_map.get(coin) else 0)
    at.run()
    assert not at.exception
    table = next(f.value for f in at.dataframe if "Fiyat" in f.value.columns)
    assert list(table["Fiyat"]) == ["canlı"]
    assert "varlık listesinde bulunamadı" not in texts(at)
