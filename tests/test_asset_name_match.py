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


# ---- Spec 0011 ---------------------------------------------------------------------------------
def test_ac05_name_is_suggested_from_the_code():
    """AC05 — Yahoo kodundan "Ad (KOD)" önerilir; LINKUSD ve LINK-USD aynı öneriyi verir, bilinmeyen kod için öneri yoktur."""
    assert assets.suggest_name({}, "LINKUSD") == "Chainlink (LINK)"
    assert assets.suggest_name({}, "link-usd") == "Chainlink (LINK)"
    assert assets.suggest_name({}, "BTC-USD") == "Bitcoin (BTC)"          # varsayılan listedeki ad
    assert assets.suggest_name({}, "ZZZZ-USD") is None and assets.suggest_name({}, "") is None
    assert assets.suggest_name({"Chainlink (LINK)": "LINK-USD"}, "LINKUSD") is None   # zaten listede


def test_ac06_new_position_records_its_own_symbol(store):
    """AC06 — Yeni pozisyon kaydı `Sembol` alanını taşır."""
    from datetime import datetime, timezone

    import trade_confirmation as tc

    portfolio = {"balance": 1000.0, "positions": []}

    class _Journal:
        trades = []

        def confirm_trade(self, *a, **k):
            return None

        def has_event(self, *a, **k):
            return False

    now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    outcome = tc.confirm_buy(portfolio, _Journal(), lambda: True, coin="Chainlink (LINK)", symbol="LINKUSD",
                             quantity=1, price=10, stop=9, executed_at=now, now=now, use_balance=True,
                             is_limit=False)
    assert outcome.ok, outcome.code
    assert portfolio["positions"][0]["Sembol"] == "LINKUSD"


def test_ac07_position_is_priced_by_its_own_symbol_when_the_asset_was_removed(store, monkeypatch, processed_df):
    """AC07 — Varlık listeden silinmiş olsa bile `Sembol` taşıyan pozisyon fiyatlanır."""
    import data_fetchers
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": "Chainlink (LINK)", "Sembol": "LINKUSD", "Giriş": 10.0, "Adet": 20.0, "Yatırım": 200.0,
        "Realized": 0.0, "Status": "ACTIVE", "Tarih": "2026-09-01", "Stop": 8.0}]})
    at = make_app(monkeypatch, processed_df)
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda coin, coin_map: 18.0 if coin_map.get(coin) == "LINKUSD" else 0)
    at.run()
    assert not at.exception
    table = next(f.value for f in at.dataframe if "Fiyat" in f.value.columns)
    assert list(table["Fiyat"]) == ["canlı"]


def test_ac08_screen_offers_the_suggested_name_and_fills_it(store, monkeypatch, processed_df):
    """AC08 — Kod yazılınca ad önerisi görünür; "Öneriyi kullan" ad alanını doldurur."""
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df)
    at.run()
    code = next(t for t in at.text_input if t.label.startswith("Yahoo Kodu"))
    code.set_value("LINKUSD").run()
    assert any("Önerilen ad: **Chainlink (LINK)**" in c.value for c in at.sidebar.caption)
    next(b for b in at.button if b.label == "Öneriyi kullan").click().run()
    assert next(t for t in at.text_input if t.label.startswith("Görünen")).value == "Chainlink (LINK)"
