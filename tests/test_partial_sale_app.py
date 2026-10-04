"""Spec 0006 A — "Kar Al / Satış Yap" ekranı (AC17)."""
from app_helpers import button, click, make_app, texts

COIN = "Bitcoin (BTC)"


def _open(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, {COIN: "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [{
        "Coin": COIN, "Giriş": 100.0, "Adet": 10.0, "Yatırım": 1000.0, "Realized": 0.0,
        "Status": "ACTIVE", "Tarih": "2026-09-01", "Stop": 90.0,
        "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00", "V1Verified": True}]})
    return make_app(monkeypatch, processed_df).run()


def _position(store):
    return store.read_doc(store.PORTFOLIO_KEY)["positions"][0]


def test_ac17_percent_box_fills_quantity_and_partial_sale_keeps_position_listed(store, monkeypatch, processed_df):
    """AC17 — Satış ekranında yüzde kutusu görünür; %40 girilip onaylanınca pozisyon kalan miktarla listede kalır."""
    at = _open(store, monkeypatch, processed_df)
    assert not at.exception
    assert at.number_input(key=f"sell_qty:{COIN}").value == 10.0      # varsayılan: tamamı (eskisi gibi)
    at = at.number_input(key=f"sell_pct:{COIN}").set_value(40.0).run()
    assert at.number_input(key=f"sell_qty:{COIN}").value == 4.0
    at = click(at, "Satışı Onayla")
    assert not at.exception
    position = _position(store)
    assert position["Status"] == "ACTIVE" and position["Adet"] == 6.0 and position["Stop"] == 90.0
    assert any(frame.value["Adet"].tolist() == [6.0] for frame in at.dataframe
               if "Adet" in frame.value.columns)


def test_ac17_preset_buttons_set_percent_and_quantity(store, monkeypatch, processed_df):
    """AC17 — Hazır %25, %50, %75, %100 düğmeleri yüzdeyi ve satılacak miktarı doldurur."""
    at = _open(store, monkeypatch, processed_df)
    for label, expected in (("%25", 2.5), ("%50", 5.0), ("%75", 7.5), ("%100", 10.0)):
        at = button(at, label).click().run()
        assert at.number_input(key=f"sell_qty:{COIN}").value == expected, label


def test_ac17_percent_rounding_to_zero_shows_reason_and_blocks_nothing_else(store, monkeypatch, processed_df):
    """AC17 — Miktar adımının altına düşen yüzde nedeniyle uyarılır; serbest miktar girişi çalışmaya devam eder."""
    store.write_doc(store.TRADE_SETTINGS_KEY, {COIN: {"quantity_step": "1"}})
    at = _open(store, monkeypatch, processed_df)
    at = at.number_input(key=f"sell_pct:{COIN}").set_value(5.0).run()
    assert "miktar adımının altında" in texts(at)
    at = at.number_input(key=f"sell_qty:{COIN}").set_value(3.0).run()
    at = click(at, "Satışı Onayla")
    assert _position(store)["Adet"] == 7.0
