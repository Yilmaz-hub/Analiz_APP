"""Spec 0014 — geçmiş testte işleme dönüşmeyen alımların nedeni anlaşılır ve eyleme dönük yazılır."""
from decimal import Decimal

import trade_settings as ts
from trading_ui import describe_blocked


def _settings(step):
    return ts.parse_settings({"capital": "10000", "notional": "1000", "quantity_step": step,
                              "spread_bps": "0", "slippage_bps": "0", "commission_pct": "0"}).settings


def test_ac01_too_large_step_names_the_setting_and_the_fix():
    """AC01 — Adım işlem tutarına sığmıyorsa metin tutarı, adımı ve ne yapılacağını söyler; ham kod yok."""
    text = describe_blocked("ASGARI_MIKTAR", 231, _settings("5"), "USD/USDT")
    assert text.startswith("231 alım sinyali işleme dönüşmedi")
    assert "1000 USD/USDT" in text and "5 adet" in text and "0.001" in text
    assert "ASGARI_MIKTAR" not in text


def test_ac02_missing_step_points_to_the_panel():
    """AC02 — Adım girilmemişse metin ⚙️ İşlem varsayımları paneline ve örnek değere yönlendirir."""
    text = describe_blocked("MIKTAR_ADIMI_BILINMIYOR", 12, _settings(""), "USD")
    assert "Adet/lot adımı girilmemiş" in text and "İşlem varsayımları" in text


def test_ac03_other_reasons_still_read_as_turkish_text():
    """AC03 — Diğer nedenler mevcut Türkçe kod metniyle yazılır."""
    assert "Nakit yetersiz" not in describe_blocked("YETERSIZ_NAKIT", 1, _settings("0.001"))
    cash = describe_blocked("YETERSIZ_NAKIT", 1, _settings("0.001"))
    assert "önceki zararlar" in cash and "işlem tutarını azaltın" in cash
    assert "İşlem maliyeti geçersiz" in describe_blocked("GECERSIZ_MALIYET", 1, _settings("0.001"))


def test_ac04_backtest_screen_explains_why_no_purchase_happened(store, monkeypatch, processed_df):
    """AC04 — Ekranda: büyük adımla geçmiş test, alımların neden yapılmadığını ayar adıyla açıklar."""
    import technical_analysis
    from app_helpers import click, make_app

    monkeypatch.setattr(technical_analysis, "build_v1_decisions",
                        lambda df, **k: {df.index[199]: "AL", df.index[205]: "AL"})
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("500")
    at = click(at.run(), "🚀 Backtest Başlat")
    assert not at.exception
    shown = " ".join(str(w.value) for w in at.warning)
    assert "alım sinyali işleme dönüşmedi" in shown and "almaya yetmiyor" in shown


def test_ac05_every_setting_has_a_help_text(store, monkeypatch, processed_df):
    """AC05 — İşlem varsayımları panelindeki her alanın açıklaması (help) vardır; zorunlu alan söylenir."""
    from app_helpers import make_app, texts

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    fields = [t for t in at.text_input if t.key and t.key.startswith("ts:BTC-USD:")]
    assert len(fields) == 6 and all(t.help for t in fields)
    assert "zorunlu tek bilgi" in texts(at)


def test_ac01_fractional_notional_is_not_rounded_in_the_message():
    """AC01 — Ondalıklı işlem tutarı mesajda yuvarlanmadan yazılır (0.5 → "0.5", 1000.75 → "1000.75")."""
    for notional, shown in (("0.5", "(0.5)"), ("1000.75", "(1000.75)")):
        settings = ts.parse_settings({"notional": notional, "quantity_step": "5"}).settings
        assert shown in describe_blocked("ASGARI_MIKTAR", 1, settings)
