"""Q7: tek "İşlem varsayımları"; boş=bilinmiyor, 0=açıkça sıfır; kalıcı; negatif reddedilir."""
from decimal import Decimal

import pytest

import storage
import trade_settings as ts
from trade_execution import execute_purchase

D = Decimal


def test_q7_empty_costs_are_unknown_not_zero():
    parsed = ts.parse_settings({"capital": "10000", "notional": "1000"})
    assert parsed.ok
    costs = parsed.settings.costs
    assert (costs.spread_bps, costs.slippage_bps, costs.commission_pct) == (None, None, None)
    assert parsed.settings.quantity_step is None and parsed.settings.net_verified is False


def test_q7_explicit_zero_is_known_zero():
    parsed = ts.parse_settings({"spread_bps": "0", "slippage_bps": "0", "commission_pct": "0"})
    assert parsed.settings.costs.commission_pct == D(0) and parsed.settings.net_verified is True


def test_q7_defaults_are_10000_and_1000():
    parsed = ts.parse_settings({})
    assert (parsed.capital, parsed.settings.notional) == (D(10000), D(1000))    # AC90


def test_q7_comma_decimal_is_accepted():
    assert ts.parse_settings({"commission_pct": "0,1"}).settings.costs.commission_pct == D("0.1")


@pytest.mark.parametrize("field,value", [
    ("commission_pct", "-0.1"), ("commission_pct", "100"), ("commission_pct", "nan"),
    ("spread_bps", "-1"), ("slippage_bps", "-5"), ("quantity_step", "0"),
    ("notional", "0"), ("capital", "-5"), ("commission_pct", "abc"),
])
def test_q7_invalid_values_are_rejected_before_any_evaluation(field, value):
    parsed = ts.parse_settings({field: value})
    assert not parsed.ok and parsed.settings is None and parsed.errors


def test_q7_negative_commission_message_is_human_readable():
    assert "Komisyon negatif olamaz" in ts.parse_settings({"commission_pct": "-1"}).errors[0]   # AC26


def test_q7_settings_are_persisted_and_per_asset(store):
    ts.save_raw("Bitcoin (BTC)", {"capital": "25000", "notional": "2500", "commission_pct": "0.1"})
    ts.save_raw("Ethereum (ETH)", {"spread_bps": "50"})
    btc = ts.parse_settings(ts.load_raw("Bitcoin (BTC)"))
    eth = ts.parse_settings(ts.load_raw("Ethereum (ETH)"))
    assert (btc.capital, btc.settings.notional) == (D(25000), D(2500))         # AC91
    assert btc.settings.costs.spread_bps is None and eth.settings.costs.spread_bps == D(50)
    assert eth.settings.costs.commission_pct is None                            # AC98: başka varlık değişmez


def test_q7_unreadable_store_raises_instead_of_defaulting(store):
    from app_helpers import break_storage
    geri_al = break_storage(store)
    try:
        with pytest.raises(storage.StorageAccessError):
            ts.load_raw("Bitcoin (BTC)")
    finally:
        geri_al()


def test_q7_execute_purchase_distinguishes_missing_stop_from_explicit_none():
    base = dict(fee=D(0), quantity_step=D(1))
    assert execute_purchase(D(10000), D(1000), D(100), **base).executed          # verilmedi: atla
    explicit = execute_purchase(D(10000), D(1000), D(100), initial_stop=None, **base)
    assert not explicit.executed and explicit.reason == "GECERSIZ_STOP"           # AC110


# ---- Uygulama: tek panel, backtest ve sanal takip aynı değeri okur ---------------
def _istek(at, **alanlar):
    for ad, deger in alanlar.items():
        at.text_input(key=f"ts:BTC-USD:{ad}").set_value(deger)
    return at.run()


def _gunluk(at):
    for sb in at.selectbox:
        if sb.label == "Periyot:":
            return sb.set_value("1d").run()
    raise AssertionError("Periyot seçimi bulunamadı")


def test_q7_app_negative_commission_blocks_evaluation_with_message(store, monkeypatch, processed_df):
    from app_helpers import make_app, texts
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = _gunluk(make_app(monkeypatch, processed_df).run())
    at = _istek(at, commission_pct="-0.5")
    assert not at.exception
    assert "Komisyon negatif olamaz" in texts(at)
    assert [b for b in at.button if b.label == "🚀 Backtest Başlat"][0].disabled
    assert [b for b in at.button if b.label == "📸 Bugünü Kaydet / Güncelle"][0].disabled


def test_q7_app_unknown_step_backtest_reports_blocked_entries_and_unverified_net(
        store, monkeypatch, processed_df):
    import technical_analysis
    from app_helpers import click, make_app, texts
    monkeypatch.setattr(technical_analysis, "build_v1_decisions",
                        lambda df, **k: {df.index[i]: "AL" for i in range(199, len(df))})
    import app_helpers  # noqa: F401
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = _gunluk(make_app(monkeypatch, processed_df).run())
    at = click(at, "🚀 Backtest Başlat")
    assert not at.exception
    metin = texts(at)
    assert "Miktar adımı bilinmiyor" in metin
    assert "doğrulanmış net sonuç değildir" in metin
