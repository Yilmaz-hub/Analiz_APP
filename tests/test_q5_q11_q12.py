"""Q5 / Q11 / Q12 (spec 0003 AK17): geçersiz veri BEKLE sayılmaz; karar zamanı
dürüst; ekranda ham makine kodu görünmez."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from app_helpers import make_app, texts
from trading_contracts import Position, Signal
from trading_ui import (
    CODE_TEXT, GENERIC_CODE_TEXT, PanelInput, build_decision_panel, describe_code,
    resolve_decision_time,
)

VARLIKLAR = {"Bitcoin (BTC)": "BTC-USD"}
NOW = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)


def _frame(processed_df, *, days_old=1):
    frame = processed_df.copy()
    frame["Open"] = frame["Open"].clip(lower=frame["Low"], upper=frame["High"])
    last = datetime.now(timezone.utc).date() - timedelta(days=days_old)
    frame.index = pd.date_range(end=pd.Timestamp(last), periods=len(frame), freq="D", tz="UTC")
    return frame


def _gunluk(at):
    """Karar paneli günlük periyotta görünsün."""
    for sb in at.selectbox:
        if sb.label == "Periyot:":
            sb.set_value("1d").run()
            return at
    raise AssertionError("Periyot seçimi bulunamadı")


def _panel(status="GECERLI", **kw):
    base = dict(signal=Signal.WAIT, position=Position.flat(), data_status=status, fee=None,
                spread_bps=None, slippage_bps=None, confidence=Decimal("50"), decision_at=NOW)
    base.update(kw)
    return build_decision_panel(PanelInput(**base))


# ---- Q12 ---------------------------------------------------------------
def test_q12_every_known_code_has_turkish_text_without_the_raw_code():
    for code, text in CODE_TEXT.items():
        assert text and "_" not in text, code


def test_q12_unknown_code_gets_safe_generic_text():
    assert describe_code("TAMAMEN_YENI_KOD") == GENERIC_CODE_TEXT
    assert "_" not in describe_code("TAMAMEN_YENI_KOD")


def test_q12_components_are_appended_by_name():
    assert describe_code("BILESEN_HAZIR_DEGIL", ("ml", "volume")) == (
        "Karar bileşeni hazır değil: yapay zekâ (ML), hacim")


def test_q12_every_panel_message_is_describable():
    from trading_contracts import PositionState
    panels = [
        _panel("ESKI_VERI", position=Position(PositionState.UNKNOWN)),
        _panel("YETERSIZ_GECMIS", position=Position(PositionState.OPEN, Decimal("1"), Decimal("10"),
                                                    NOW, Decimal("9"))),
        _panel(in_scope=False, asset_kind="XAU", missing_component="BILESEN_HAZIR_DEGIL"),
        _panel(asset_kind="GRAM_TRY", ambiguous_sequence=True, quantity_error="MIKTAR_ADIMI_BILINMIYOR"),
    ]
    for panel in panels:
        for message in panel.messages:
            assert message in CODE_TEXT, f"{message} için Türkçe metin yok"


# ---- Q11 ---------------------------------------------------------------
def test_q11_valid_status_records_and_returns_current_time():
    store = {}
    assert resolve_decision_time(store, "BTC-USD", "GECERLI", NOW) == NOW
    assert store["BTC-USD"] == NOW


def test_q11_invalid_status_returns_last_valid_time_not_now():
    store = {"BTC-USD": NOW}
    later = NOW + timedelta(hours=5)
    assert resolve_decision_time(store, "BTC-USD", "ESKI_VERI", later) == NOW
    assert store["BTC-USD"] == NOW


def test_q11_invalid_status_without_any_prior_valid_decision_is_none():
    assert resolve_decision_time({}, "BTC-USD", "VERI_YOK", NOW) is None


def test_q11_decisions_are_kept_per_symbol():
    store = {}
    resolve_decision_time(store, "BTC-USD", "GECERLI", NOW)
    assert resolve_decision_time(store, "ETH-USD", "ESKI_VERI", NOW) is None


def test_q11_screen_shows_last_valid_decision_time_with_stale_note(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    at = make_app(monkeypatch, _frame(processed_df)).run()
    assert not at.exception
    ilk = at.session_state["last_valid_decision"]["BTC-USD"]

    import data_fetchers
    monkeypatch.setattr(data_fetchers, "get_market_data",
                        lambda *a, **k: (_frame(processed_df, days_old=5), "Binance"))
    at.run()

    assert not at.exception
    assert at.session_state["last_valid_decision"]["BTC-USD"] == ilk
    metin = texts(at)
    assert "Son geçerli karar" in metin and "güncel değil" in metin


# ---- Q5 + Q12 (uygulama) ----------------------------------------------
def test_q5_stale_data_shows_cause_instead_of_valid_wait(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    at = _gunluk(make_app(monkeypatch, _frame(processed_df, days_old=5)).run())

    assert not at.exception
    metin = texts(at)
    assert "Veri güncel değil" in metin
    assert "İşlem sinyali yok. Piyasa izleniyor" not in metin
    gunluk = [m.value for m in at.sidebar.markdown if "at-sig" in m.value and "Günlük" in m.value]
    assert gunluk and "BEKLE" not in gunluk[0]


def test_q5_insufficient_history_is_reported_by_its_own_cause(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    at = _gunluk(make_app(monkeypatch, _frame(processed_df).iloc[-150:]).run())
    assert not at.exception
    assert "Karar için yeterli fiyat geçmişi yok" in texts(at)


def test_q12_screen_never_shows_raw_machine_codes(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    at = _gunluk(make_app(monkeypatch, _frame(processed_df, days_old=5)).run())
    metin = texts(at)
    for code in ("POZISYON_BILINMIYOR", "ESKI_KARAR", "ESKI_VERI", "GUNCEL_RISK_DEGERLENDIRILEMIYOR",
                 "KOMISYON_BILINMIYOR", "MAKAS_BILINMIYOR", "KAYMA_BILINMIYOR"):
        assert code not in metin, code
