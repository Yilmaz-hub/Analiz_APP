"""Spec 0005 Rev 6 — B13 bağlama: aktif aday canlı karar panelini sürer (AC104), AC77."""
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

import candidate_ui
import strategy_candidates as sc
from regime_classifier import DUSEN, YUKSELEN
from trading_contracts import Signal

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
BREAKOUT, PULLBACK = sc.default_candidates("KRIPTO", None, None)


def _rising(n=40):
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100, 140, n)           # her gün yeni zirve: kırılım koşulu son gün sağlanır
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close}, index=index)


def _select(candidate):
    sc.preregister(candidate, NOW)
    sc.choose_active_strategy(candidate)


def _label(monkeypatch, regime):
    monkeypatch.setattr(candidate_ui, "classify_latest", lambda frame: (regime, "REJIM-1"))


def test_ac104_v1_active_keeps_the_live_signal_unchanged():
    """AC104 — Seçim yapılmadıkça (aktif strateji V1) canlı sinyal olduğu gibi kalır."""
    for signal in (Signal.WAIT, Signal.BUY, Signal.SELL):
        assert candidate_ui.active_entry_signal(_rising(), signal) == (signal, None)


def test_ac104_selected_candidate_turns_wait_into_buy_in_a_rising_market(monkeypatch):
    """AC104 — Seçilen kırılım adayı, yükselen piyasada ve yeni zirvede V1 BEKLE derken giriş (AL) üretir."""
    _label(monkeypatch, YUKSELEN)
    _select(BREAKOUT)
    signal, note = candidate_ui.active_entry_signal(_rising(), Signal.WAIT)
    assert signal is Signal.BUY
    assert note == "Aktif strateji: Kırılım (20 gün) — giriş sinyali bu aday kuralına göre; çıkış V1 kurallarıyla."


def test_ac104_candidate_gives_no_entry_when_its_rule_is_not_met(monkeypatch):
    """AC104 — Aday kuralı sağlanmıyorsa (düşen piyasa) V1 AL dese bile giriş verilmez; çıkış (SAT) korunur."""
    _label(monkeypatch, DUSEN)
    _select(BREAKOUT)
    assert candidate_ui.active_entry_signal(_rising(), Signal.BUY)[0] is Signal.WAIT
    assert candidate_ui.active_entry_signal(_rising(), Signal.SELL)[0] is Signal.SELL


def test_ac104_missing_candidate_record_falls_back_to_v1_visibly():
    """AC104 — Aktif adayın kaydı bulunamazsa V1 kullanılır ve bu açıkça belirtilir."""
    sc.choose_active_strategy(BREAKOUT)  # ön kayıt yok
    signal, note = candidate_ui.active_entry_signal(_rising(), Signal.BUY)
    assert signal is Signal.BUY and "kaydı bulunamadı" in note and "V1" in note


def test_ac77_open_position_keeps_entry_version_and_says_so():
    """AC77 — Aktif sürüm değişince açık gerçek pozisyon giriş anındaki sürümle yönetilir ve ekranda belirtilir."""
    _select(BREAKOUT)
    active = sc.active_strategy()
    position = {"Coin": "BTC", "Status": "ACTIVE", "StrategyVersion": "V1"}
    assert candidate_ui.management_note(position, active) == (
        "Bu pozisyon giriş anındaki strateji sürümüyle (V1) yönetilir; aktif sürüm farklı olsa da "
        "çıkış kuralları değişmez.")
    assert candidate_ui.management_note(position, "V1") is None
    legacy = {"Coin": "ETH", "Status": "ACTIVE"}        # sürüm yazılmamış eski kayıt: V1
    assert "(V1)" in candidate_ui.management_note(legacy, active)
    same = {"Coin": "SOL", "Status": "ACTIVE", "StrategyVersion": active}
    assert candidate_ui.management_note(same, active) is None


def test_ac104_decision_panel_shows_the_active_candidate_note(store, monkeypatch, processed_df):
    """AC104 — Karar panelinde aktif aday notu görünür; ekran çökmez."""
    from app_helpers import make_app, texts

    _select(BREAKOUT)
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    assert not at.exception
    assert "Aktif strateji: Kırılım (20 gün) — giriş sinyali bu aday kuralına göre" in texts(at)
