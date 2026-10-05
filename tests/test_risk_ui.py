"""Spec 0005 Adım 6 — risk paneli görünürlüğü (AC23, AC30, AC72, AC73, AC74, AC81)."""
from datetime import datetime, timezone
from decimal import Decimal

import risk_sizing as rs
import risk_ui

D = Decimal
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
COINS = {"Ethereum (ETH)": "ETH-USD", "Türk Hava Yolları": "THYAO.IS"}


def _portfolio(*positions):
    return {"balance": 10000.0, "positions": list(positions)}


def _open(coin="Ethereum (ETH)", entry=100.0, qty=10.0, stop=90.0):
    return {"Coin": coin, "Giriş": entry, "Adet": qty, "Yatırım": entry * qty, "Stop": stop, "Status": "ACTIVE",
            "Gerçekleşme Zamanı": NOW.isoformat()}


def _closed(coin="Ethereum (ETH)", realized=-500.0, at=NOW):
    return {"Coin": coin, "Giriş": 100.0, "Adet": 0.0, "Stop": 90.0, "Realized": realized,
            "Status": "CLOSED_CONFIRMED", "Çıkış Zamanı": at.isoformat()}


def _view(portfolio, **kw):
    params = dict(entry=D("100"), stop=D("90"), quantity_step=D("1"), currency="USD", signals={})
    params.update(kw)
    return " | ".join(risk_ui.build_risk_view(portfolio, COINS, **params).lines)


def test_ac23_panel_without_profile_shows_reason_and_no_quantity():
    """AC23 — Profil yokken panel miktar göstermez (tahmin de yok) ve eksik bilgiyi açıklar."""
    text = _view(_portfolio())
    assert "Risk profili belirlenmedi" in text
    assert "Önerilen miktar" not in text


def test_ac81_panel_uses_saved_profile_for_quantity():
    """AC81 — Kaydedilmiş profil panelde miktar hesabında kullanılır."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("3"))
    assert "Önerilen miktar: 10" in _view(_portfolio())


def test_ac30_loss_limit_block_keeps_open_position_stop_and_sell_visible():
    """AC30 — Kayıp sınırı girişleri engellerken açık pozisyonun SAT ve stop bilgileri görünür kalır."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("10"))
    rs.set_loss_limit(D("500"), "USD", datetime(2026, 10, 1, tzinfo=timezone.utc))
    portfolio = _portfolio(_open(), _closed(realized=-500.0))
    text = _view(portfolio, signals={"Ethereum (ETH)": "SAT"})
    assert "yeni giriş önerilmez" in text.lower()
    assert "Önerilen miktar" not in text
    assert "Ethereum (ETH): stop 90 · son uyarı SAT" in text


def test_ac72_panel_shows_reset_record_and_new_period_start():
    """AC72 — Sıfırlama eylem kaydı ve yeni dönem başlangıcı panelde görünür."""
    rs.set_loss_limit(D("500"), "USD", datetime(2026, 10, 1, tzinfo=timezone.utc))
    rs.reset_loss_period(NOW)
    text = _view(_portfolio())
    assert f"Kayıp dönemi başlangıcı: {NOW.isoformat()}" in text
    assert "Sıfırlama kaydı: 2026-10-01" in text


def test_ac73_mixed_currency_open_risk_is_explained_in_panel():
    """AC73 — TL ve USD karışık açık risk panelde doğrulanmamış olarak açıklanır."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("10"))
    text = _view(_portfolio(_open(), _open(coin="Türk Hava Yolları")))
    assert "birden fazla para birimi" in text
    assert "Önerilen miktar" not in text


def test_ac74_mixed_currency_losses_are_explained_in_panel():
    """AC74 — Farklı para birimli dönem kaybında kayıp sınırı doğrulanmadı diye açıklanır."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("10"))
    rs.set_loss_limit(D("500"), "USD", datetime(2026, 10, 1, tzinfo=timezone.utc))
    text = _view(_portfolio(_closed(realized=-10.0), _closed(coin="Türk Hava Yolları", realized=-10.0)))
    assert "kayıp sınırı doğrulanamadı" in text
