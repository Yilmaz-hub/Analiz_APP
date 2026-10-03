"""Spec 0005 Adım 6 / S4 — risk bazlı miktar, toplam risk ve kayıp sınırı (R10–R12, Q04)."""
from datetime import datetime, timezone
from decimal import Decimal

import pytest

import risk_sizing as rs

D = Decimal
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _size(budget="100", entry="100", stop="90", unit_cost="0", step="1", cash="100000", **kw):
    return rs.size(D(budget), D(entry), D(stop), unit_cost=D(unit_cost),
                   quantity_step=None if step is None else D(step), cash=D(cash), **kw)


def test_ac24_risk_quantity_is_budget_over_distance():
    """AC24 — Bütçe 100, giriş 100, stop 90, maliyet 0, adım 1 → miktar 10."""
    assert _size().quantity == D("10")


def test_ac82_known_unit_cost_reduces_quantity():
    """AC82 — Bütçe 100, giriş 100, stop 90, birim maliyet 1, adım 1 → miktar 9 (aşağı yuvarlanır)."""
    assert _size(unit_cost="1").quantity == D("9")


def test_ac25_stop_equal_to_entry_is_rejected():
    """AC25 — Girişe eşit stop ile miktar istenirse reddedilir ve nedeni gösterilir."""
    result = _size(stop="100")
    assert result.quantity is None and "stop" in result.reason.lower()


def test_ac67_stop_above_entry_is_rejected():
    """AC67 — Girişten yüksek stop reddedilir ve nedeni gösterilir."""
    result = _size(stop="101")
    assert result.quantity is None and "stop" in result.reason.lower()


@pytest.mark.parametrize("stop", ["0", "-5"])
def test_ac68_zero_or_negative_stop_is_rejected(stop):
    """AC68 — Stop 0 veya negatif ise hesap reddedilir ve nedeni gösterilir."""
    result = _size(stop=stop)
    assert result.quantity is None and "stop" in result.reason.lower()


def test_ac31_unknown_quantity_step_gives_no_quantity():
    """AC31 — Miktar adımı bilinmeyen varlıkta miktar hiç önerilmez, neden gösterilir."""
    result = _size(step=None)
    assert result.quantity is None and "miktar adımı" in result.reason.lower()


def test_ac83_below_step_or_minimum_notional_gives_no_quantity():
    """AC83 — Hesaplanan miktar adımın veya asgari işlem tutarının altındaysa miktar üretilmez."""
    below_step = _size(budget="5", step="1")  # 0,5 birim → adım altı
    assert below_step.quantity is None and below_step.reason
    below_notional = _size(minimum_notional=D("2000"))  # 10 × 100 = 1000 < 2000
    assert below_notional.quantity is None and "asgari" in below_notional.reason.lower()


def test_ac24_cash_caps_quantity_never_rounds_up():
    """AC24 — Nakit yetmiyorsa miktar nakitle sınırlanır; yukarı yuvarlama yoktur."""
    assert _size(cash="550").quantity == D("5")


def test_ac23_undefined_profile_gives_no_quantity():
    """AC23 — Risk profili belirlenmemişse miktar hiç önerilmez ve eksik bilgi açıklanır."""
    result = rs.suggest(capital=D("10000"), entry=D("100"), stop=D("90"),
                        quantity_step=D("1"), cash=D("10000"))
    assert result.quantity is None
    assert "risk profili" in result.reason.lower()


@pytest.mark.parametrize("value", ["0", "-1", "100", "150"])
def test_ac26_invalid_ratio_is_rejected(value):
    """AC26 — Risk oranı 0, negatif, %100 veya üstü ise ayrı ayrı reddedilir."""
    ratio, reason = rs.parse_ratio(value)
    assert ratio is None and reason


def test_ac58_ratio_rounding_to_zero_on_percent_scale_is_rejected():
    """AC58 — Yüzde ölçeğinde sıfıra yuvarlanan oran (%0,001) reddedilir."""
    ratio, reason = rs.parse_ratio("0.001")
    assert ratio is None and "ölçek" in reason.lower()
    assert rs.parse_ratio("0.01")[0] == D("0.01")


def test_ac81_saved_limits_persist_and_drive_next_quantity():
    """AC81 — Kaydedilen işlem başı ve toplam risk sınırı sonraki miktar hesabında kullanılır."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("3"))
    assert rs.load_profile() == rs.RiskProfile(D("1"), D("3"))
    result = rs.suggest(capital=D("10000"), entry=D("100"), stop=D("90"),
                        quantity_step=D("1"), cash=D("10000"))
    assert result.quantity == D("10")  # %1 × 10.000 = 100 bütçe


def test_ac27_total_risk_at_limit_is_allowed():
    """AC27 — Limit 300, mevcut 200, yeni 100 → toplam risk engeli yok (sınır dahil)."""
    check = rs.total_risk_check(D("300"), [("USD", D("200"))], D("100"), "USD")
    assert check.allowed and check.verified


def test_ac28_total_risk_over_limit_blocks():
    """AC28 — Limit 300, mevcut 200, yeni 100,01 → yeni giriş engellenir."""
    check = rs.total_risk_check(D("300"), [("USD", D("200"))], D("100.01"), "USD")
    assert not check.allowed and check.reason


def test_ac75_stop_above_entry_counts_as_zero_risk():
    """AC75 — Stopu girişin üstüne taşınmış pozisyonun riski sıfırdır, bütçeyi artırmaz."""
    assert rs.position_risk(D("100"), D("110"), D("5")) == D("0")
    check = rs.total_risk_check(D("100"), [("USD", rs.position_risk(D("100"), D("110"), D("5")))],
                                D("100"), "USD")
    assert check.allowed


def test_ac73_mixed_currency_total_risk_is_not_verified():
    """AC73 — TL ve USD karışık açık riskte toplam risk doğrulanmış gösterilmez, neden açıklanır."""
    check = rs.total_risk_check(D("300"), [("USD", D("50")), ("TRY", D("50"))], D("10"), "USD")
    assert not check.verified and not check.allowed
    assert "para birimi" in check.reason.lower()


def test_ac29_loss_equal_to_limit_blocks_new_entry():
    """AC29 — Dönem kaybı sınıra tam eşitse yeni giriş engellenir."""
    rs.set_loss_limit(D("500"), "USD", NOW)
    gate = rs.loss_gate([("USD", D("-300")), ("USD", D("-200"))])
    assert not gate.allowed and gate.verified


def test_ac29_loss_below_limit_allows_entry():
    """AC29 — Dönem kaybı sınırın altındaysa giriş engellenmez."""
    rs.set_loss_limit(D("500"), "USD", NOW)
    assert rs.loss_gate([("USD", D("-499.99")), ("USD", D("100"))]).allowed


def test_ac74_mixed_currency_losses_are_not_verified():
    """AC74 — Farklı para birimlerinden oluşan dönem kaybıyla sınır doğrulanmış hesaplanmaz."""
    rs.set_loss_limit(D("500"), "USD", NOW)
    gate = rs.loss_gate([("USD", D("-100")), ("TRY", D("-100"))])
    assert not gate.verified and "para birimi" in gate.reason.lower()


def test_ac72_limit_resets_only_by_explicit_recorded_action():
    """AC72 — Sınır yalnız açık kullanıcı eylemiyle sıfırlanır; eylem kaydı ve yeni dönem görünür."""
    rs.set_loss_limit(D("500"), "USD", NOW)
    started = rs.loss_period()["started_at"]
    assert rs.loss_period()["started_at"] == started  # kendiliğinden değişmez
    later = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)
    rs.reset_loss_period(later)
    period = rs.loss_period()
    assert period["started_at"] == later.isoformat()
    assert period["history"] == [{"action": "SIFIRLAMA", "at": later.isoformat(),
                                  "closed_period_start": started}]
