"""Spec 0005 Rev 5 — QA bulguları: risk paneli (B4, B5)."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import risk_sizing as rs
import risk_ui
from trade_execution import CostAssumptions

D = Decimal
T1 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
COINS = {"Ethereum (ETH)": "ETH-USD"}


def _portfolio(balance, *positions):
    return {"balance": balance, "positions": list(positions)}


def _open(invested=1000.0):
    return {"Coin": "Ethereum (ETH)", "Giriş": 100.0, "Adet": invested / 100, "Yatırım": invested,
            "Stop": 90.0, "Status": "ACTIVE", "Gerçekleşme Zamanı": T1.isoformat()}


def _text(portfolio, **kw):
    params = dict(entry=D("100"), stop=D("90"), quantity_step=D("1"), currency="USD", signals={}, costs=None)
    params.update(kw)
    return " | ".join(risk_ui.build_risk_view(portfolio, COINS, **params).lines)


def test_ac93_quantity_uses_real_cash_not_a_default_capital():
    """AC93 — Bakiye 5 iken risk bazlı miktar önerilmez (gerçek nakit yetmez); 10.000 varsayılan sermaye kullanılmaz."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("3"))
    text = _text(_portfolio(5.0))
    assert "Önerilen miktar" not in text and "miktar adımının altında" in text


def test_ac93_capital_is_cash_plus_open_positions_and_cash_caps_quantity():
    """AC93 — Sermaye = nakit + açık pozisyonların maliyeti; miktar bütçeden değil nakit azsa nakitten çıkar."""
    rs.save_profile(per_trade_pct=D("50"), total_pct=D("90"))
    portfolio = _portfolio(300.0, _open(1000.0))     # sermaye 1.300, bütçe %50 = 650, nakit 300
    assert risk_ui.portfolio_capital(portfolio) == (D("1300"), D("300"))
    assert "Önerilen miktar: 3 " in _text(portfolio)  # min(650 / 10 = 65, 300 / 100 = 3)


def test_ac93_missing_portfolio_cash_gives_no_quantity():
    """AC93 — Nakit ya da sermaye kaydı yoksa (sıfır) miktar önerilmez ve eksik bilgi açıklanır."""
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("3"))
    text = _text(_portfolio(0.0))
    assert "Önerilen miktar" not in text and "eksik" in text.lower()


def test_ac93_known_unit_cost_enters_the_quantity_formula():
    """AC93 — Bilinen birim maliyet (komisyon %1, makas 20 bps, kayma 10 bps) miktar formülüne girer."""
    costs = CostAssumptions(D("20"), D("10"), D("1"))
    assert risk_ui.unit_cost(D("100"), costs) == D("1.20")
    rs.save_profile(per_trade_pct=D("2"), total_pct=D("10"))
    portfolio = _portfolio(5000.0)                  # bütçe %2 = 100
    assert "Önerilen miktar: 10 " in _text(portfolio)                    # 100 / 10
    assert "Önerilen miktar: 8 " in _text(portfolio, costs=costs)        # 100 / 11,2 = 8,9 → 8


def test_ac93_unknown_costs_are_disclosed_not_treated_as_zero():
    """AC93 — Bilinmeyen maliyet kalemi miktar hesabına girmez ama kullanıcıya açıkça söylenir."""
    rs.save_profile(per_trade_pct=D("2"), total_pct=D("10"))
    text = _text(_portfolio(5000.0), costs=CostAssumptions(None, None, D("1")))
    assert "Bilinmeyen maliyet" in text and "makas" in text and "kayma" in text


def test_ac94_saving_the_same_limit_keeps_the_period_and_reset_records():
    """AC94 — Açık dönem varken aynı sınırı yeniden kaydetmek dönemi ve sıfırlama kayıtlarını silmez."""
    rs.set_loss_limit(D("500"), "USD", T1)
    rs.reset_loss_period(T1 + timedelta(days=1))
    before = rs.loss_period()
    rs.set_loss_limit(D("500"), "USD", T1 + timedelta(days=5))
    assert rs.loss_period() == before


def test_ac94_changing_the_limit_is_recorded_without_restarting_the_period():
    """AC94 — Sınır değeri değişince dönem başlangıcı korunur ve değişiklik ayrı kayıt olarak eklenir."""
    rs.set_loss_limit(D("500"), "USD", T1)
    rs.reset_loss_period(T1 + timedelta(days=1))
    started = rs.loss_period()["started_at"]
    rs.set_loss_limit(D("800"), "USD", T1 + timedelta(days=2))
    period = rs.loss_period()
    assert period["started_at"] == started and period["limit"] == "800"
    assert [item["action"] for item in period["history"]] == ["SIFIRLAMA", "SINIR_DEGISTI"]
    assert (period["history"][-1]["old"], period["history"][-1]["new"]) == ("500", "800")


def test_ac94_a_full_limit_cannot_be_bypassed_by_saving_it_again():
    """AC94 — Dolu sınır, aynı değeri yeniden kaydederek aşılamaz; giriş engeli sürer."""
    rs.set_loss_limit(D("500"), "USD", T1)
    losses = [("USD", D("-500"))]
    assert not rs.loss_gate(losses).allowed
    rs.set_loss_limit(D("500"), "USD", T1 + timedelta(hours=1))
    assert not rs.loss_gate(losses).allowed


def test_ac94_panel_save_button_does_not_restart_the_period(store, monkeypatch, processed_df):
    """AC94 — Ekranda "Kayıp sınırını kaydet" iki kez basıldığında dönem başlangıcı değişmez."""
    import technical_analysis
    from app_helpers import click, make_app

    monkeypatch.setattr(technical_analysis, "build_v1_decisions", lambda df, **k: {df.index[199]: "AL"})
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("0.001")
    at = click(at.run(), "🚀 Backtest Başlat")
    at.text_input(key="risk_loss_limit").set_value("500")
    at = click(at, "Kayıp sınırını kaydet")
    first = rs.loss_period()["started_at"]
    at = click(at, "Kayıp sınırını kaydet")
    assert rs.loss_period()["started_at"] == first
