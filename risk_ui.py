"""Risk paneli (spec 0005, Adım 6 / S4).

Hesap `risk_sizing.py`'dedir; burası portföy kaydından açık riskleri ve dönem
sonuçlarını okur (yazmaz) ve gösterim satırlarını üretir. Engel varken miktar
gösterilmez; açık pozisyonların stop ve son uyarı bilgisi her durumda görünür (R11).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

import market_map
import risk_sizing as rs

_UNKNOWN = "BILINMIYOR"


@dataclass(frozen=True)
class RiskView:
    lines: list[str]


def _currency(coin: str, coin_map: Mapping[str, str]) -> str:
    info = market_map.market_of(coin_map.get(coin, ""))
    return _UNKNOWN if info is None else info[1]


def _number(value) -> Decimal:
    return Decimal(str(value))


def _active(portfolio: Mapping) -> list[Mapping]:
    from positions import AKTIF, group_positions
    return group_positions(portfolio.get("positions", []))[AKTIF]


def open_risks(portfolio: Mapping, coin_map: Mapping[str, str]) -> list[tuple[str, Decimal]]:
    return [(_currency(p["Coin"], coin_map),
             rs.position_risk(_number(p["Giriş"]), _number(p["Stop"]), _number(p["Adet"])))
            for p in _active(portfolio)]


def period_results(portfolio: Mapping, coin_map: Mapping[str, str],
                   started_at: datetime) -> list[tuple[str, Decimal]]:
    results = []
    for position in portfolio.get("positions", []):
        exits = position.get("Çıkışlar")
        if exits and position.get("Status") in ("ACTIVE", "CLOSED_CONFIRMED"):
            # Kısmi satışlı kayıt: her parçanın sonucu kendi zamanında sayılır (spec 0006 R04).
            entry = _number(position.get("Giriş", 0))
            for exit_ in exits:
                if datetime.fromisoformat(exit_["executed_at"]) >= started_at:
                    results.append((_currency(position["Coin"], coin_map),
                                    (_number(exit_["price"]) - entry) * _number(exit_["quantity"])))
            continue
        closed_at = position.get("Çıkış Zamanı")
        if position.get("Status") != "CLOSED_CONFIRMED" or not closed_at:
            continue
        if datetime.fromisoformat(closed_at) >= started_at:
            results.append((_currency(position["Coin"], coin_map), _number(position.get("Realized", 0))))
    return results


def portfolio_capital(portfolio: Mapping) -> tuple[Decimal, Decimal]:
    """`(sermaye, nakit)`: sermaye = nakit + açık pozisyonların maliyeti; kayıt yoksa 0 (AC93)."""
    cash = _number(portfolio.get("balance", 0) or 0)
    invested = sum((_number(p.get("Yatırım", 0) or 0) for p in _active(portfolio)), Decimal("0"))
    return cash + invested, cash


def unit_cost(entry: Decimal, costs) -> Decimal:
    """Birim başına **bilinen** giriş maliyeti: komisyon (%) + makas/2 + kayma (baz puan).

    Bilinmeyen kalem sıfır sayılmaz, hesaba katılmaz ve ekranda açıkça belirtilir."""
    if costs is None:
        return Decimal("0")
    total = Decimal("0")
    if costs.commission_pct is not None:
        total += entry * Decimal(costs.commission_pct) / Decimal("100")
    if costs.spread_bps is not None:
        total += entry * Decimal(costs.spread_bps) / Decimal("2") / Decimal("10000")
    if costs.slippage_bps is not None:
        total += entry * Decimal(costs.slippage_bps) / Decimal("10000")
    return total


def _unknown_cost_names(costs) -> list[str]:
    if costs is None:
        return ["komisyon", "makas", "kayma"]
    return [name for name, value in (("komisyon", costs.commission_pct), ("makas", costs.spread_bps),
                                     ("kayma", costs.slippage_bps)) if value is None]


def build_risk_view(portfolio: Mapping, coin_map: Mapping[str, str], *,
                    entry: Decimal, stop: Decimal | None, quantity_step: Decimal | None,
                    currency: str, signals: Mapping[str, str], costs=None,
                    minimum_notional: Decimal | None = None) -> RiskView:
    lines: list[str] = []
    blocked = False
    profile = rs.load_profile()
    capital, cash = portfolio_capital(portfolio)

    period = rs.loss_period()
    if period is not None:
        lines.append(f"Kayıp dönemi başlangıcı: {period['started_at']} · sınır {period['limit']} "
                     f"{period['currency']}")
        for item in period["history"]:
            if item["action"] == "SIFIRLAMA":
                lines.append(f"Sıfırlama kaydı: {item['closed_period_start'][:10]} başlayan dönem "
                             f"{item['at']} tarihinde kullanıcı tarafından kapatıldı.")
            else:
                lines.append(f"Sınır değişikliği kaydı: {item['old']} → {item['new']} {item['currency']} "
                             f"({item['at']}); dönem sıfırlanmadı.")
        gate = rs.loss_gate(period_results(portfolio, coin_map,
                                           datetime.fromisoformat(period["started_at"])))
        if not gate.allowed:
            blocked = True
            lines.append(gate.reason)

    if profile is None:
        lines.append(rs.suggest(capital=capital, entry=entry, stop=stop or Decimal("0"),
                                quantity_step=quantity_step, cash=cash).reason)
        blocked = True
    elif not blocked:
        if stop is None:
            lines.append("Başlangıç stopu hesaplanamadı; miktar önerilmez.")
            blocked = True
        else:
            suggestion = rs.suggest(capital=capital, entry=entry, stop=stop,
                                    quantity_step=quantity_step, cash=cash,
                                    unit_cost=unit_cost(entry, costs),
                                    minimum_notional=minimum_notional)
            limit = capital * profile.total_pct / Decimal("100")
            new_risk = (rs.position_risk(entry, stop, suggestion.quantity)
                        if suggestion.quantity is not None else Decimal("0"))
            total = rs.total_risk_check(limit, open_risks(portfolio, coin_map), new_risk, currency)
            if suggestion.quantity is None:
                lines.append(suggestion.reason)
            elif not total.allowed:
                lines.append(total.reason)
            else:
                lines.append(f"Önerilen miktar: {suggestion.quantity} (giriş {entry}, stop {stop}, "
                             f"işlem başına risk %{profile.per_trade_pct})")
                lines.append("Stop bütçesi, fiyat boşluklarında azami kayıp garantisi değildir.")
                unknown = _unknown_cost_names(costs)
                if unknown:
                    lines.append("Bilinmeyen maliyet (" + ", ".join(unknown) + ") miktar hesabına "
                                 "girmedi; gerçek risk daha yüksek olabilir.")

    for position in _active(portfolio):
        signal = signals.get(position["Coin"], "—")
        lines.append(f"{position['Coin']}: stop {_number(position['Stop']).normalize():f} · "
                     f"son uyarı {signal}")
    return RiskView(lines)


def render_risk_panel(source: Mapping, portfolio: Mapping, coin_map: Mapping[str, str]) -> None:
    import streamlit as st

    from storage import StorageAccessError

    try:
        _render_risk_panel(st, source, portfolio, coin_map)
    except StorageAccessError:
        st.warning("Risk kayıtları okunamadı; kayıt deposu erişimini kontrol edin.")


def _render_risk_panel(st, source: Mapping, portfolio: Mapping, coin_map: Mapping[str, str]) -> None:
    from datetime import timezone

    from trade_decisions import initial_stop

    st.markdown("**🛡️ Risk Büyüklüğü ve Kayıp Sınırı**")
    profile = rs.load_profile()
    first, second, third = st.columns(3)
    per_trade = first.text_input("İşlem başına risk (%)",
                                 value="" if profile is None else str(profile.per_trade_pct),
                                 key="risk_per_trade")
    total = second.text_input("Toplam açık risk (%)",
                              value="" if profile is None else str(profile.total_pct),
                              key="risk_total")
    if third.button("Risk profilini kaydet", key="risk_save"):
        a, reason_a = rs.parse_ratio(per_trade)
        b, reason_b = rs.parse_ratio(total)
        if a is None or b is None:
            st.warning(reason_a or reason_b)
        else:
            rs.save_profile(per_trade_pct=a, total_pct=b)
    info = market_map.market_of(source["symbol"])
    currency = info[1] if info else _UNKNOWN
    limit_col, limit_btn = st.columns(2)
    loss_limit = limit_col.text_input(f"Dönem kayıp sınırı ({currency})", key="risk_loss_limit")
    if limit_btn.button("Kayıp sınırını kaydet", key="risk_loss_save"):
        try:
            amount = Decimal(loss_limit.strip().replace(",", "."))
        except Exception:
            amount = None
        if amount is None or not amount.is_finite() or amount <= 0:
            st.warning("Kayıp sınırı 0'dan büyük bir tutar olmalı.")
        else:
            rs.set_loss_limit(amount, currency, datetime.now(timezone.utc))
    asset = source.get("asset") or source["symbol"]
    min_col, min_btn = st.columns(2)
    saved_min = rs.load_min_notional(asset)
    min_text = min_col.text_input(f"Asgari işlem tutarı ({currency}, boş = yok)",
                                  value="" if saved_min is None else str(saved_min),
                                  key=f"risk_min:{source['symbol']}")
    if min_btn.button("Asgari tutarı kaydet", key=f"risk_min_save:{source['symbol']}"):
        amount, reason = rs.parse_min_notional(min_text)
        if reason:
            st.warning(reason)
        else:
            rs.save_min_notional(asset, amount)
            saved_min = amount
    frame = source["frame"]
    entry = Decimal(str(frame["Close"].iloc[-1]))
    atr = frame["ATR"].iloc[-1] if "ATR" in frame.columns else None
    stop = None if atr is None or atr != atr else initial_stop(entry, Decimal(str(atr)))
    view = build_risk_view(
        portfolio, coin_map, entry=entry, stop=stop, quantity_step=source["quantity_step"],
        currency=currency, signals=source.get("signals", {}), costs=source["costs"],
        minimum_notional=saved_min)
    for line in view.lines:
        st.caption(line)
    if rs.loss_period() is not None and st.button("Kayıp dönemini sıfırla (yeni dönem başlat)",
                                                  key="risk_reset"):
        rs.reset_loss_period(datetime.now(timezone.utc))
