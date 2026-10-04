"""Adım 5–7 panellerinin durumlarını çizen kanıt sayfası (gerçek arayüz kodu, sabit veri).

Kayıt deposu geçici bir klasöre yönlendirilir; hiçbir gerçek kayda dokunulmaz.
Yeniden üretim: `streamlit run docs/evidence/0005/adim5-7/harness.py`, ardından
`?state=regime|candidates|risk|protection|forward`.
"""
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
os.environ["ANALIZ_APP_DATA_DIR"] = os.path.join(tempfile.gettempdir(), "analiz_evidence_adim5_7")
os.makedirs(os.environ["ANALIZ_APP_DATA_DIR"], exist_ok=True)

import pandas as pd
import streamlit as st

import candidate_ui
import forward_tracker as ft
import forward_ui
import performance_ui as pui
import profit_protection as pp
import protection_ui
import risk_sizing as rs
import risk_ui
import storage
import strategy_candidates as sc
from performance_report import EquityPoint, ReportFilters, ReportTrade, build_report
from trade_execution import CostAssumptions

D = Decimal
st.set_page_config(layout="centered")
state = st.query_params.get("state", "regime")
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def captions(lines):
    for line in lines:
        (st.warning if line.startswith("Uyarı") else st.caption)(line)


if state == "regime":
    st.subheader("📑 Rapor — piyasa koşulu filtresi (R04, AC89)")
    d0 = date(2026, 1, 1)
    equity = [EquityPoint(d0 + timedelta(days=i), D(v)) for i, v in enumerate(["10000", "10100", "10060", "10560"])]
    trades = [ReportTrade("ETH-USD", "KRIPTO", "USD", D(p), d0 + timedelta(days=i), "V1", True, r)
              for i, (p, r) in enumerate([("100", "YUKSELEN"), ("-40", "YUKSELEN"), ("500", "DUSEN")], 1)]
    st.caption("Seçili: Piyasa koşulu = Yükselen")
    pui.render_report_view(pui.build_report_view(
        build_report({"ETH-USD": equity}, trades, ReportFilters(regime="YUKSELEN")), "YUKSELEN"))
elif state == "candidates":
    st.subheader("🧪 Aday stratejiler (AC01, AC16, AC17, AC48, AC85, AC96, AC98)")
    breakout, pullback = sc.default_candidates("KRIPTO", date(2025, 1, 1), date(2026, 1, 1))
    sc.preregister(breakout, NOW)
    sc.preregister(pullback, NOW)
    sc.record_result(breakout, "KRIPTO", sc.Verdict(sc.OLCUTU_KARSILADI), NOW)
    sc.record_result(pullback, "KRIPTO", sc.Verdict(
        sc.YETERSIZ_VERI, "Komisyon, makas ya da kayma bilinmiyor; sonuçlar üst sınırdır ve tercih "
                          "ölçütü doğrulanamaz. Maliyetleri girin."), NOW)
    sc.choose_active_strategy(breakout)
    frame = pd.DataFrame({"Close": [100 + i * 0.4 for i in range(260)]},
                         index=pd.date_range("2025-01-01", periods=260, freq="D", tz="UTC"))
    frame["ADX"] = 30.0
    captions(candidate_ui.build_candidate_view(frame).lines)
    st.button("Aktif yap: Kırılım (20 gün)")
    st.button("V1'e dön")
elif state == "risk":
    st.subheader("🛡️ Risk büyüklüğü ve kayıp sınırı (AC23, AC30, AC72, AC93, AC94)")
    rs.save_profile(per_trade_pct=D("1"), total_pct=D("5"))
    rs.set_loss_limit(D("500"), "USD", NOW - timedelta(days=9))
    rs.reset_loss_period(NOW - timedelta(days=5))
    rs.set_loss_limit(D("800"), "USD", NOW - timedelta(days=2))
    portfolio = {"balance": 4000.0, "positions": [
        {"Coin": "Ethereum (ETH)", "Giriş": 100.0, "Adet": 10.0, "Yatırım": 1000.0, "Stop": 90.0,
         "Status": "ACTIVE", "Gerçekleşme Zamanı": NOW.isoformat()}]}
    costs = CostAssumptions(None, D("10"), D("0.1"))
    captions(risk_ui.build_risk_view(portfolio, {"Ethereum (ETH)": "ETH-USD"}, entry=D("120"), stop=D("108"),
                                     quantity_step=D("0.01"), currency="USD",
                                     signals={"Ethereum (ETH)": "SAT"}, costs=costs).lines)
elif state == "protection":
    st.subheader("🛡️ Kâr koruma karşılaştırması (R13, AC95)")
    idx = pd.date_range("2026-09-01", periods=7, freq="D", tz="UTC")
    rows = [(99, 101, 98, 100), (100, 105, 99, 104), (104, 112, 103, 110), (110, 121, 109, 120),
            (120, 121, 108, 109), (109, 111, 105, 108), (108, 108, 100, 100)]
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(ATR=4.0)
    from technical_analysis import run_v1_strategy_backtest
    zero = CostAssumptions(D("0"), D("0"), D("0"))
    decisions = {idx[0]: "AL", idx[5]: "SAT"}
    backtest = run_v1_strategy_backtest(frame, decisions, initial_cash=D("10000"), trade_notional=D("1000"),
                                        quantity_step=D("1"), costs=zero)
    captions(protection_ui.build_protection_view(pp.compare(frame, decisions, backtest, zero, D("1")),
                                                  "USD", zero).lines)
elif state == "forward":
    st.subheader("📡 İleri dönem sanal takip (AC35, AC36, AC38, AC71, AC99, AC103)")
    base = date(2026, 9, 20)
    assumptions = {"dolum": "ertesi gün açılışı", "komisyon": "0.1", "makas": "bilinmiyor", "kayma": "bilinmiyor",
                   "miktar adımı": "0.001"}
    for offset in (0, 1, 2, 5):
        day = base + timedelta(days=offset)
        late = offset == 5
        evaluated = ft.available_at("KRIPTO", day) + (timedelta(days=1) if late else timedelta(minutes=17))
        ft.record(ft.ForwardDecision(
            asset="LINKUSD", strategy_version="V1~3fa9c1d2", candle_day=day,
            decision="AL" if offset == 1 else "BEKLE", source="Binance", evaluated_at=evaluated,
            candle={"close": "18.50"}, assumptions=assumptions, on_time=ft.is_on_time("KRIPTO", day, evaluated),
            real_clock=True, regime="YUKSELEN", regime_version="REJIM-1"))
    ft.record(ft.ForwardDecision(
        asset="LINKUSD", strategy_version="V1~3fa9c1d2", candle_day=base + timedelta(days=1),
        decision="AL", source="Binance", evaluated_at=NOW, candle={"close": "19.10"}))
    ft.record_run(NOW - timedelta(hours=11), True, "tamam")
    ft.record_run(NOW - timedelta(hours=1), False, "HBARUSD: veri alınamadı (Binance); LINKUSD: miktar adımı/varsayımlar eksik; sanal işlem hesaplanmadı.")
    captions(forward_ui.build_forward_view(NOW).lines)
