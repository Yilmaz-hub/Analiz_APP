"""Adım 4 rapor panelinin durumlarını çizen kanıt sayfası (gerçek performance_ui çizimi)."""
import os
import sys
from datetime import date, timedelta
from decimal import Decimal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import streamlit as st
import performance_ui as ui
from performance_report import EquityPoint, ReportFilters, ReportTrade, build_report

st.set_page_config(layout="centered")
d0 = date(2026, 1, 1)
eq = lambda vals: [EquityPoint(d0 + timedelta(days=i), Decimal(v)) for i, v in enumerate(vals)]
tr = lambda pnl, known=True, sym="ETH-USD", cur="USD", mk="KRIPTO": ReportTrade(
    sym, mk, cur, Decimal(pnl), d0 + timedelta(days=1), "V1", known)
state = st.query_params.get("state", "missing_cost")

if state == "missing_cost":
    st.subheader("📑 Güvenilir Performans Raporu — maliyeti eksik işlem (AC07, AC84)")
    ui.render_report_view(ui.build_report_view(build_report(
        {"ETH-USD": eq(["10000", "12000", "9000", "11000"])},
        [tr("100"), tr("-40", False)], ReportFilters())))
elif state == "empty":
    st.subheader("📑 Boş dönem (AC02, AC78)")
    ui.render_report_view(ui.build_report_view(build_report(
        {"ETH-USD": eq(["10000", "10100"])}, [],
        ReportFilters(start=d0 + timedelta(days=50), end=d0 + timedelta(days=60)))))
    st.subheader("Sıfır sermaye (AC49)")
    ui.render_report_view(ui.build_report_view(build_report({"ETH-USD": eq(["0", "500"])}, [], ReportFilters())))
elif state == "error":
    st.subheader("📑 Geçersiz istek (AC09, AC64)")
    ui.render_report_view(ui.build_report_view(build_report(
        {"ETH-USD": eq(["10000", "10100"])}, [],
        ReportFilters(start=d0 + timedelta(days=3), end=d0))))
    ui.render_report_view(ui.build_report_view(build_report(
        {"ETH-USD": eq(["10000", "10100"])}, [], ReportFilters(market="MARS"))))
elif state == "currencies":
    st.subheader("📑 Para birimleri ayrı (AC10)")
    ui.render_report_view(ui.build_report_view(build_report(
        {"ETH-USD": eq(["10000", "10100"]), "THYAO.IS": eq(["10000", "10100"])},
        [tr("100"), tr("100", sym="THYAO.IS", cur="TRY", mk="BIST")], ReportFilters())))
if state == "comparison":
    import pandas as pd
    from technical_analysis import run_v1_strategy_backtest
    from trade_execution import CostAssumptions
    zero = Decimal("0")
    idx = pd.date_range("2026-09-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame({"Open": [100, 100, 110, 120], "High": [101, 200, 115, 125],
                          "Low": [98, 95, 105, 118], "Close": [100, 110, 112, 120], "ATR": [4, 5, 5, 5]}, index=idx)
    costs = CostAssumptions(zero, zero, Decimal("0.1"))
    res = run_v1_strategy_backtest(frame, {idx[0]: "AL", idx[1]: "BEKLE", idx[2]: "SAT"},
                                   initial_cash=Decimal("2000"), trade_notional=Decimal("1000"),
                                   quantity_step=Decimal("1"), costs=costs)
    st.subheader("Strateji ve Al-Tut (AC47, AC62)")
    ui.render_report_view(ui.build_report_view(ui.report_from_backtest("ETH-USD", res)))
    ui.render_report_view(ui.comparison_from_backtest("ETH-USD", frame, res, Decimal("1"), costs))
