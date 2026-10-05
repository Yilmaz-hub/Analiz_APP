"""Spec 0012 kanıtı: sanal takip ekranı (seed'li geçici SQLite ile)."""
import os, sys, tempfile
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("ANALIZ_APP_DB_URL", "sqlite:///" + os.path.join(tempfile.gettempdir(), "evidence_0012.db"))
import streamlit as st
import forward_tracker as ft
import forward_ui

if "seeded" not in st.session_state:
    now = datetime.now(timezone.utc)
    for asset in ("BTC-USD", "ETH-USD", "AVAX-USD"):
        for i, decision in enumerate(("AL", "BEKLE", "AL")):
            day = date.today() - timedelta(days=3 - i)
            at = ft.available_at("KRIPTO", day) + timedelta(minutes=20)
            ft.record(ft.ForwardDecision(asset=asset, strategy_version="V1~147411cc", candle_day=day,
                decision=decision, source="Binance", evaluated_at=at, candle={"close": "100"},
                assumptions={"dolum": "ertesi gün açılışı", "komisyon": "bilinmiyor", "sermaye": "bilinmiyor"},
                on_time=True, real_clock=True, regime="YUKSELEN", regime_version="R1"))
    note = "; ".join(f"{s}: miktar adımı/varsayımlar eksik; sanal işlem hesaplanmadı." for s in
                     ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "DOGE-USD")) + "; EURUSD=X: piyasası tanınmıyor, atlandı."
    ft.record_run(now - timedelta(hours=3), True, note)
    st.session_state["seeded"] = True

st.title("Sanal takip — ekran kanıtı")
with st.expander("📡 İleri Dönem Sanal Takip (ekran kapalıyken)", expanded=True):
    forward_ui.render_forward_status(datetime.now(timezone.utc))
    if st.toggle("Sanal takip durumunu göster", key="forward_show"):
        forward_ui.render_forward_panel(datetime.now(timezone.utc))
