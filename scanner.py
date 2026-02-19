import streamlit as st
import pandas as pd
from data_fetchers import get_market_data
from technical_analysis import calculate_sr_advanced, calculate_oracle_signal_v2

def _signal_to_score(status):
    """Sinyal durumunu sayısal skora çevirir (yatırım oranı hesabı için)."""
    if "GÜÇLÜ AL" in status: return 3
    if "AL" in status: return 2
    if "GÜÇLÜ SAT" in status: return -2
    if "SAT" in status: return -1
    return 0  # NÖTR

def render_opportunity_scanner(coin_map, source_pref):
    st.sidebar.divider()
    with st.sidebar.expander("🔎 Fırsatları Tara", expanded=False):
        st.write("Tüm varlıkları Günlük ve Haftalık dilimde tarar.")
        if st.button("Taramayı Başlat", key="btn_scan_opportunities"):
            with st.spinner("Piyasa taranıyor... bu biraz sürebilir."):
                scan_results = []
                total = len(coin_map)
                progress_bar = st.progress(0)

                for idx, (coin_name, symbol) in enumerate(coin_map.items()):
                    row = {
                        "Varlık": coin_name,
                        "Günlük Sinyal": "—",
                        "Haftalık Sinyal": "—",
                        "Fiyat": "—",
                        "_daily_score": 0,
                        "_weekly_score": 0
                    }

                    for tf, col_name, score_key in [("1d", "Günlük Sinyal", "_daily_score"), ("1wk", "Haftalık Sinyal", "_weekly_score")]:
                        try:
                            df, _ = get_market_data(source_pref, symbol, tf)
                            if isinstance(df, pd.DataFrame) and not getattr(df, 'empty', True) and len(df) > 50:  # type: ignore
                                s_list, r_list = calculate_sr_advanced(df, tf)
                                status, color, _ = calculate_oracle_signal_v2(df, s_list, r_list)
                                row[col_name] = status
                                row[score_key] = _signal_to_score(status)

                                if tf == "1d":
                                    row["Fiyat"] = f"${df['Close'].iloc[-1]:,.4f}"  # type: ignore
                        except Exception:
                            row[col_name] = "HATA"

                    scan_results.append(row)
                    progress_bar.progress((idx + 1) / total)

                if scan_results:
                    # --- Yatırım Oranı Hesaplama ---
                    # Günlük skora %60, haftalık skora %40 ağırlık ver
                    for r in scan_results:
                        combined = (r["_daily_score"] * 0.6) + (r["_weekly_score"] * 0.4)
                        r["_combined"] = max(combined, 0)  # Negatif skor = yatırım önerme

                    total_positive_score = sum(r["_combined"] for r in scan_results)

                    for r in scan_results:
                        if total_positive_score > 0:
                            r["Yatırım %"] = f"%{(r['_combined'] / total_positive_score) * 100:.1f}"
                        else:
                            r["Yatırım %"] = "%0.0"

                    # Dahili skor sütunlarını kaldır
                    display_cols = ["Varlık", "Günlük Sinyal", "Haftalık Sinyal", "Fiyat", "Yatırım %"]
                    df_results = pd.DataFrame(scan_results)[display_cols]

                    # Renklendirme
                    def highlight_signal(val):
                        if isinstance(val, str):
                            if "GÜÇLÜ AL" in val: return "color: lime; font-weight: bold"
                            if "AL" in val: return "color: green"
                            if "GÜÇLÜ SAT" in val: return "color: red; font-weight: bold"
                            if "SAT" in val: return "color: salmon"
                        return ""

                    styled = df_results.style.applymap(highlight_signal, subset=["Günlük Sinyal", "Haftalık Sinyal"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # Özet
                    daily_buy = sum(1 for r in scan_results if "AL" in r["Günlük Sinyal"])
                    daily_sell = sum(1 for r in scan_results if "SAT" in r["Günlük Sinyal"])
                    st.caption(f"📊 Günlük: {daily_buy} AL / {daily_sell} SAT / {total - daily_buy - daily_sell} NÖTR")

                    # En yüksek yatırım önerisi
                    top_picks = sorted(scan_results, key=lambda x: x["_combined"], reverse=True)[:3]
                    top_picks = [t for t in top_picks if t["_combined"] > 0]
                    if top_picks:
                        st.markdown("**🏆 En İyi Fırsatlar:**")
                        for t in top_picks:
                            st.success(f"**{t['Varlık']}** → {t['Yatırım %']} yatırım oranı önerisi")
                else:
                    st.warning("Tarama sonucu boş döndü.")
