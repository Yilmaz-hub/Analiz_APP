import streamlit as st
import pandas as pd
from data_fetchers import get_market_data
from technical_analysis import calculate_sr_advanced, calculate_oracle_signal_v2

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
                    row = {"Varlık": coin_name, "Günlük Sinyal": "—", "Haftalık Sinyal": "—", "Fiyat": "—"}

                    for tf, col_name in [("1d", "Günlük Sinyal"), ("1wk", "Haftalık Sinyal")]:
                        try:
                            df, _ = get_market_data(source_pref, symbol, tf)
                            if isinstance(df, pd.DataFrame) and not getattr(df, 'empty', True) and len(df) > 50:  # type: ignore
                                s_list, r_list = calculate_sr_advanced(df, tf)
                                status, color, _ = calculate_oracle_signal_v2(df, s_list, r_list)
                                row[col_name] = status

                                if tf == "1d":
                                    row["Fiyat"] = f"${df['Close'].iloc[-1]:,.4f}"  # type: ignore
                        except Exception:
                            row[col_name] = "HATA"

                    scan_results.append(row)
                    progress_bar.progress((idx + 1) / total)

                if scan_results:
                    df_results = pd.DataFrame(scan_results)

                    # Color-code signals for readability
                    def highlight_signal(val):
                        if isinstance(val, str):
                            if "GÜÇLÜ AL" in val: return "color: lime; font-weight: bold"
                            if "AL" in val: return "color: green"
                            if "GÜÇLÜ SAT" in val: return "color: red; font-weight: bold"
                            if "SAT" in val: return "color: salmon"
                        return ""

                    styled = df_results.style.applymap(highlight_signal, subset=["Günlük Sinyal", "Haftalık Sinyal"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # Summary counts
                    daily_buy = sum(1 for r in scan_results if "AL" in r["Günlük Sinyal"])
                    daily_sell = sum(1 for r in scan_results if "SAT" in r["Günlük Sinyal"])
                    st.caption(f"📊 Günlük: {daily_buy} AL / {daily_sell} SAT / {total - daily_buy - daily_sell} NÖTR")
                else:
                    st.warning("Tarama sonucu boş döndü.")
