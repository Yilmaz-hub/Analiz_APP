import streamlit as st
import pandas as pd
from data_fetchers import get_market_data
from portfolio import multi_timeframe_confirmation
from technical_analysis import calculate_sr_advanced, calculate_oracle_signal_v2

def render_opportunity_scanner(coin_map, source_pref):
    st.sidebar.divider()
    with st.sidebar.expander("🔎 Fırsatları Tara", expanded=False):
        st.write("Tüm kayıtlı varlıkları tarar ve güçlü AL/SAT sinyallerini listeler.")
        if st.button("Taramayı Başlat", key="btn_scan_opportunities"):
            with st.spinner("Piyasa taranıyor... bu biraz sürebilir."):
                opportunities = []
                total = len(coin_map)
                progress_bar = st.progress(0)
                
                for idx, (coin_name, symbol) in enumerate(coin_map.items()):
                    try:
                        df, _ = get_market_data(source_pref, symbol, "1d")
                        if isinstance(df, pd.DataFrame) and not getattr(df, 'empty', True) and len(df) > 50:  # type: ignore
                            s_list, r_list = calculate_sr_advanced(df, "1d")
                            status, color, target = calculate_oracle_signal_v2(df, s_list, r_list)
                            
                            # Güçlü AL veya AL sinyali olanları filtrele
                            if "AL" in status:
                                mtf_status, _ = multi_timeframe_confirmation(coin_name, symbol, source_pref)
                                
                                # Sadece MTF onayı alanları listele
                                if "AL" in mtf_status:
                                    opportunities.append({
                                        "Varlık": coin_name,
                                        "1G Sinyal": status,
                                        "MTF Onay": mtf_status,
                                        "Fiyat": f"${df['Close'].iloc[-1]:.4f}"  # type: ignore
                                    })
                    except Exception as e:
                        pass # Ignore individual coin errors to keep scanner running
                        
                    progress_bar.progress((idx + 1) / total)
                
                if opportunities:
                    st.success(f"{len(opportunities)} Güçlü Fırsat Bulundu!")
                    st.dataframe(pd.DataFrame(opportunities), use_container_width=True)
                else:
                    st.warning("Şu an için çok güçlü bir fırsat bulunamadı.")
