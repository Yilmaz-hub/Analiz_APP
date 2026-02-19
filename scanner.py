import streamlit as st
import pandas as pd
import time
from data_fetchers import get_market_data
from technical_analysis import calculate_sr_advanced, calculate_oracle_signal_v2

def render_opportunity_scanner(coin_map, source_pref, intervals):
    """
    Full AI Market Scanner — placed in main content area.
    Scans all assets with RSI, trend, support scoring and allocation recommendations.
    """
    st.divider()
    st.header("💼 Varlık ve Fırsat Yönetimi")

    with st.expander("🔍 Piyasayı Tara & Fırsat Bul (AI)", expanded=False):
        st.info("Bu modül RSI, Trend ve Destek/Direnç analizi yaparak kasa yönetimi önerir.")

        scan_tf = st.radio("Tarama Periyodu:", ["4h", "1d", "1wk"], horizontal=True, format_func=lambda x: intervals.get(x, x))

        if st.button("🚀 Taramayı Başlat"):
            best_opp = None
            best_score = -100

            progress_bar = st.progress(0)
            status_text = st.empty()

            if 'coin_map' in st.session_state:
                current_map = st.session_state['coin_map']
            else:
                current_map = coin_map.copy()
            coins_to_scan = list(current_map.keys())
            total_coins = len(coins_to_scan)

            results_scan = []

            for i, c_name in enumerate(coins_to_scan):
                status_text.text(f"Analiz ediliyor: {c_name}...")
                progress_bar.progress((i + 1) / total_coins)

                sym = current_map[c_name]
                try:
                    d_scan, _ = get_market_data("Yahoo Finance", sym, scan_tf)
                except Exception:
                    d_scan = None

                if isinstance(d_scan, pd.DataFrame) and not getattr(d_scan, 'empty', True) and len(d_scan) > 20:  # type: ignore
                    last_price = d_scan['Close'].iloc[-1]  # type: ignore
                    last_rsi = d_scan['RSI'].iloc[-1]  # type: ignore
                    ema_50 = d_scan['EMA_50'].iloc[-1]  # type: ignore

                    # Destek/Direnç
                    sup_list, res_list = calculate_sr_advanced(d_scan, scan_tf)
                    nearest_sup = max([s for s in sup_list if s < last_price], default=0)

                    # === ANA SİNYAL SİSTEMİ ===
                    main_signal, main_color, target_msg = calculate_oracle_signal_v2(d_scan, sup_list, res_list)

                    # --- TARAYICI PUANLAMASI ---
                    score = 50
                    reasons = []

                    # RSI
                    if last_rsi < 35:
                        score += 20; reasons.append("RSI Dip")
                    elif last_rsi > 65:
                        score -= 20; reasons.append("RSI Zirve")

                    # Trend
                    if last_price > ema_50:
                        score += 10; reasons.append("Uptrend")
                    else:
                        reasons.append("Downtrend")

                    # Destek
                    if nearest_sup > 0 and (last_price - nearest_sup) / last_price < 0.03:
                        score += 25; reasons.append("Desteğe Yakın")

                    # Sinyal bonus
                    if "GÜÇLÜ AL" in main_signal:
                        score += 15; reasons.append("Güçlü AL Sinyali")
                    elif "AL" in main_signal:
                        score += 5
                    elif "GÜÇLÜ SAT" in main_signal:
                        score -= 15
                    elif "SAT" in main_signal:
                        score -= 5

                    # --- BAKİYE YÖNETİMİ ÖNERİSİ ---
                    alloc_pct = 0
                    if score >= 85: alloc_pct = 10
                    elif score >= 70: alloc_pct = 5
                    elif score >= 60: alloc_pct = 2.5

                    # Emoji
                    if "GÜÇLÜ AL" in main_signal or "AL" in main_signal:
                        signal_display = f"🟢 {main_signal}"
                    elif "GÜÇLÜ SAT" in main_signal or "SAT" in main_signal:
                        signal_display = f"🔴 {main_signal}"
                    else:
                        signal_display = f"⚪ {main_signal}"

                    results_scan.append({
                        "Coin": c_name,
                        "Ana Sinyal": signal_display,
                        "Tarayıcı Puanı": score,
                        "Önerilen Kasa (%)": f"%{alloc_pct}" if alloc_pct > 0 else "-",
                        "Fiyat": f"${last_price:,.2f}",
                        "RSI": f"{last_rsi:.0f}",
                        "Sebep": ", ".join(reasons) if reasons else "Standart"
                    })

                    if score > best_score:
                        best_score = score
                        best_opp = results_scan[-1]

            progress_bar.empty()
            status_text.empty()

            # === EN İYİ FIRSAT KARTI ===
            if best_opp:
                st.success(f"🌟 **EN İYİ FIRSAT:** {best_opp['Coin']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tarayıcı Puanı", best_opp['Tarayıcı Puanı'])
                c2.metric("Ana Sinyal", best_opp['Ana Sinyal'])
                c3.metric("Fiyat", best_opp['Fiyat'])
                c4.metric("Önerilen Yatırım", best_opp['Önerilen Kasa (%)'])
                st.caption(f"📌 Sebep: {best_opp['Sebep']}")

            # === TABLO (Sıralı & Filtrelenebilir) ===
            if results_scan:
                st.divider()
                st.markdown("### 📊 Tüm Tarama Sonuçları")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    filter_signal = st.multiselect(
                        "Sinyal Filtresi",
                        options=["AL", "SAT", "NÖTR"],
                        default=["AL", "SAT", "NÖTR"]
                    )
                with col_f2:
                    min_score = st.slider("Minimum Puan", 0, 100, 0)

                df_results = pd.DataFrame(results_scan)

                # Filtreleme
                filtered_df = df_results[
                    (df_results['Ana Sinyal'].str.contains('|'.join(filter_signal))) &
                    (df_results['Tarayıcı Puanı'] >= min_score)
                ].sort_values(by="Tarayıcı Puanı", ascending=False)

                st.dataframe(filtered_df, use_container_width=True, height=400)

                # === İSTATİSTİKLER ===
                st.divider()
                st.markdown("### 📈 Tarama İstatistikleri")

                total_analyzed = len(df_results)
                buy_signals = len(df_results[df_results['Ana Sinyal'].str.contains('AL')])
                sell_signals = len(df_results[df_results['Ana Sinyal'].str.contains('SAT')])
                neutral_signals = total_analyzed - buy_signals - sell_signals

                stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
                stat_c1.metric("Toplam Analiz", total_analyzed)
                stat_c2.metric("🟢 AL Sinyali", buy_signals)
                stat_c3.metric("🔴 SAT Sinyali", sell_signals)
                stat_c4.metric("⚪ NÖTR", neutral_signals)

                if total_analyzed > 0:
                    st.info(f"""
                    **📊 Piyasa Dağılımı:**
                    - AL Sinyali: %{(buy_signals/total_analyzed*100):.1f}
                    - SAT Sinyali: %{(sell_signals/total_analyzed*100):.1f}
                    - NÖTR: %{(neutral_signals/total_analyzed*100):.1f}
                    """)

                # === CSV İNDİRME ===
                st.divider()
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Sonuçları CSV olarak İndir",
                    data=csv,
                    file_name=f"piyasa_tarama_{scan_tf}_{time.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
