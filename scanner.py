import streamlit as st
import pandas as pd
import time
from data_fetchers import get_market_data
from technical_analysis import calculate_sr_advanced
from signal_engine import generate_composite_signal

def render_opportunity_scanner(coin_map, source_pref, intervals):
    """
    Full AI Market Scanner — uses the new Composite Decision Engine
    for consistent, clear buy/sell signals across all assets.
    """
    st.divider()
    st.header("💼 Varlık ve Fırsat Yönetimi")

    with st.expander("🔍 Piyasayı Tara & Fırsat Bul (AI)", expanded=False):
        st.info("Karar Motoru ile tüm varlıkları tarar. Günlük ve Haftalık sinyaller tek tabloda gösterilir.")

        if st.button("🚀 Taramayı Başlat"):
            best_opp = None
            best_score = -999

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

                row = {
                    "Coin": c_name,
                    "Günlük Sinyal": "—",
                    "Haftalık Sinyal": "—",
                    "Fiyat": "—",
                    "RSI (1G)": "—",
                    "Karar Skoru": 0,
                    "Güven (%)": 0,
                    "Önerilen Kasa (%)": "-",
                    "Sebep": ""
                }

                daily_score = 0
                weekly_score = 0
                all_reasons = []
                daily_confidence = 0

                for tf, signal_col in [("1d", "Günlük Sinyal"), ("1wk", "Haftalık Sinyal")]:
                    try:
                        d_scan, _ = get_market_data("Yahoo Finance", sym, tf)
                    except Exception:
                        d_scan = None

                    if isinstance(d_scan, pd.DataFrame) and not getattr(d_scan, 'empty', True) and len(d_scan) > 20:
                        # Use composite signal engine
                        sup_list, res_list = calculate_sr_advanced(d_scan, tf)
                        comp_signal = generate_composite_signal(d_scan, tf, sup_list, res_list)

                        # Display signal with emoji
                        if "AL" in comp_signal.verdict:
                            signal_display = f"🟢 {comp_signal.verdict}"
                        elif "SAT" in comp_signal.verdict:
                            signal_display = f"🔴 {comp_signal.verdict}"
                        else:
                            signal_display = f"⚪ {comp_signal.verdict}"

                        row[signal_col] = signal_display

                        if tf == "1d":
                            last_price = d_scan['Close'].iloc[-1]
                            row["Fiyat"] = f"${last_price:,.2f}"
                            last_rsi = d_scan['RSI'].iloc[-1] if 'RSI' in d_scan.columns else 50
                            row["RSI (1G)"] = f"{last_rsi:.0f}"
                            daily_score = comp_signal.final_score
                            daily_confidence = comp_signal.confidence
                            all_reasons.extend(comp_signal.reasons[:3])  # Top 3 reasons

                        if tf == "1wk":
                            weekly_score = comp_signal.final_score

                # Combined score (daily 60% + weekly 40%)
                combined_score = daily_score * 0.6 + weekly_score * 0.4

                # Allocation recommendation based on combined score and confidence
                alloc_pct = 0
                if combined_score >= 50 and daily_confidence >= 60:
                    alloc_pct = 10
                elif combined_score >= 30 and daily_confidence >= 50:
                    alloc_pct = 5
                elif combined_score >= 15 and daily_confidence >= 40:
                    alloc_pct = 2.5

                row["Karar Skoru"] = round(combined_score, 1)
                row["Güven (%)"] = round(daily_confidence, 0)
                row["Önerilen Kasa (%)"] = f"%{alloc_pct}" if alloc_pct > 0 else "-"
                row["Sebep"] = ", ".join(all_reasons[:3]) if all_reasons else "Standart"

                results_scan.append(row)

                if combined_score > best_score:
                    best_score = combined_score
                    best_opp = results_scan[-1]

            progress_bar.empty()
            status_text.empty()

            # === EN İYİ FIRSAT KARTI ===
            if best_opp:
                st.success(f"🌟 **EN İYİ FIRSAT:** {best_opp['Coin']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Karar Skoru", best_opp['Karar Skoru'])
                c2.metric("Günlük Sinyal", best_opp['Günlük Sinyal'])
                c3.metric("Fiyat", best_opp['Fiyat'])
                c4.metric("Önerilen Yatırım", best_opp['Önerilen Kasa (%)'])
                st.caption(f"📌 Sebep: {best_opp['Sebep']}")

            # === TABLO ===
            if results_scan:
                st.divider()
                st.markdown("### 📊 Tüm Tarama Sonuçları")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    filter_signal = st.multiselect(
                        "Sinyal Filtresi",
                        options=["AL", "SAT", "BEKLE"],
                        default=["AL", "SAT", "BEKLE"]
                    )
                with col_f2:
                    min_score = st.slider("Minimum Skor", -100, 100, -100)

                display_cols = ["Coin", "Günlük Sinyal", "Haftalık Sinyal", "Fiyat", "RSI (1G)", "Karar Skoru", "Güven (%)", "Önerilen Kasa (%)", "Sebep"]
                df_results = pd.DataFrame(results_scan)[display_cols]

                # Filtreleme
                mask = df_results['Günlük Sinyal'].str.contains('|'.join(filter_signal)) | \
                       df_results['Haftalık Sinyal'].str.contains('|'.join(filter_signal))
                filtered_df = df_results[mask & (df_results['Karar Skoru'] >= min_score)].sort_values(
                    by="Karar Skoru", ascending=False
                )

                st.dataframe(filtered_df, use_container_width=True, height=400, hide_index=True)

                # === İSTATİSTİKLER ===
                st.divider()
                st.markdown("### 📈 Tarama İstatistikleri")

                total_analyzed = len(df_results)
                buy_signals = len(df_results[df_results['Günlük Sinyal'].str.contains('AL')])
                sell_signals = len(df_results[df_results['Günlük Sinyal'].str.contains('SAT')])
                neutral_signals = total_analyzed - buy_signals - sell_signals

                stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
                stat_c1.metric("Toplam Analiz", total_analyzed)
                stat_c2.metric("🟢 AL Sinyali", buy_signals)
                stat_c3.metric("🔴 SAT Sinyali", sell_signals)
                stat_c4.metric("⚪ BEKLE", neutral_signals)

                if total_analyzed > 0:
                    st.info(f"""
                    **📊 Piyasa Dağılımı (Günlük):**
                    - AL Sinyali: %{(buy_signals/total_analyzed*100):.1f}
                    - SAT Sinyali: %{(sell_signals/total_analyzed*100):.1f}
                    - BEKLE: %{(neutral_signals/total_analyzed*100):.1f}
                    """)

                # === CSV İNDİRME ===
                st.divider()
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Sonuçları CSV olarak İndir",
                    data=csv,
                    file_name=f"piyasa_tarama_{time.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
