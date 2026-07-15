import streamlit as st
import pandas as pd
import time
from data_fetchers import get_market_data
from signal_engine import generate_stable_signal
from technical_analysis import calculate_regime_score
from config import RegimeConfig, SizingConfig

def vol_sizing_factor(asset_vol, median_vol):
    """Inverse-volatility sizing multiplier (validated 2026-07-15, 3 samples):
    calm assets get up to ADJ_MAX x the base allocation, wild ones as little
    as ADJ_MIN x. Neutral (1.0) when volatility can't be measured."""
    if not asset_vol or not median_vol or asset_vol <= 0 or median_vol <= 0:
        return 1.0
    return max(SizingConfig.ADJ_MIN, min(median_vol / asset_vol, SizingConfig.ADJ_MAX))

def render_opportunity_scanner(coin_map, source_pref, intervals):
    """
    Full AI Market Scanner — uses the new Composite Decision Engine
    for consistent, clear buy/sell signals across all assets.
    """
    st.divider()
    st.header("💼 Varlık ve Fırsat Yönetimi")

    with st.expander("🔍 Piyasayı Tara & Fırsat Bul (AI)", expanded=False):
        st.info("Karar Motoru ile tüm varlıkları tarar. Sonuçlar Fırsat Puanı'na göre sıralanır: sinyal gücü + trend rejimi. Strateji trendli varlıklarda kazandığı için zayıf rejimdeki varlıklara kasa önerilmez.")

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
                    "Trend Rejimi": "—",
                    "Fiyat": "—",
                    "RSI (1G)": "—",
                    "Fırsat Puanı": 0,
                    "Karar Skoru": 0,
                    "Güven (%)": 0,
                    "Önerilen Kasa (%)": "-",
                    "Sebep": ""
                }

                daily_score = 0
                weekly_score = 0
                all_reasons = []
                daily_confidence = 0
                regime_score = 0

                for tf, signal_col in [("1d", "Günlük Sinyal"), ("1wk", "Haftalık Sinyal")]:
                    try:
                        d_scan, _ = get_market_data("Yahoo Finance", sym, tf)
                    except Exception:
                        d_scan = None

                    if isinstance(d_scan, pd.DataFrame) and not getattr(d_scan, 'empty', True) and len(d_scan) > 20:
                        # Stable (whipsaw-filtered) composite signal — same as dashboard
                        comp_signal = generate_stable_signal(d_scan, tf)

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

                            # Trend regime quality — the strategy's edge lives
                            # in trending assets, so this drives the ranking
                            regime = calculate_regime_score(d_scan)
                            if regime:
                                regime_score = regime['score']
                                row["Trend Rejimi"] = f"{regime['label']} ({regime['score']})"

                            # Realized volatility for inverse-vol sizing
                            asset_vol = d_scan['Close'].pct_change().tail(SizingConfig.VOL_LOOKBACK).std()
                            row["_vol"] = float(asset_vol) if pd.notna(asset_vol) else 0.0

                        if tf == "1wk":
                            weekly_score = comp_signal.final_score

                # Combined score (daily 60% + weekly 40%)
                combined_score = daily_score * 0.6 + weekly_score * 0.4

                # Opportunity rank: signal strength blended with trend regime
                # (regime 0-100 mapped to -100..+100 so a choppy asset drags
                # the rank down even when its signal score looks good)
                rank_score = (RegimeConfig.RANK_SIGNAL_WEIGHT * combined_score
                              + RegimeConfig.RANK_REGIME_WEIGHT * (regime_score * 2 - 100))

                # Allocation recommendation — only in a tradeable trend regime,
                # since the backtest shows the signal loses money on choppy assets
                alloc_pct = 0
                if regime_score >= RegimeConfig.MIN_TRADEABLE_SCORE:
                    if combined_score >= 50 and daily_confidence >= 60:
                        alloc_pct = 10
                    elif combined_score >= 30 and daily_confidence >= 50:
                        alloc_pct = 5
                    elif combined_score >= 15 and daily_confidence >= 40:
                        alloc_pct = 2.5
                elif combined_score >= 15:
                    all_reasons.insert(0, "Zayıf trend rejimi — strateji bu profilde kanıtsız")

                row["Fırsat Puanı"] = round(rank_score, 1)
                row["Karar Skoru"] = round(combined_score, 1)
                row["Güven (%)"] = round(daily_confidence, 0)
                row["_alloc"] = alloc_pct  # finalized after the loop (needs cross-asset vol)
                row["Sebep"] = ", ".join(all_reasons[:3]) if all_reasons else "Standart"

                results_scan.append(row)

                if rank_score > best_score:
                    best_score = rank_score
                    best_opp = results_scan[-1]

            progress_bar.empty()
            status_text.empty()

            # Inverse-volatility sizing: scale each base allocation by
            # median_vol / asset_vol (clipped) so calm assets get more capital
            # and volatile ones less — validated vs equal-weight on 3 samples.
            vols = [r["_vol"] for r in results_scan if r.get("_vol", 0) > 0]
            median_vol = float(pd.Series(vols).median()) if vols else 0.0
            for r in results_scan:
                alloc = r.pop("_alloc", 0)
                v = r.pop("_vol", 0.0)
                if alloc > 0:
                    adj_alloc = alloc * vol_sizing_factor(v, median_vol)
                    r["Önerilen Kasa (%)"] = f"%{adj_alloc:.1f}"

            # === EN İYİ FIRSAT KARTI ===
            if best_opp:
                st.success(f"🌟 **EN İYİ FIRSAT:** {best_opp['Coin']}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Fırsat Puanı", best_opp['Fırsat Puanı'])
                c2.metric("Günlük Sinyal", best_opp['Günlük Sinyal'])
                c3.metric("Trend Rejimi", best_opp['Trend Rejimi'])
                c4.metric("Fiyat", best_opp['Fiyat'])
                c5.metric("Önerilen Yatırım", best_opp['Önerilen Kasa (%)'])
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

                display_cols = ["Coin", "Günlük Sinyal", "Haftalık Sinyal", "Trend Rejimi", "Fiyat", "RSI (1G)", "Fırsat Puanı", "Karar Skoru", "Güven (%)", "Önerilen Kasa (%)", "Sebep"]
                df_results = pd.DataFrame(results_scan)[display_cols]

                # Filtreleme
                mask = df_results['Günlük Sinyal'].str.contains('|'.join(filter_signal)) | \
                       df_results['Haftalık Sinyal'].str.contains('|'.join(filter_signal))
                filtered_df = df_results[mask & (df_results['Karar Skoru'] >= min_score)].sort_values(
                    by="Fırsat Puanı", ascending=False
                )

                st.dataframe(filtered_df, width="stretch", height=400, hide_index=True)

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
