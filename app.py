import streamlit as st
import pandas as pd
import time
import requests
import traceback
from config import DEFAULT_COIN_MAP, PATTERN_INFO, UIConfig, FileConfig, DataFetchConfig
from logger import logger

# === YENİ MODÜLLERDEN IMPORTLAR ===
from data_fetchers import get_market_data, get_fear_greed_index
from technical_analysis import detect_advanced_patterns, calculate_extended_trendlines, detect_patterns, run_strategy_backtest
from ml_models import calculate_smart_prediction_FIXED
from portfolio import load_portfolio, save_portfolio, validate_portfolio_risk, check_active_positions_auto_close, multi_timeframe_confirmation
from ui_components import render_sidebar_settings, render_asset_management, render_main_chart
from scanner import render_opportunity_scanner
from data_fetchers import get_live_price_for_portfolio
from signal_engine import generate_stable_signal, CompositeSignal
from weight_profiles import get_weights_for_symbol
from advanced_analysis import detect_elliott_wave, analyze_ichimoku, detect_wyckoff_phase, analyze_market_structure
import theme

st.set_page_config(layout="wide", page_title="Pro Trader V48 (Modular Edition)")

# CSS
theme.inject()

# GİRİŞ VERİLERİNİ YÜKLE
def load_assets():
    f = FileConfig.ASSETS_FILE
    import os, json
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Asset list load error, using defaults: {e}")
            return DEFAULT_COIN_MAP.copy()
    return DEFAULT_COIN_MAP.copy()

def save_assets(data):
    f = FileConfig.ASSETS_FILE
    import json, os
    tmp = f"{f}.tmp"
    with open(tmp, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(tmp, f)

if 'coin_map' not in st.session_state:
    st.session_state['coin_map'] = load_assets()

if 'portfolio_data' not in st.session_state:
    st.session_state['portfolio_data'] = load_portfolio()

# --- ARAYÜZ (SIDEBAR) ---
tg_token, tg_chat = render_sidebar_settings()
render_asset_management(st.session_state['coin_map'], save_assets)

st.sidebar.divider()
current_assets = list(st.session_state['coin_map'].keys())
src_pref = st.sidebar.radio("📡 Kaynak:", ["Binance", "OKX", "Yahoo Finance"])

if not current_assets: current_assets = ["Bitcoin (BTC)"]

sel_c = st.sidebar.selectbox("Enstrüman:", current_assets)
symbol = st.session_state['coin_map'].get(sel_c, "BTC-USD")

st.sidebar.divider()
show_cloud = st.sidebar.checkbox("☁️ Destek/Direnç Bulutu", value=True)
show_ai = st.sidebar.checkbox("🤖 AI Trend", value=True)
show_pred = st.sidebar.checkbox("🔮 AI Tahmin", value=True)

st.sidebar.subheader("🔍 Filtreler")
show_all_pats = st.sidebar.checkbox("Hepsini Aç/Kapat", value=True)
f_wm = st.sidebar.checkbox("- W ve M", value=True)
f_candle = st.sidebar.checkbox("- Mumlar", value=True)
f_advanced = st.sidebar.checkbox("- Gelişmiş Formasyonlar (Üçgen, ABCD, Baş-Omuz)", value=False)
auto = st.sidebar.checkbox("Otomatik Bot")

with st.sidebar.expander("🔄 Çoklu Dilim Onayı", expanded=False):
    if st.button("Analiz Et"):
        confirmation, details = multi_timeframe_confirmation(sel_c, symbol, src_pref)
        st.markdown(f"### {confirmation}")
        for tf, sig in details.items():
            color = "green" if "AL" in sig else ("red" if "SAT" in sig else "gray")
            st.markdown(f"**{tf}:** <span style='color:{color}'>{sig}</span>", unsafe_allow_html=True)

try:
    fg_value, fg_class = get_fear_greed_index()
    fg_color = "green" if fg_value < 30 else ("red" if fg_value > 70 else "orange")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**😱 Piyasa Duygusu:** <span style='color:{fg_color}'>{fg_class} ({fg_value})</span>", unsafe_allow_html=True)
except Exception as e:
    logger.debug(f"Fear & Greed sidebar render failed: {e}")

intervals = {"4h": "4 Saatlik", "1d": "Günlük", "1wk": "Haftalık"}
results = {}
active_src = ""

# --- ANA VERİ DÖNGÜSÜ ---
composite_signals = {}  # Store composite signals per timeframe
for tf, label in intervals.items():
    df, src = get_market_data(src_pref, symbol, tf)
    if tf == "1d": active_src = src
    results[tf] = df
    
    if df is not None:
        # Tuned weights only apply to the daily signal — the optimizer
        # only ever validates against daily data (see weight_profiles.py).
        sig_weights = get_weights_for_symbol(symbol) if tf == "1d" else None
        # Stable (whipsaw-filtered) composite signal — closed candles only
        comp_sig = generate_stable_signal(df, tf, weights=sig_weights)
        composite_signals[tf] = comp_sig

        st.sidebar.markdown("---")
        st.sidebar.markdown(theme.sidebar_signal(label, comp_sig), unsafe_allow_html=True)
        if comp_sig.entry_price > 0 and "AL" in comp_sig.verdict:
            st.sidebar.caption(f"🎯 TP: ${comp_sig.take_profit_1:,.2f} | 🛑 SL: ${comp_sig.stop_loss:,.2f}")
        elif "SAT" in comp_sig.verdict:
            st.sidebar.caption("⚠️ Çıkış sinyali — short önerisi değildir")

        adv_patterns = detect_advanced_patterns(df)
    
        if adv_patterns:
            st.sidebar.markdown("**🔍 Formasyonlar:**")
            for pat in adv_patterns:
                emoji_dir = "🟢" if pat['direction'] == 'BULLISH' else ("🔴" if pat['direction'] == 'BEARISH' else "⚪")
                st.sidebar.caption(f"{emoji_dir} {pat['name']} (%{pat['confidence']})")
    else: 
        st.sidebar.warning(f"{label}: Bekleniyor...")

safe_src = str(active_src) if active_src else ""
st.markdown(theme.page_header(sel_c, safe_src), unsafe_allow_html=True)

view_tf = st.selectbox("Periyot:", list(intervals.keys()), format_func=lambda x: intervals[x])
df_view = results[view_tf]

if df_view is not None:
    curr = df_view['Close'].iloc[-1]
    prev = df_view['Close'].iloc[-2] if len(df_view) > 1 else curr
    change_pct = ((curr - prev) / prev) * 100 if prev > 0 else 0
    current_atr = df_view.iloc[-1]['ATR'] if 'ATR' in df_view.columns else curr * 0.02
    st.metric(label=f"{sel_c} Anlık Fiyat", value=f"${curr:,.2f}", delta=f"{change_pct:+.2f}%")
    
    with st.sidebar.expander("🧮 Hızlı Risk Hesapla", expanded=False):
        st.caption("Pozisyon büyüklüğü hesaplar.")
        calc_price = st.number_input("Giriş Fiyatı", value=float(curr), format="%.4f")
        calc_sl = st.number_input("Stop Fiyatı", value=float(curr)*0.98, format="%.4f")
        calc_balance = st.number_input("Kasa ($)", value=1000.0)
        calc_risk = st.number_input("Risk (%)", value=1.0, step=0.5)
        
        if calc_price > 0 and calc_sl > 0 and calc_price != calc_sl:
            risk_amt = calc_balance * (calc_risk / 100)
            diff_pct = abs(calc_price - calc_sl) / calc_price
            pos_size = risk_amt / diff_pct
            coin_qty = pos_size / calc_price
            st.divider()
            st.write(f"💸 **Risk Tutarı:** ${risk_amt:.2f}")
            st.success(f"💰 **İşlem Büyüklüğü:** ${pos_size:.2f}")
            st.info(f"🪙 **Alınacak Adet:** {coin_qty:.4f}")

    # --- ANA GRAFİK ÇİZİMİ ---
    f_dates, f_prices, ai_score = [], [], 0
    if show_pred: f_dates, f_prices, ai_score = calculate_smart_prediction_FIXED(df_view)
    
    lines = calculate_extended_trendlines(df_view) if show_ai else []
    items_raw = detect_patterns(df_view) if show_all_pats else []
    
    s_l, r_l = render_main_chart(df_view, view_tf, curr, f_dates, f_prices, ai_score, show_cloud, show_pred, show_ai, show_all_pats, f_wm, f_candle, f_advanced, items_raw, lines)

    # --- ALT PANELLER (Karar Paneli & Analiz) ---
    st.divider()
    
    # Get the composite signal for the current view timeframe
    active_signal = composite_signals.get(view_tf)
    if active_signal is None:
        fallback_weights = get_weights_for_symbol(symbol) if view_tf == "1d" else None
        active_signal = generate_stable_signal(df_view, view_tf, weights=fallback_weights)
    
    # ═══════════════════════════════════════════
    # DECISION DASHBOARD (Karar Paneli)
    # ═══════════════════════════════════════════
    st.markdown("### 🎯 KARAR PANELİ")
    
    dash_col1, dash_col2, dash_col3 = st.columns([1.5, 1, 1.5])
    
    with dash_col1:
        st.markdown(theme.verdict_card(active_signal), unsafe_allow_html=True)
        
        if "AL" in active_signal.verdict:
            st.markdown(theme.trade_plan_card(active_signal), unsafe_allow_html=True)
        elif "SAT" in active_signal.verdict:
            # Short trades are backtest-falsified; SAT = exit/stay out only
            st.markdown(theme.exit_warning_card(active_signal), unsafe_allow_html=True)
        else:
            st.info("ℹ️ İşlem sinyali yok. Piyasa izleniyor...")
    
    with dash_col2:
        st.markdown("**📊 Boyut Skorları**")
        dim_labels = {
            "trend": ("📈 Trend", active_signal.dimension_scores.get("trend", 0)),
            "momentum": ("⚡ Momentum", active_signal.dimension_scores.get("momentum", 0)),
            "volume": ("📦 Hacim", active_signal.dimension_scores.get("volume", 0)),
            "pattern": ("📐 Formasyon", active_signal.dimension_scores.get("pattern", 0)),
            "ml": ("🤖 AI", active_signal.dimension_scores.get("ml", 0)),
            "advanced": ("🌊 Gelişmiş", active_signal.dimension_scores.get("advanced", 0))
        }
        
        for idx, (key, (label, value)) in enumerate(dim_labels.items()):
            st.markdown(theme.dimension_bar(label, value, delay_idx=idx), unsafe_allow_html=True)
        
        # Market Summary Metrics
        st.markdown("---")
        trend = "YÜKSELİŞ ↗️" if curr > df_view['EMA_50'].iloc[-1] else "DÜŞÜŞ ↘️"
        rsi_val = df_view['RSI'].iloc[-1] if 'RSI' in df_view.columns else 50
        st.metric("Trend", trend)
        st.metric("RSI", f"{rsi_val:.1f}")
    
    with dash_col3:
        st.markdown("**💡 Sinyal Gerekçeleri**")
        if active_signal.reasons:
            for reason in active_signal.reasons[:8]:
                st.markdown(theme.reason_line(reason), unsafe_allow_html=True)
        else:
            st.write("Analiz tamamlandı, belirgin sinyal yok.")
        
        st.markdown("---")
        st.markdown("**🕐 Zaman Dilimleri Özeti**")
        for tf_key, tf_label in intervals.items():
            cs = composite_signals.get(tf_key)
            if cs:
                st.markdown(theme.tf_row(tf_label, cs.verdict, cs.confidence), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**🧠 Tespitler**")
        if show_all_pats and items_raw:
            visible_names = []
            for item in items_raw:
                if (item['type'] == 'box' and f_wm) or (item['type'] == 'icon' and f_candle): visible_names.append(item['name'])
            if visible_names:
                for p in list(set(visible_names)): st.caption(PATTERN_INFO.get(p, p))
            else: st.caption("Filtreli formasyon yok.")
        else: st.caption("Formasyonlar kapalı.")

    # --- GELİŞMİŞ ANALİZ PANELİ ---
    with st.expander("🌊 Gelişmiş Analiz (Elliott Wave, Ichimoku, Wyckoff, Piyasa Yapısı)", expanded=False):
        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            ew = detect_elliott_wave(df_view)
            if ew["detected"]:
                rows = f"<div>Tip: <b>{ew['type']}</b> | Yön: <b>{ew['direction']}</b></div><div>Güven: %{ew['confidence']}</div>"
                targets = f"<br>🎯 Hedefler: {' | '.join(f'${t:,.2f}' for t in ew['targets'])}" if ew["targets"] else ""
                st.markdown(theme.analysis_card("🌊 Elliott Wave", rows + targets, ew["direction"], ew["description"]), unsafe_allow_html=True)
            else:
                st.info("🌊 Elliott Wave: Geçerli dalga sayımı bulunamadı.")

            wyck = detect_wyckoff_phase(df_view)
            rows = f"<div>Faz: <b>{wyck['phase']}</b></div>"
            st.markdown(theme.analysis_card("📦 Wyckoff Fazı", rows, wyck["signal"], wyck["description"]), unsafe_allow_html=True)

        with adv_col2:
            ich = analyze_ichimoku(df_view)
            rows = f"<div>Bulut: <b>{ich['cloud_status']}</b></div><div>TK: {ich['tk_cross']}</div><div>{ich['chikou']}</div><div style='margin-top:6px;'>Skor: <b>{ich['score']:+d}</b></div>"
            st.markdown(theme.analysis_card("☁️ Ichimoku Cloud", rows, ich["signal"]), unsafe_allow_html=True)

            ms = analyze_market_structure(df_view)
            badges = ""
            if ms["bos"]: badges += '<span class="at-badge">BOS</span>'
            if ms["choch"]: badges += '<span class="at-badge alarm">CHoCH</span>'
            rows = f"<div>Yapı: <b>{ms['structure']}</b></div>"
            st.markdown(theme.analysis_card("📐 Piyasa Yapısı", rows, ms["signal"], ms["description"], badges), unsafe_allow_html=True)


    # --- BACKTEST ---
    if df_view is not None:
        st.divider()
        with st.expander("📊 Backtest: Strateji Performansı", expanded=False):
            st.info("Ekranda gördüğünüz Karar Motoru sinyalini (onay + histerezis filtreli) geçmiş veride test eder. Not: ML boyutu hız nedeniyle backtestte devre dışıdır, ağırlığı diğer boyutlara dağıtılır.")
            if st.button("🚀 Backtest Başlat"):
                with st.spinner("Backtest çalışıyor..."):
                    import plotly.graph_objects as go
                    tuned_weights = get_weights_for_symbol(symbol) if view_tf == "1d" else None

                    bt_progress = st.progress(0.0)
                    bt_results = run_strategy_backtest(df_view, initial_balance=10000, timeframe=view_tf,
                                                       weights=tuned_weights,
                                                       progress_callback=lambda p: bt_progress.progress(min(1.0, p)))
                    bt_progress.empty()

                    if bt_results is None:
                        st.warning("Yeterli işlem oluşmadı. Daha uzun veri gerekebilir.")
                    else:
                        if tuned_weights:
                            st.caption("🎯 Bu varlık sınıfı için ayarlanmış ağırlıklar kullanılıyor.")
                            bt_default = run_strategy_backtest(df_view, initial_balance=10000, timeframe=view_tf, weights=None)
                            if bt_default is not None:
                                comp_col1, comp_col2 = st.columns(2)
                                with comp_col1:
                                    st.markdown("**Varsayılan Ağırlıklar**")
                                    st.metric("Toplam Getiri", f"%{bt_default['total_return']:.2f}")
                                    st.metric("Kazanma Oranı", f"%{bt_default['win_rate']:.1f}")
                                with comp_col2:
                                    st.markdown("**Ayarlanmış Ağırlıklar**")
                                    st.metric("Toplam Getiri", f"%{bt_results['total_return']:.2f}",
                                               delta=f"{bt_results['total_return'] - bt_default['total_return']:+.2f}")
                                    st.metric("Kazanma Oranı", f"%{bt_results['win_rate']:.1f}",
                                               delta=f"{bt_results['win_rate'] - bt_default['win_rate']:+.1f}")
                                st.divider()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Toplam Getiri", f"%{bt_results['total_return']:.2f}")
                        col2.metric("Kazanma Oranı", f"%{bt_results['win_rate']:.1f}")
                        col3.metric("Toplam İşlem", bt_results['total_trades'])
                        col4.metric("Profit Factor", f"{bt_results['profit_factor']:.2f}")
                        st.divider()
                        col_det1, col_det2 = st.columns(2)
                        col_det1.write(f"✅ Kazanan İşlem: {bt_results['winning_trades']}")
                        col_det1.write(f"💰 Ort. Kazanç: ${bt_results['avg_win']:.2f}")
                        col_det2.write(f"❌ Kaybeden İşlem: {bt_results['losing_trades']}")
                        col_det2.write(f"💸 Ort. Kayıp: ${bt_results['avg_loss']:.2f}")
                        st.divider()
                        st.subheader("📈 Sermaye Eğrisi")
                        eq_df = pd.DataFrame(bt_results['equity_curve'])
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['equity'], mode='lines', name='Sermaye', line=dict(color='cyan', width=2)))
                        fig_eq.add_hline(y=10000, line_dash="dot", line_color="gray", annotation_text="Başlangıç")
                        fig_eq.update_layout(height=400, template="plotly_dark", hovermode='x unified', yaxis_title="Bakiye ($)", xaxis_title="Tarih")
                        st.plotly_chart(fig_eq, use_container_width=True)
                        with st.expander("📋 Tüm İşlemler"):
                            trades_df = pd.DataFrame(bt_results['trades'])
                            st.dataframe(trades_df, width="stretch")

    # --- KAĞIT TİCARET DOĞRULAMASI ---
    with st.expander("🧪 Kağıt Ticaret Doğrulaması (Canlı Sinyal Takibi)", expanded=False):
        st.info("Canlı sinyali (ML dahil) her gün kapanan mumda kaydeder ve backtest kurallarıyla "
                "sanal işlem yapar. Birkaç hafta sonra gerçek davranış ile backtest beklentisi "
                "karşılaştırılır. Günlük otomatik görev kuruluysa buton sadece kontrol içindir.")
        from paper_trading import run_paper_update, paper_report
        if st.button("📸 Bugünü Kaydet / Güncelle"):
            pp_bar = st.progress(0.0)
            pp_txt = st.empty()
            status = run_paper_update(st.session_state['coin_map'], src_pref,
                                      progress_callback=lambda p, n: (pp_bar.progress(min(1.0, p)), pp_txt.text(f"Kaydediliyor: {n}")))
            pp_bar.empty(); pp_txt.empty()
            if status["errors"]:
                st.warning(f"{status['new_rows']} yeni kayıt. Veri alınamayan: {', '.join(status['errors'])}")
            else:
                st.success(f"{status['new_rows']} yeni kayıt eklendi ({status['assets']} varlık).")
        paper_df, paper_totals = paper_report()
        if len(paper_df):
            st.dataframe(paper_df, width="stretch", hide_index=True)
            st.caption(f"Başlangıç: {paper_totals['başlangıç']} | Ort. getiri: %{paper_totals['toplam_getiri_pct']} "
                       f"| Al&Tut: %{paper_totals['al_tut_pct']} | Toplam kayıt: {paper_totals['kayıt']}")
        else:
            st.caption("Henüz kayıt yok — ilk kaydı almak için butona basın.")

    # --- AI PİYASA TARAYICI ---
    render_opportunity_scanner(st.session_state['coin_map'], src_pref, intervals)

    # --- PORTFÖY VE CÜZDAN YÖNETİMİ ---
    st.divider()
    col_risk, col_wallet = st.columns([1, 2])
    current_balance = st.session_state['portfolio_data'].get('balance', 0.0)

    with col_risk:
        st.subheader("🧮 Emir Gir")
        entry_price = st.number_input("Giriş Fiyatı ($)", value=float(curr), step=0.01, format="%.4f")
        investment = st.number_input("İşlem Tutarı ($)", value=1000.0, step=100.0)
        is_limit = st.checkbox("⏳ Limit Emir", value=False)
        use_balance = st.checkbox(f"🏦 Bakiyeden Kullan (${current_balance:,.2f})", value=True)
        atr_val = current_atr if 'current_atr' in locals() else entry_price*0.02
        st.caption(f"Stop Önerisi: ${(entry_price - atr_val * 1.5):.2f}")

        if st.button("➕ Emri Gir / Ekle"):
            is_valid, risk_msg = validate_portfolio_risk(investment, current_balance, st.session_state['portfolio_data']['positions'])
            if not is_valid: st.error(risk_msg)
            else:
                proceed = True
                if use_balance:
                    if investment > current_balance: st.error("Yetersiz Bakiye! Lütfen Bakiye Düzenle kısmından para ekleyin."); proceed = False
                    else: st.session_state['portfolio_data']['balance'] -= investment
                
                if proceed:
                    st.session_state['portfolio_data']['positions'].append({
                        "Coin": sel_c, "Giriş": entry_price, "Adet": investment / entry_price,
                        "Yatırım": investment, "Realized": 0.0, "Status": "PENDING" if is_limit else "ACTIVE", "Tarih": time.strftime("%Y-%m-%d")
                    })
                    save_portfolio(st.session_state['portfolio_data'])
                    st.success("Limit Emir Girildi! Fiyat bekleniyor..." if is_limit else "Pozisyon Açıldı!")
                    time.sleep(1)
                    st.rerun()

        st.write("---") 
        with st.expander("💳 Cüzdan Bakiyesi Düzenle"):
            new_balance_input = st.number_input("Güncel USDT Bakiyesi", value=float(current_balance), step=100.0)
            if st.button("Bakiyeyi Güncelle"):
                st.session_state['portfolio_data']['balance'] = new_balance_input
                save_portfolio(st.session_state['portfolio_data'])
                st.success("Bakiye güncellendi!"); time.sleep(0.5); st.rerun()

    with col_wallet:
        st.subheader("💰 Varlıklarım")
        positions = st.session_state['portfolio_data']['positions']
        total_active_value = 0.0 
        
        if positions:
            active_pos = [p for p in positions if p.get('Status', 'ACTIVE') == 'ACTIVE']
            pending_pos = [p for p in positions if p.get('Status') == 'PENDING']
            
            if active_pos:
                st.markdown("##### ✅ Aktif Pozisyonlar")
                with st.expander("💸 Kar Al / Satış Yap"):
                    p_coins = list(set([p['Coin'] for p in active_pos]))
                    s_coin = st.selectbox("Coin", p_coins, key="sell_sel")
                    target_pos = next((p for p in active_pos if p['Coin'] == s_coin), None)
                    if target_pos:
                        sell_price = st.number_input("Satış Fiyatı", value=float(curr if s_coin == sel_c else target_pos['Giriş']))
                        sell_pct = st.slider("Satış %", 0, 100, 50)
                        sell_amt = target_pos['Adet'] * (sell_pct / 100)
                        total_return = sell_amt * sell_price
                        st.write(f"**Gelecek Nakit:** ${total_return:,.2f}")
                        if st.button("Satışı Onayla"):
                            st.session_state['portfolio_data']['balance'] += total_return
                            cost_basis = float(target_pos.get('Giriş', 0.0)) * float(sell_amt)
                            target_pos['Adet'] = float(target_pos.get('Adet', 0.0)) - float(sell_amt)
                            target_pos['Yatırım'] = float(target_pos.get('Yatırım', 0.0)) - cost_basis
                            target_pos['Realized'] = float(target_pos.get('Realized', 0.0)) + float(total_return - cost_basis)
                            save_portfolio(st.session_state['portfolio_data'])
                            st.success("Satış gerçekleşti!")
                            st.rerun()

                active_data = []
                for item in active_pos:
                    if item['Adet'] > 0:
                        lp = curr if item['Coin'] == sel_c else get_live_price_for_portfolio(item['Coin'], st.session_state['coin_map'])
                        if lp == 0: lp = item['Giriş']
                        val = float(str(item.get('Adet', '0.0'))) * float(str(lp))
                        total_active_value = float(str(total_active_value)) + val
                        active_data.append({
                            "Coin": item['Coin'], "Giriş": item['Giriş'], "Adet": item['Adet'],
                            "Değer ($)": val, "Kar/Zarar ($)": val - item['Yatırım'], "Kar/Zarar (%)": f"%{((val - item['Yatırım']) / item['Yatırım']) * 100:.2f}"
                        })
                if active_data: st.dataframe(pd.DataFrame(active_data), width="stretch")

            if pending_pos:
                st.markdown("##### ⏳ Bekleyen Limit Emirler")
                pending_data = []
                for item in pending_pos:
                    lp = curr if item['Coin'] == sel_c else get_live_price_for_portfolio(item['Coin'], st.session_state['coin_map'])
                    pending_data.append({
                        "Coin": item['Coin'], "Hedef Giriş": item['Giriş'], "Anlık Fiyat": lp,
                        "Uzaklık (%)": f"%{((lp - item['Giriş']) / max(lp, 0.001)) * 100:.2f}", "Kilitli Tutar": item['Yatırım']
                    })
                st.dataframe(pd.DataFrame(pending_data), width="stretch")
                
                with st.expander("🛠️ Emri Yönet"):
                    p_opts = [f"{p['Coin']} - ${p['Giriş']}" for p in pending_pos]
                    selected_opt = st.selectbox("İşlem Yapılacak Emir", p_opts)
                    sel_coin_name, sel_price_val = selected_opt.split(" - ")[0], float(selected_opt.split(" - $")[1])
                    target_pending = next((p for p in pending_pos if p['Coin'] == sel_coin_name and abs(p['Giriş'] - sel_price_val) < 0.0001), None)
                    
                    if target_pending:
                        c_man1, c_man2, c_man3 = st.columns(3)
                        with c_man1:
                            if st.button("❌ İptal Et"):
                                st.session_state['portfolio_data']['balance'] += target_pending['Yatırım']
                                st.session_state['portfolio_data']['positions'].remove(target_pending)
                                save_portfolio(st.session_state['portfolio_data'])
                                st.success("Emir iptal edildi.")
                                time.sleep(1); st.rerun()
                        with c_man2:
                            new_limit_price = st.number_input("Yeni Hedef Fiyat", value=float(target_pending['Giriş']), format="%.4f")
                            if st.button("✏️ Güncelle") and new_limit_price > 0:
                                target_pending['Giriş'] = new_limit_price
                                target_pending['Adet'] = target_pending['Yatırım'] / new_limit_price
                                save_portfolio(st.session_state['portfolio_data'])
                                st.success("Fiyat güncellendi.")
                                time.sleep(1); st.rerun()
                        with c_man3:
                            if st.button("🚀 Başlat"):
                                target_pending['Status'] = 'ACTIVE'; save_portfolio(st.session_state['portfolio_data']); st.rerun()

            st.divider()
            total_equity = current_balance + total_active_value + sum([p['Yatırım'] for p in pending_pos])
            m1, m2, m3 = st.columns(3)
            m1.metric("Boştaki USDT", f"${current_balance:,.2f}")
            m2.metric("Aktif Pozisyonlar", f"${total_active_value:,.2f}")
            m3.metric("🏆 TOPLAM VARLIK", f"${total_equity:,.2f}")
            if st.button("🗑️ Portföyü Sıfırla"):
                st.session_state['portfolio_data'] = {"balance": 1000.0, "positions": []}; save_portfolio(st.session_state['portfolio_data']); st.rerun()
        else:
            st.info("Portföy boş.")
            st.metric("Mevcut Bakiye", f"${current_balance:,.2f}")
else: st.error("Veri Alınamadı.")

# OTOMATİK KAPATMA / TELEGRAM
if st.session_state.get('portfolio_data'):
    closed_count, closed_trades = check_active_positions_auto_close(st.session_state['portfolio_data'], st.session_state['coin_map'])
    if closed_count > 0:
        st.toast(f"🔔 {closed_count} pozisyon otomatik kapandı!", icon="✅")
        for trade in closed_trades:
            emoji = "✅" if trade['profit'] > 0 else "❌"
            st.sidebar.success(f"{emoji} {trade['coin']}: {trade['type']} | ${trade['profit']:.2f} (%{trade['pct']:.1f})")

def send_tg(token, chat_id, msg):
    try:
        requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")

if auto or st.session_state.get('auto_mode', False):
    msg_str = ""
    for tf, label in intervals.items():
        cs = composite_signals.get(tf)
        if cs and ("AL" in cs.verdict or "SAT" in cs.verdict):
            msg_str += f"\n⏰ {label}: {cs.emoji} {cs.verdict} (Güven: %{cs.confidence:.0f})"
            if "AL" in cs.verdict and cs.entry_price > 0:
                msg_str += f"\n   🎯 TP: ${cs.take_profit_1:,.2f} | 🛑 SL: ${cs.stop_loss:,.2f}"
            elif "SAT" in cs.verdict:
                msg_str += "\n   ⚠️ Çıkış sinyali — short önerisi değildir"
    
    if msg_str and tg_token and tg_chat:
        full_msg = f"🚨 **{sel_c} BOT** 🚨\n{msg_str}\nFiyat: {curr:.2f}"
        if 'last_msg' not in st.session_state or st.session_state['last_msg'] != full_msg:
            send_tg(tg_token, tg_chat, full_msg)
            st.session_state['last_msg'] = full_msg
    # Re-check on the market-data cache's own TTL instead of sleeping for
    # hours: a multi-hour sleep() blocks this session's server thread the
    # entire time (no reruns, no UI responsiveness, likely killed by a
    # reverse-proxy/browser idle timeout long before it completes). Sleeping
    # only as long as the cached data stays fresh (DataFetchConfig.CACHE_TTL)
    # keeps the bot checking signals promptly while each blocking window
    # stays short enough that the app remains usable between checks.
    time.sleep(DataFetchConfig.CACHE_TTL)
    st.rerun()

