import streamlit as st
import pandas as pd
import time
import requests
import traceback
from config import DEFAULT_COIN_MAP, PATTERN_INFO, UIConfig, FileConfig
from logger import logger

# === YENİ MODÜLLERDEN IMPORTLAR ===
from data_fetchers import get_market_data, get_fear_greed_index
from technical_analysis import calculate_sr_advanced, detect_advanced_patterns, calculate_oracle_signal_v2, calculate_trade_setup, calculate_extended_trendlines, detect_patterns
from ml_models import calculate_smart_prediction_FIXED
from portfolio import load_portfolio, save_portfolio, validate_portfolio_risk, check_active_positions_auto_close, multi_timeframe_confirmation
from ui_components import render_sidebar_settings, render_asset_management, render_main_chart
from scanner import render_opportunity_scanner
from data_fetchers import get_live_price_for_portfolio

st.set_page_config(layout="wide", page_title="Pro Trader V48 (Modular Edition)")

# CSS
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { font-size: 2rem !important; margin-bottom: 0rem; }
        .stMarkdown p { font-size: 14px; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
        .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
        .stExpander { border: 1px solid #333; border-radius: 8px; }
        hr { margin: 1em 0; border-color: #333; }
        .trade-card { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border-left: 4px solid; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# GİRİŞ VERİLERİNİ YÜKLE
def load_assets():
    f = FileConfig.ASSETS_FILE
    import os, json
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                return json.load(file)
        except: return DEFAULT_COIN_MAP.copy()
    return DEFAULT_COIN_MAP.copy()

def save_assets(data):
    f = FileConfig.ASSETS_FILE
    import json
    with open(f, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if 'coin_map' not in st.session_state:
    st.session_state['coin_map'] = load_assets()

if 'portfolio_data' not in st.session_state:
    st.session_state['portfolio_data'] = load_portfolio()

# --- ARAYÜZ (SIDEBAR) ---
tg_token, tg_chat = render_sidebar_settings()
render_asset_management(st.session_state['coin_map'], save_assets)
render_opportunity_scanner(st.session_state['coin_map'], src_pref)

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
except: pass

intervals = {"4h": "4 Saatlik", "1d": "Günlük", "1wk": "Haftalık"}
results = {}
active_src = ""

# --- ANA VERİ DÖNGÜSÜ ---
for tf, label in intervals.items():
    df, src = get_market_data(src_pref, symbol, tf)
    if tf == "1d": active_src = src
    results[tf] = df
    
    if df is not None:
        s_list, r_list = calculate_sr_advanced(df, tf)
        status, color, target_msg = calculate_oracle_signal_v2(df, s_list, r_list)
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {label}")
        st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{status}</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"{target_msg}")
        adv_patterns = detect_advanced_patterns(df)
    
        if adv_patterns:
            st.sidebar.markdown("**🔍 Tespit Edilen Formasyonlar:**")
            for pat in adv_patterns:
                emoji_dir = "🟢" if pat['direction'] == 'BULLISH' else ("🔴" if pat['direction'] == 'BEARISH' else "⚪")
                st.sidebar.caption(f"{emoji_dir} {pat['name']} (Güven: %{pat['confidence']})")
    else: 
        st.sidebar.warning(f"{label}: Bekleniyor...")

st.title(f"📈 {sel_c} V48 (Modular & Re-Architected)")
safe_src = str(active_src) if active_src else ""
c = "green" if "Binance" in safe_src else ("blue" if "OKX" in safe_src else "orange")
st.markdown(f"**Veri Kaynağı:** <span style='color:{c}; font-weight:bold'>{safe_src}</span>", unsafe_allow_html=True)

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

    # --- ALT PANELLER (Analiz & Yönetim) ---
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.info("### 🧠 Tespitler")
        if show_all_pats and items_raw:
            visible_names = []
            for item in items_raw:
                if (item['type'] == 'box' and f_wm) or (item['type'] == 'icon' and f_candle): visible_names.append(item['name'])
            if visible_names:
                for p in list(set(visible_names)): st.write(PATTERN_INFO.get(p, p))
            else: st.write("Filtreli formasyon yok.")
        else: st.write("Formasyonlar kapalı.")
        
    with col2:
        st.warning("### 📊 Piyasa Özeti")
        trend = "YÜKSELİŞ" if curr > df_view['EMA_50'].iloc[-1] else "DÜŞÜŞ"
        prices_list = list(f_prices) if f_prices else []
        pred_dir = "YUKARI ↗️" if show_pred and len(prices_list) > 0 and float(prices_list[-1]) > curr else "AŞAĞI ↘️"
        st.metric("Trend (EMA50)", trend)
        st.metric("Tahmin", pred_dir)
        st.metric("RSI", f"{df_view['RSI'].iloc[-1]:.1f}")

    with col3:
        st.success("### 🎯 AI Stratejisi")
        signal_status, color, target_msg = calculate_oracle_signal_v2(df_view, s_l, r_l)
        is_uptrend = curr > df_view['EMA_50'].iloc[-1]
        trend_note = ""
        if "AL" in signal_status:
            if is_uptrend: signal_status, trend_note = "GÜÇLÜ AL (Trend Yönünde) 🚀", "Trend seninle, güvenli işlem."
            else: signal_status, trend_note, color = "TEPKİ ALIMI (Riskli) ⚠️", "Trend Düşüşte! Sadece kısa vadeli tepki (Scalp).", "orange"
        elif "SAT" in signal_status:
            if not is_uptrend: signal_status, trend_note = "GÜÇLÜ SAT (Trend Yönünde) 🔻", "Trend aşağı, düşüş derinleşebilir."
            else: signal_status, trend_note = "DÜZELTME SATIŞI (Riskli) ⚠️", "Trend Yükselişte! Fiyat sadece dinleniyor olabilir."
        else:
            signal_status, trend_note, color, target_msg = "NÖTR (BEKLE) 💤", "Piyasa kararsız veya yatay. İşlem yapma, izle.", "gray", "Yön Belirsiz"

        st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:20px'>{signal_status}</span>", unsafe_allow_html=True)
        st.caption(trend_note)
        st.write(f"**{target_msg}**")
        
        if "AL" in signal_status or "SAT" in signal_status:
            setup = calculate_trade_setup(df_view, "AL" if "AL" in signal_status else "SAT")
            if setup:
                st.divider()
                st.write(f"**Giriş:** ${setup['entry']:,.2f} | **🛑 Stop:** ${setup['sl']:,.2f} | **🎯 TP:** ${setup['tp']:,.2f}")
        else: st.info("Setup oluşmadı. Güvenli bölge bekleniyor.")

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
                if active_data: st.dataframe(pd.DataFrame(active_data), use_container_width=True)

            if pending_pos:
                st.markdown("##### ⏳ Bekleyen Limit Emirler")
                pending_data = []
                for item in pending_pos:
                    lp = curr if item['Coin'] == sel_c else get_live_price_for_portfolio(item['Coin'], st.session_state['coin_map'])
                    pending_data.append({
                        "Coin": item['Coin'], "Hedef Giriş": item['Giriş'], "Anlık Fiyat": lp,
                        "Uzaklık (%)": f"%{((lp - item['Giriş']) / max(lp, 0.001)) * 100:.2f}", "Kilitli Tutar": item['Yatırım']
                    })
                st.dataframe(pd.DataFrame(pending_data), use_container_width=True)
                
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
    try: requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
    except: pass

if auto or st.session_state.get('auto_mode', False):
    msg_str = ""
    for tf, res in results.items():
        if res is not None:
            s_l, r_l = calculate_sr_advanced(res, tf)
            stat, _, target = calculate_oracle_signal_v2(res, s_l, r_l)
            if "GÜÇLÜ" in stat or "AL" in stat:
                msg_str = str(msg_str) + f"\n⏰ {tf}: {stat} | {target}"
    
    if msg_str and tg_token and tg_chat:
        full_msg = f"🚨 **{sel_c} BOT** 🚨\n{msg_str}\nFiyat: {curr:.2f}"
        if 'last_msg' not in st.session_state or st.session_state['last_msg'] != full_msg:
            send_tg(tg_token, tg_chat, full_msg)
            st.session_state['last_msg'] = full_msg
    time.sleep(14400) 
    st.rerun()
