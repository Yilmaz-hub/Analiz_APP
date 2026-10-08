import streamlit as st
import pandas as pd
import copy
import time
import requests
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo
from config import PATTERN_INFO, UIConfig, FileConfig, DataFetchConfig
import storage
from storage import StorageAccessError
from assets import load_assets, save_assets, resolve_name
from logger import logger

# === YENİ MODÜLLERDEN IMPORTLAR ===
from data_fetchers import get_market_data, get_fear_greed_index
from technical_analysis import (detect_advanced_patterns, calculate_extended_trendlines,
                                detect_patterns, run_strategy_backtest,
                                run_v1_strategy_backtest, build_v1_decisions)
from ml_models import calculate_smart_prediction_FIXED
from portfolio import (load_portfolio, save_portfolio, reset_portfolio, validate_portfolio_risk,
                       check_active_positions_auto_close, multi_timeframe_confirmation)
from ui_components import (render_sidebar_settings, render_asset_management, render_main_chart,
                           is_chart_renderable, is_mobile_mode, is_empty_data_reason,
                           render_records_status, render_no_records_state, render_trade_settings)
from scanner import render_opportunity_scanner
from data_fetchers import fetch_prices, get_live_price_for_portfolio
from signal_engine import generate_stable_signal, generate_validated_signal, CompositeSignal, invalid_data_signal
from market_validation import is_v1_scope, policy_for_symbol, validate_market_data
from positions import (total_value_note, group_positions, build_active_rows, asset_name as position_asset_name,
                       AKTIF as POS_AKTIF, BEKLEYEN as POS_BEKLEYEN,
                       SORUNLU as POS_SORUNLU)
from weight_profiles import get_weights_for_symbol
from prediction_tracker import record_prediction, evaluate_predictions, get_track_record
from advanced_analysis import detect_elliott_wave, analyze_ichimoku, detect_wyckoff_phase, analyze_market_structure
from position_journal import PositionJournal
from trade_decisions import Position, PositionState, Signal
from trade_decisions import initial_stop
from trade_confirmation import (confirm_buy, confirm_sell, istanbul_to_utc, quantity_from_percent,
                                reconcile)
from trade_settings import known_quantity_step
from trading_ui import (PanelInput, bars_since_loss_exit, build_decision_panel, describe_blocked, describe_code,
                        format_decision_time, resolve_decision_time)
import theme

st.set_page_config(layout="wide", page_title="Pro Trader V48 (Modular Edition)")

# CSS
theme.inject()

# KAYITLARI YÜKLE (spec 0004)
# Kayıtlar dosyadan değil, ortamdan bağımsız belge deposundan okunur. Erişim
# sorununda varsayılan listeye DÜŞÜLMEZ: kullanıcıya durum bildirilir ve kayıt
# değiştiren işlemler gösterilmez (R8.1 / AC16 / AC16b).
if 'records_loaded' not in st.session_state:
    try:
        storage.import_legacy_documents()
        st.session_state['coin_map'] = load_assets()
        st.session_state['portfolio_data'] = load_portfolio()
        st.session_state['portfolio_snapshot'] = copy.deepcopy(st.session_state['portfolio_data'])
        st.session_state['storage_ok'] = True
        st.session_state['storage_msg'] = storage.check_access()[1]
    except StorageAccessError as e:
        logger.error(f"Records unavailable at startup: {e}")
        st.session_state['coin_map'] = {}
        st.session_state['portfolio_data'] = None
        st.session_state['portfolio_snapshot'] = None
        st.session_state['storage_ok'] = False
        st.session_state['storage_msg'] = "Kayıtlara şu anda erişilemiyor; değişiklik yapılamaz."
    st.session_state['records_loaded'] = True


def safe_save_portfolio():
    """Portföyü yazar; başarısızsa bellekteki değişikliği **geri alır**.

    Akışlar önce bellekteki portföyü değiştirip sonra yazıyor. Yazma koparsa
    kullanıcıya "işlem kaydedilmedi" denip bellek değişmiş bırakılırsa, sonraki
    başarılı yazma o işlemi kalıcılaştırıyordu (QA F1). Bu yüzden başarısızlıkta
    son bilinen kayıtlı duruma dönülür.

    Teknik hata metni kullanıcıya sızmaz.
    """
    try:
        save_portfolio(st.session_state['portfolio_data'])
    except StorageAccessError as e:
        logger.error(f"Portfolio save blocked: {e}")
        snapshot = st.session_state.get('portfolio_snapshot')
        if snapshot is not None:
            st.session_state['portfolio_data'] = copy.deepcopy(snapshot)
        st.session_state['storage_ok'] = False
        st.error("Kayıtlara şu anda erişilemiyor; işlem kaydedilmedi.")
        return False
    remember_saved_portfolio()
    return True


def remember_saved_portfolio():
    """Bellekteki portföyü "son kayıtlı durum" olarak işaretler.

    Geri alma bu anlık görüntüye döner. Portföyü **diske yazan her yol** bunu
    çağırmak zorundadır; otomatik kapatma kendi içinden yazdığı için (bkz.
    `portfolio.check_active_positions_auto_close`) anlık görüntü bayat kalıyor
    ve ilk başarısız yazmada meşru bir kapanış geri alınıyordu (QA F10).
    """
    st.session_state['portfolio_snapshot'] = copy.deepcopy(st.session_state['portfolio_data'])


@st.fragment
def sell_panel(sel_c, curr, records_writable):
    """"Kâr Al / Satış Yap" paneli (spec 0006 + spec 0007 R06).

    Parça (fragment) olarak çalışır: satış fiyatını, yüzdeyi ya da miktarı yazmak yalnız bu paneli
    yeniden çalıştırır; grafik, sinyal ve fiyat çağrıları yeniden hesaplanmaz. Satış onaylanınca
    `st.rerun()` sayfayı bir kez tümüyle yeniler."""
    # Parça yeniden çalışırken argümanlar bayat kalır; pozisyonlar her seferinde güncel portföyden alınır
    # (yazma kopup portföy geri alınmış olabilir — spec 0007 AC11).
    active_pos = group_positions(st.session_state['portfolio_data']['positions'])[POS_AKTIF]
    p_coins = list(dict.fromkeys(p['Coin'] for p in active_pos))
    if not p_coins:
        st.info("Satılacak aktif pozisyon yok.")
        return
    s_coin = st.selectbox("Coin", p_coins, key="sell_sel")
    target_pos = next((p for p in active_pos if p['Coin'] == s_coin), None)
    if target_pos:
        sell_price = st.number_input("Satış Fiyatı", value=float(curr if s_coin == sel_c else target_pos['Giriş']), key=f"sell_price:{s_coin}")
        st.caption("V1 SAT sinyali ve stop çıkışı pozisyonun tamamını kapatır.")
        qty_key, pct_key = f"sell_qty:{s_coin}", f"sell_pct:{s_coin}"
        msg_key = f"sell_pct_msg:{s_coin}"
        held_qty = Decimal(str(target_pos['Adet']))
        try:
            sell_step = known_quantity_step(s_coin)
        except StorageAccessError:
            sell_step = None   # kayıtlar okunamıyor: satış zaten kapalı (records_writable)
        st.session_state.setdefault(qty_key, float(held_qty))
        st.session_state.setdefault(pct_key, 100.0)

        def apply_percent(percent=None, qty_key=qty_key, pct_key=pct_key,
                          msg_key=msg_key, held_qty=held_qty, sell_step=sell_step):
            """Yüzdeyi satılacak miktara çevirir (spec 0006 R05); callback'te çalışır."""
            if percent is not None:
                st.session_state[pct_key] = float(percent)
            quantity, code = quantity_from_percent(
                held_qty, Decimal(str(st.session_state[pct_key])), sell_step)
            st.session_state[msg_key] = "" if quantity is not None else describe_code(code)
            if quantity is not None:
                st.session_state[qty_key] = float(quantity)

        preset_cols = st.columns(4)
        for column, preset in zip(preset_cols, (25, 50, 75, 100)):
            column.button(f"%{preset}", key=f"sell_preset:{s_coin}:{preset}",
                          on_click=apply_percent, args=(preset,))
        st.number_input("Yüzde (%)", min_value=0.0, max_value=100.0, step=1.0,
                        key=pct_key, on_change=apply_percent)
        if st.session_state.get(msg_key):
            st.warning(st.session_state[msg_key])
        sell_amt = st.number_input(
            "Satılan miktar (adet)", min_value=0.0,
            step=0.00000001, format="%.8f", key=qty_key)
        remaining_qty = held_qty - Decimal(str(sell_amt))
        st.caption(f"Kalan: {format(max(remaining_qty, Decimal('0')).normalize(), 'f')} adet")
        sell_now_tr = datetime.now(ZoneInfo("Europe/Istanbul"))
        sell_date = st.date_input("İşlem tarihi (İstanbul)", value=sell_now_tr.date(), key="sell_date")
        sell_time = st.time_input("İşlem saati (İstanbul)", value=sell_now_tr.time().replace(microsecond=0), key="sell_time")
        total_return = sell_amt * sell_price
        st.write(f"**Gelecek Nakit:** ${total_return:,.2f}")
        if st.button("Satışı Onayla", disabled=not records_writable):
            outcome = confirm_sell(
                st.session_state['portfolio_data'], st.session_state['position_journal'],
                safe_save_portfolio, position=target_pos,
                symbol=target_pos.get('Sembol') or st.session_state['coin_map'].get(s_coin, s_coin),
                quantity=sell_amt, price=sell_price,
                executed_at=istanbul_to_utc(sell_date, sell_time),
                now=datetime.now(timezone.utc),
            )
            if outcome.ok:
                if target_pos.get('Status') == 'CLOSED_CONFIRMED':
                    st.session_state[f'flat_confirmed:{s_coin}'] = True
                for stale in (qty_key, pct_key, msg_key):
                    st.session_state.pop(stale, None)
                st.success("Satış gerçekleşti!")
                st.rerun()
            elif outcome.code != "KAYIT_YAZILAMADI":
                st.error(describe_code(outcome.code))


_RUN_PRICES = {}    # betik her çalıştırmada baştan işlendiği için bu sözlük çalıştırma başına tazedir


def position_coin_map(positions):
    """Pozisyondaki ad -> sembol. Önce kaydın kendi `Sembol` alanı (varlık listeden silinse ya da yeniden
    adlandırılsa da fiyat gelir), yoksa ad listedeki varlığa eşlenir (spec 0010/0011); eşleşme yoksa ad
    haritada olmaz ve satır "bulunamadı" der."""
    coin_map = st.session_state.get('coin_map', {})
    resolved = {}
    for pos in positions:
        coin = pos['Coin']
        own = pos.get('Sembol')
        if isinstance(own, str) and own.strip():
            resolved.setdefault(coin, own.strip())
            continue
        key = resolve_name(coin_map, coin)
        if key is not None:
            resolved.setdefault(coin, coin_map[key])
    return resolved


def run_position_prices():
    """Bu çalıştırmada tüm pozisyon fiyatları: eşzamanlı ve süre bütçeli (spec 0007 R03/R04).

    Tablolar seçili varlık için grafik fiyatını (`curr`) kullanır; stop teması uyarısı ise her
    varlıkta canlı fiyata bakar. Alınamayan fiyat 0'dır ve ekranda "fiyat alınamadı" görünür."""
    if 'prices' not in _RUN_PRICES:
        positions = (st.session_state.get('portfolio_data') or {}).get('positions', [])
        # Tabloya giren her kayıt fiyatlanır: aktiflik yorumu tek sınıflandırıcıdan gelir (Status alanı olmayan
        # eski kayıtlar ve kalan miktarı olan kayıtlar dahil — spec 0009 AC09).
        groups = group_positions(positions)
        listed = groups[POS_AKTIF] + groups[POS_BEKLEYEN]
        coins = list(dict.fromkeys(p['Coin'] for p in listed))
        _RUN_PRICES['prices'] = fetch_prices(coins, position_coin_map(listed))
    return _RUN_PRICES['prices']


# --- F2: kayıt yazan kontroller erişim durumuna bağlıdır --------------------
# Erişim oturumun ortasında koparsa sayfa durmaz; bu yüzden her çalıştırmada
# durum yeniden ölçülür ve yazan kontroller kapatılır.
if st.session_state.get('records_loaded'):
    _erisim_ok, _erisim_msg = storage.check_access()
    st.session_state['storage_ok'] = _erisim_ok
    st.session_state['storage_msg'] = _erisim_msg
records_writable = st.session_state.get('storage_ok', True)

# Spec 0003'ün işlem günlüğü; kayıtları kendi dosyasında tutar.
# Günlük kalıcı depoda tutulur (Q13); okunamazsa boş günlükle devam edilmez, işlem
# günlüğü gerektiren kontroller kapanır.
journal_error = None
if st.session_state.get('position_journal') is None:
    try:
        st.session_state['position_journal'] = PositionJournal()
    except StorageAccessError as e:
        logger.error(f"Position journal unavailable: {e}")
        st.session_state['position_journal'] = None
        journal_error = "İşlem günlüğü okunamadı; işlem teyidi kapalı. Kayıtlarınız silinmedi."
journal_ready = st.session_state.get('position_journal') is not None
records_writable = records_writable and journal_ready
# Portföy yazıldı ama günlük yazımı koptuysa yarım kalan işlem bir kereliğine tamamlanır (Q3).
if journal_ready and st.session_state.get('portfolio_data') and not st.session_state.get('journal_reconciled'):
    try:
        reconcile(st.session_state['portfolio_data'], st.session_state['position_journal'],
                  st.session_state.get('coin_map', {}))
        st.session_state['journal_reconciled'] = True
    except StorageAccessError as e:
        logger.error(f"Journal reconcile blocked: {e}")

# --- ARAYÜZ (SIDEBAR) ---
tg_token, tg_chat = render_sidebar_settings()
render_records_status(st.session_state.get('storage_ok', True), st.session_state.get('storage_msg', ''))
if journal_error:
    st.sidebar.error(journal_error)
render_asset_management(st.session_state['coin_map'], st.session_state['portfolio_data'],
                        st.session_state.get('storage_ok', True))

st.sidebar.divider()
current_assets = list(st.session_state['coin_map'].keys())
src_pref = st.sidebar.radio("📡 Kaynak:", ["Binance", "OKX", "Yahoo Finance"])

# Kayıt yoksa uydurma varlık gösterilmez: gerçekten boş olan liste ile
# erişilemeyen liste ayrı durumlardır ve boş durumda kendiliğinden kayıt
# oluşmaz (spec 0004, R8 / AC07).
if not current_assets:
    render_no_records_state(st.session_state.get('storage_ok', True),
                            st.session_state.get('storage_msg', ''))
    st.stop()

sel_c = st.sidebar.selectbox("Enstrüman:", current_assets)
symbol = st.session_state['coin_map'].get(sel_c)

# Seçilen varlığın karşılığı yoksa başka bir varlığın bilgileri gösterilmez
# (spec 0004, R8.2); eski sürüm sessizce BTC verisine düşüyordu.
if symbol is None:
    st.warning(f"'{sel_c}' adlı varlık kayıtlarda bulunamadı. Lütfen listeden başka bir varlık seçin.")
    st.stop()

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
reasons = {}  # per-tf reason string when there is no frame (empty vs error)
active_src = ""

# --- ANA VERİ DÖNGÜSÜ ---
composite_signals = {}  # Store composite signals per timeframe
validation_statuses = {}
for tf, label in intervals.items():
    df, src = get_market_data(src_pref, symbol, tf)
    if tf == "1d": active_src = src
    results[tf] = df
    reasons[tf] = src
    
    if df is not None:
        # Tuned weights only apply to the daily signal — the optimizer
        # only ever validates against daily data (see weight_profiles.py).
        sig_weights = get_weights_for_symbol(symbol) if tf == "1d" else None
        # Stable (whipsaw-filtered) composite signal — closed candles only
        if tf == "1d" and not is_v1_scope(symbol):
            # V1 kapsamı dışı (ör. döviz): bileşen denetimi uygulanmaz, ekranda belirtilir.
            validation_statuses[tf] = "V1_DOGRULANMADI"
            comp_sig = generate_stable_signal(df, tf, weights=sig_weights)
        elif tf == "1d":
            evaluation_time = datetime.now(timezone.utc)
            policy = policy_for_symbol(symbol, evaluation_time)
            validation = validate_market_data(
                df, evaluation_time, policy,
                provider_open=df.attrs.get("provider_open", set()),
                require_components=True,
            )
            validation_statuses[tf] = validation.status
            comp_sig = generate_validated_signal(
                df, evaluation_time, policy,
                provider_open=df.attrs.get("provider_open", set()),
                timeframe=tf, weights=sig_weights,
            )
            if comp_sig is None:
                comp_sig = invalid_data_signal(
                    validation.status, tf, validation.missing_components)
            elif comp_sig.unavailable_components:
                validation_statuses[tf] = comp_sig.data_status
            # Karar zamanı yalnız veri geçerliyken ilerler; geçersizken son geçerli
            # karar saklı kalır ve "güncel değil" notuyla gösterilir (Q11).
            resolve_decision_time(
                st.session_state.setdefault("last_valid_decision", {}), symbol,
                validation_statuses[tf], evaluation_time)
        else:
            validation_statuses[tf] = "V1_DOGRULANMADI"
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

# Görünüm modu — grafiğin hemen üstünde (kenar çubuğu telefonda kapalı gelir).
# key='view_mode' seçimi rerun'lar arası korur; varsayılan Masaüstü.
# required=True: aktif segmente ikinci dokunuş seçimi düşürmesin -- yoksa mod
# None'a düşer ve grafik 900px'e dönüp ekrandan taşardı (spec 0001, QA BULGU-3).
view_mode = st.segmented_control(
    "🖥️ Görünüm",
    [UIConfig.VIEW_MODE_DESKTOP, UIConfig.VIEW_MODE_MOBILE],
    default=UIConfig.VIEW_MODE_DESKTOP,
    key="view_mode",
    required=True,
)
mobile_view = is_mobile_mode(view_mode)

if is_chart_renderable(df_view):
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
    
    s_l, r_l, adv_pattern_statuses = render_main_chart(df_view, view_tf, curr, f_dates, f_prices, ai_score, show_cloud, show_pred, show_ai, show_all_pats, f_wm, f_candle, f_advanced, items_raw, lines, mobile=mobile_view)

    # --- AI TAHMİN KARNESİ (tahmin hafızası + güven karnesi) ---
    if show_pred:
        if f_prices:
            record_prediction(symbol, view_tf, df_view, f_dates, f_prices, ai_score)
        pred_recs = evaluate_predictions(symbol, view_tf, df_view)
        track = get_track_record(pred_recs)
        with st.expander("🧠 AI Tahmin Karnesi — geçmiş tahminler ne dedi, ne çıktı?", expanded=False):
            if track["n"] > 0:
                k1, k2, k3 = st.columns(3)
                k1.metric("Olgunlaşan tahmin", track["n"])
                k2.metric("Yön isabeti", f"%{track['hit_rate']:.0f}")
                k3.metric("Ort. sapma", f"±%{track['avg_err']:.1f}")
                if track["hit_rate"] >= 60:
                    st.success("Bu varlık/periyotta AI'nin geçmiş yön isabeti iyi — tahmine ağırlık verilebilir.")
                elif track["hit_rate"] <= 45:
                    st.warning("Bu varlık/periyotta AI'nin geçmiş isabeti zayıf — tahmini tek başına kullanmayın.")
            else:
                st.caption("Henüz olgunlaşmış tahmin yok. Her tahmin, ufku (~15 bar) dolunca "
                           "otomatik puanlanır ve karne burada birikir.")
            if pred_recs:
                st.markdown("**Son tahminler** (yeniden eskiye):")
                for r in list(reversed(pred_recs))[:6]:
                    anchor_lbl = str(r["anchor"])[:16]
                    pred_pct = r["pred_change_pct"]
                    yon = "📈 yükseliş" if pred_pct >= 0 else "📉 düşüş"
                    line = f"`{anchor_lbl}` → {yon} **%{pred_pct:+.1f}** (skor {r['ai_score']:+.0f})"
                    out, prog = r.get("outcome"), r.get("progress")
                    if out:
                        isaret = "✅" if out["direction_hit"] else "❌"
                        line += f" — gerçekleşen **%{out['realized_change_pct']:+.1f}** {isaret}"
                    elif prog:
                        line += (f" — {prog['bars_elapsed']}/{r['horizon']} bar geçti, "
                                 f"şu ana dek %{prog['realized_change_pct']:+.1f}")
                    else:
                        line += " — henüz yeni"
                    st.markdown(line)

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
    
    matching_positions = [p for p in st.session_state['portfolio_data'].get('positions', [])
                          if p.get('Coin') == sel_c and p.get('Status') == 'ACTIVE']
    verified_position = next((p for p in matching_positions if p.get('V1Verified')), None)
    if verified_position:
        position = Position(
            PositionState.OPEN,
            Decimal(str(verified_position.get('Adet', 0))),
            Decimal(str(verified_position.get('Giriş', 0))),
            datetime.fromisoformat(verified_position['Gerçekleşme Zamanı']),
            Decimal(str(verified_position['Stop'])) if verified_position.get('Stop') else None,
        )
    elif matching_positions:
        position = Position(PositionState.UNKNOWN)
    elif st.session_state.get(f'flat_confirmed:{sel_c}', False):
        position = Position.flat()
    else:
        position = Position(PositionState.UNKNOWN)

    market_signal = Signal.BUY if "AL" in active_signal.verdict else (
        Signal.SELL if "SAT" in active_signal.verdict else Signal.WAIT
    )
    if view_tf == "1d":
        # Kullanıcı bir adayı aktif seçtiyse giriş sinyali onun kuralından gelir (spec 0005 R09, AC104).
        import candidate_ui
        try:
            market_signal, active_note = candidate_ui.active_entry_signal(df_view, market_signal)
            if active_note:
                st.caption(active_note)
            if verified_position:
                management_note = candidate_ui.management_note(
                    verified_position, candidate_ui.sc.active_strategy())
                if management_note:
                    st.caption(management_note)
        except StorageAccessError:
            st.caption("Aktif strateji kaydı okunamadı; V1 kuralı kullanıldı.")
    decision_panel = build_decision_panel(PanelInput(
        signal=market_signal, position=position,
        data_status=validation_statuses.get(view_tf, "V1_DOGRULANMADI"),
        fee=None, spread_bps=None, slippage_bps=None,
        confidence=Decimal(str(active_signal.confidence)),
        decision_at=st.session_state.get("last_valid_decision", {}).get(symbol),
        suggested_stop=Decimal(str(active_signal.stop_loss)) if active_signal.stop_loss else None,
        asset_kind=symbol if symbol in {"XAU_GOLD", "GRAM_TRY"} else None,
        in_scope=view_tf == "1d" and symbol != "GRAM_TRY" and is_v1_scope(symbol),
        current_price=Decimal(str(curr)),
        bars_since_loss_exit=bars_since_loss_exit(
            st.session_state['portfolio_data'].get('positions', []), sel_c,
            df_view.index, datetime.now(timezone.utc)) if view_tf == "1d" else None,
    ))
    if decision_panel.action == "TAMAMINI SAT" and decision_panel.exit_reason == "STOP":
        st.error(f"Pozisyonuna göre eylem: **{decision_panel.action}** — stop seviyesine temas edildi")
    elif decision_panel.action:
        st.success(f"Pozisyonuna göre eylem: **{decision_panel.action}**")
    else:
        st.warning("Pozisyon durumu teyit edilmeden kişisel işlem eylemi gösterilmez.")
    if position.state is PositionState.UNKNOWN and st.button("Bu varlıkta pozisyonum yok", key=f"flat:{sel_c}"):
        st.session_state[f'flat_confirmed:{sel_c}'] = True
        st.rerun()
    st.caption("Uyum puanı kazanma olasılığı değildir. Gerçek işlem yalnız kullanıcı teyidiyle değişir.")
    if decision_panel.old_decision_at is not None:
        st.caption(f"🕒 Son geçerli karar: {format_decision_time(decision_panel.old_decision_at)} "
                   "— güncel değil")
    elif active_signal.data_status != "GECERLI" or decision_panel.messages.count("ESKI_KARAR"):
        st.caption("🕒 Henüz geçerli bir karar üretilmedi — güncel değil")
    for warning in decision_panel.messages:
        st.caption(f"• {describe_code(warning)}")

    dash_col1, dash_col2, dash_col3 = st.columns([1.5, 1, 1.5])
    
    with dash_col1:
        if active_signal.data_status != "GECERLI":
            # Doğrulanamayan veri geçerli bir "BEKLE" gibi gösterilmez (Q5).
            st.warning("⚠️ Karar üretilemedi: " + describe_code(
                active_signal.data_status, active_signal.unavailable_components))
        else:
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

        if adv_pattern_statuses:
            st.markdown("---")
            st.markdown("**📐 Formasyon Durumu**")
            for msg in adv_pattern_statuses:
                st.caption(msg)

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


    # --- İŞLEM VARSAYIMLARI (geçmiş test + sanal takip ortak) ---
    paper_currency = "TRY" if symbol.endswith(".IS") or symbol == "GRAM_TRY" else "USD/USDT"
    trade_parsed = render_trade_settings(sel_c, symbol, paper_currency, records_writable, price=curr)

    # --- BACKTEST ---
    if df_view is not None:
        st.divider()
        with st.expander("📊 Backtest: Strateji Performansı", expanded=False):
            st.info("Günlük V1 testi; ML dahil kapanmış mum kararını sonraki açılışta uygular, 2,5 ATR başlangıç stopunu sabit tutar ve hedef/otomatik iz süren stopla satış yapmaz.")
            if st.button("🚀 Backtest Başlat", disabled=view_tf == "1d" and not trade_parsed.ok):
                with st.spinner("Backtest çalışıyor..."):
                    import plotly.graph_objects as go
                    tuned_weights = get_weights_for_symbol(symbol) if view_tf == "1d" else None

                    bt_progress = st.progress(0.0)
                    if view_tf == "1d":
                        v1_decisions = build_v1_decisions(df_view, weights=tuned_weights, include_ml=True)
                        bt_results = run_v1_strategy_backtest(
                            df_view, v1_decisions, initial_cash=trade_parsed.capital,
                            trade_notional=trade_parsed.settings.notional,
                            quantity_step=trade_parsed.settings.quantity_step,
                            costs=trade_parsed.settings.costs,
                        )
                    else:
                        st.warning("Bu zaman aralığı V1 kapsamı dışında; sonuç araştırma amaçlıdır.")
                        bt_results = run_strategy_backtest(
                            df_view, initial_balance=10000, timeframe=view_tf,
                            weights=tuned_weights,
                            progress_callback=lambda p: bt_progress.progress(min(1.0, p)),
                        )
                    bt_progress.empty()

                    if bt_results is None:
                        st.warning("Yeterli işlem oluşmadı. Daha uzun veri gerekebilir.")
                    else:
                        if view_tf == "1d":
                            if not bt_results["net_verified"]:
                                st.warning("Komisyon bilinmiyor: sonuç brüt modeldir, doğrulanmış net sonuç değildir.")
                            if not bt_results["spread_known"] or not bt_results["slippage_known"]:
                                st.caption("Makas veya kayma bilinmiyor; ilgili etki hesaplanmadı (sıfır maliyet değildir).")
                            for reason, count in bt_results["blocked"].items():
                                st.warning(describe_blocked(reason, count, trade_parsed.settings, paper_currency))
                        if tuned_weights and view_tf != "1d":
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
                        if view_tf == "1d":
                            import performance_ui as _perf
                            _perf.render_hold_comparison(_perf.hold_comparison(
                                df_view, bt_results, trade_parsed.capital, trade_parsed.settings.notional))
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
                        if view_tf == "1d":
                            st.session_state["perf_report_source"] = {
                                "symbol": symbol, "asset": sel_c, "backtest": bt_results, "frame": df_view,
                                "decisions": v1_decisions, "notional": trade_parsed.settings.notional,
                                "quantity_step": trade_parsed.settings.quantity_step,
                                "costs": trade_parsed.settings.costs,
                            }
            perf_source = st.session_state.get("perf_report_source")
            if view_tf == "1d" and perf_source is not None and perf_source["symbol"] == symbol:
                import performance_ui
                st.divider()
                st.subheader("📑 Güvenilir Performans Raporu")
                performance_ui.render_report_panel(perf_source)
                import candidate_ui
                candidate_ui.render_candidate_panel(perf_source)
                import risk_ui
                risk_ui.render_risk_panel(perf_source, st.session_state['portfolio_data'],
                                          st.session_state.get('coin_map', {}))
                import protection_ui
                from market_map import market_of as _market_of
                protection_ui.render_protection_panel(
                    perf_source, (_market_of(perf_source["symbol"]) or ("", "USD"))[1])

    # --- AI PİYASA TARAYICI ---
    render_opportunity_scanner(st.session_state['coin_map'], src_pref, intervals)

    # --- PORTFÖY VE CÜZDAN YÖNETİMİ ---
    st.divider()
    col_risk, col_wallet = st.columns([1, 2])
    current_balance = st.session_state['portfolio_data'].get('balance', 0.0)

    with col_risk:
        st.subheader("🧮 Emir Gir")
        entry_price = st.number_input("Giriş Fiyatı ($)", value=float(curr), step=0.01, format="%.4f", key=f"buy_price:{symbol}")
        buy_quantity = st.number_input(
            "Gerçekleşen miktar (adet)", min_value=0.0, value=round(1000.0 / float(curr), 8) if curr else 0.0,
            step=0.00000001, format="%.8f", key=f"buy_qty:{symbol}")
        investment = float(Decimal(str(buy_quantity)) * Decimal(str(entry_price)))
        st.caption(f"İşlem tutarı: ${investment:,.2f}")
        buy_now_tr = datetime.now(ZoneInfo("Europe/Istanbul"))
        buy_date = st.date_input("İşlem tarihi (İstanbul)", value=buy_now_tr.date(), key="buy_date")
        buy_time = st.time_input("İşlem saati (İstanbul)", value=buy_now_tr.time().replace(microsecond=0), key="buy_time")
        is_limit = st.checkbox("⏳ Limit Emir", value=False)
        use_balance = st.checkbox(f"🏦 Bakiyeden Kullan (${current_balance:,.2f})", value=True)
        atr_val = current_atr if 'current_atr' in locals() else entry_price*0.02
        # Başlangıç stopu ortak kuraldan gelir (AC83/AC110): pozitif değilse öneri yoktur.
        stop_default = initial_stop(Decimal(str(entry_price)), Decimal(str(atr_val)))
        stop_input = st.number_input(
            "Kuruma koyduğum stop", value=float(stop_default or Decimal("0")),
            step=0.01, format="%.4f",
        )
        if stop_default is None:
            st.caption("2,5 ATR başlangıç stopu pozitif çıkmıyor; kurumdaki stopu kendiniz belirleyin.")
        else:
            st.caption(f"2,5 ATR başlangıç stop önerisi: ${stop_default:.2f}")

        if st.button("➕ Emri Gir / Ekle", disabled=not records_writable):
            is_valid, risk_msg = validate_portfolio_risk(investment, current_balance, st.session_state['portfolio_data']['positions'])
            if not is_valid:
                st.error(risk_msg)
            else:
                # Doğrulama, ardından önce portföy sonra günlük (Q3): bakiye ve pozisyon
                # doğrulamadan önce değişmez; kimlik işlemin kendi alanlarından türer.
                outcome = confirm_buy(
                    st.session_state['portfolio_data'], st.session_state['position_journal'],
                    safe_save_portfolio, coin=sel_c, symbol=symbol, quantity=buy_quantity,
                    price=entry_price, stop=stop_input,
                    executed_at=istanbul_to_utc(buy_date, buy_time), now=datetime.now(timezone.utc),
                    use_balance=use_balance, is_limit=is_limit,
                )
                if outcome.ok:
                    st.session_state[f'flat_confirmed:{sel_c}'] = False
                    st.success("Limit Emir Girildi! Fiyat bekleniyor..." if is_limit else "Pozisyon Açıldı!")
                    time.sleep(1)
                    st.rerun()
                elif outcome.code != "KAYIT_YAZILAMADI":      # bu durumu safe_save_portfolio bildirir
                    st.error(describe_code(outcome.code))

        st.write("---") 
        with st.expander("💳 Cüzdan Bakiyesi Düzenle"):
            new_balance_input = st.number_input("Güncel USDT Bakiyesi", value=float(current_balance), step=100.0)
            if st.button("Bakiyeyi Güncelle", disabled=not records_writable):
                import cash_flows
                from datetime import datetime as _dtc, timezone as _tzc
                _flow = cash_flows.delta(current_balance, new_balance_input)
                _flow_saved = True
                if _flow != 0:
                    # Hareket önce kaydedilir: kaydedilemezse bakiye değişmez, AC14 engeli atlanmaz.
                    try:
                        _flow_at = cash_flows.record(_flow, _dtc.now(_tzc.utc), "elle bakiye güncelleme")
                    except StorageAccessError:
                        _flow_saved = False
                        st.error("Bakiye değiştirilmedi: nakit hareketi kaydedilemedi. "
                                 "Kayıt deposu erişimini kontrol edin.")
                if _flow_saved:
                    st.session_state['portfolio_data']['balance'] = new_balance_input
                    if safe_save_portfolio():
                        st.success("Bakiye güncellendi!"); time.sleep(0.5); st.rerun()
                    elif _flow != 0:
                        # Bakiye yazılamadı: kaydedilen hareketi ters kayıtla geri al (hayalet hareket kalmasın).
                        try:
                            cash_flows.record(-_flow, _dtc.now(_tzc.utc), "geri alma: bakiye yazılamadı",
                                              reverses=_flow_at)
                        except StorageAccessError:
                            st.warning("Nakit hareketi kaydı geri alınamadı; bu günün strateji karşılaştırması "
                                       "dışarıdan nakit hareketi nedeniyle uygun sayılmayabilir.")

    with col_wallet:
        st.subheader("💰 Varlıklarım")
        if not records_writable:
            st.error("⚠️ Kayıtlara şu anda erişilemiyor; kayıt değiştiren işlemler kapalı. "
                     "Kayıtlarınız silinmedi.")
        positions = st.session_state['portfolio_data']['positions']
        total_active_value = 0.0 
        
        if positions:
            # Aktiflik yorumu görünürlükte ve silme engelinde aynı kaynaktan
            # gelir (spec 0004, R4.1): tek sınıflandırıcı.
            groups = group_positions(positions)
            active_pos = groups[POS_AKTIF]
            pending_pos = groups[POS_BEKLEYEN]
            broken_pos = groups[POS_SORUNLU]

            if broken_pos:
                st.markdown("##### ⚠️ Sorunlu Kayıtlar")
                st.warning("Aşağıdaki kayıtların bilgileri okunamadı. Kayıtlar korunuyor, "
                           "silinmedi; ilgili varlıklar da silinemez.")
                st.dataframe(pd.DataFrame([
                    {"Coin": position_asset_name(item) or "(adı okunamadı)",
                     "Kayıt": str(item)[:120]}
                    for item in broken_pos
                ]), width="stretch")
            
            if active_pos:
                with st.expander("🛑 Stop Seviyesini Yükselt"):
                    stop_coins = list(dict.fromkeys(p['Coin'] for p in active_pos))
                    stop_coin = st.selectbox("Varlık", stop_coins, key="stop_update_asset")
                    stop_position = next(p for p in active_pos if p['Coin'] == stop_coin)
                    current_stop = float(stop_position.get('Stop') or 0.0)
                    raised_stop = st.number_input(
                        "Yeni stop", min_value=0.0, value=current_stop,
                        step=0.01, format="%.4f", key="raised_stop",
                    )
                    istanbul_now = datetime.now(ZoneInfo("Europe/Istanbul"))
                    effective_date = st.date_input(
                        "Geçerlilik tarihi", value=istanbul_now.date(), key="stop_effective_date"
                    )
                    effective_time = st.time_input(
                        "Geçerlilik saati", value=istanbul_now.time().replace(microsecond=0),
                        key="stop_effective_time",
                    )
                    if st.button("Stop Yükseltmesini Teyit Et", disabled=not records_writable):
                        if raised_stop <= current_stop:
                            st.error("Yeni stop mevcut stop seviyesinden yüksek olmalıdır.")
                        else:
                            effective_at = datetime.combine(
                                effective_date, effective_time, tzinfo=ZoneInfo("Europe/Istanbul")
                            ).astimezone(timezone.utc)
                            stop_position.setdefault('Stop Geçmişi', []).append({
                                "Önceki": current_stop, "Yeni": raised_stop,
                                "Geçerlilik Zamanı": effective_at.isoformat(),
                            })
                            stop_position['Stop'] = raised_stop
                            # Portföyü yazan her yol safe_save_portfolio'dan geçer: yazma
                            # koparsa bellekteki değişiklik geri alınır (spec 0004, F1/F10).
                            if safe_save_portfolio():
                                st.success("Stop yükseltmesi kaydedildi.")
                                st.rerun()
                st.markdown("##### ✅ Aktif Pozisyonlar")
                with st.expander("💸 Kar Al / Satış Yap"):
                    sell_panel(sel_c, curr, records_writable)

                run_prices = run_position_prices()
                active_data, total_active_value = build_active_rows(
                    active_pos,
                    lambda coin: curr if coin == sel_c else run_prices.get(coin, 0),
                )
                if active_data: st.dataframe(pd.DataFrame(active_data), width="stretch")
                price_note = total_value_note(active_data)
                if price_note:
                    st.warning(price_note)
                    with st.expander("Fiyat kaynağı ayrıntısı"):
                        from data_fetchers import price_diagnostics
                        for row in active_data:
                            if str(row.get("Fiyat", "")).startswith("fiyat alınamadı"):
                                row_symbol = position_coin_map(
                                    [p for p in active_pos if p['Coin'] == row["Coin"]]).get(row["Coin"], "")
                                details = price_diagnostics(row_symbol)
                                if not row_symbol:
                                    note = "varlık listesinde bulunamadı (varlık adı değişmiş ya da silinmiş olabilir)"
                                else:
                                    note = "; ".join(details) if details else "kaynaklara henüz sorulmadı"
                                st.caption(f"{row['Coin']}: {note}")

            if pending_pos:
                st.markdown("##### ⏳ Bekleyen Limit Emirler")
                pending_data = []
                for item in pending_pos:
                    lp = curr if item['Coin'] == sel_c else run_position_prices().get(item['Coin'], 0)
                    priced = bool(lp) and lp > 0
                    pending_data.append({
                        "Coin": item['Coin'], "Hedef Giriş": item['Giriş'],
                        "Anlık Fiyat": f"{lp:.2f}" if priced else "fiyat alınamadı",
                        "Uzaklık (%)": (f"%{((lp - item['Giriş']) / lp) * 100:.2f}" if priced
                                        else "hesaplanamıyor"),
                        "Kilitli Tutar": item['Yatırım']
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
                            if st.button("❌ İptal Et", disabled=not records_writable):
                                st.session_state['portfolio_data']['balance'] += target_pending['Yatırım']
                                st.session_state['portfolio_data']['positions'].remove(target_pending)
                                if safe_save_portfolio():
                                    st.success("Emir iptal edildi.")
                                    time.sleep(1); st.rerun()
                        with c_man2:
                            new_limit_price = st.number_input("Yeni Hedef Fiyat", value=float(target_pending['Giriş']), format="%.4f")
                            if st.button("✏️ Güncelle", disabled=not records_writable) and new_limit_price > 0:
                                target_pending['Giriş'] = new_limit_price
                                target_pending['Adet'] = round(target_pending['Yatırım'] / new_limit_price, 8)
                                if safe_save_portfolio():
                                    st.success("Fiyat güncellendi.")
                                    time.sleep(1); st.rerun()
                        with c_man3:
                            if st.button("🚀 Başlat", disabled=not records_writable):
                                target_pending['Status'] = 'ACTIVE'
                                if safe_save_portfolio():
                                    st.rerun()

            st.divider()
            total_equity = current_balance + total_active_value + sum([p['Yatırım'] for p in pending_pos])
            m1, m2, m3 = st.columns(3)
            m1.metric("Boştaki USDT", f"${current_balance:,.2f}")
            m2.metric("Aktif Pozisyonlar", f"${total_active_value:,.2f}")
            m3.metric("🏆 TOPLAM VARLIK", f"${total_equity:,.2f}")
            # Sıfırlama birden çok kaydı birden kaldırdığı için ayrıca onay ister
            # ve engelleyici kayıt varken çalışmaz (spec 0004, AC11d).
            reset_confirmed = st.checkbox("Portföyü sıfırlamayı onaylıyorum", key="pf_reset_ok",
                                          disabled=not records_writable)
            if st.button("🗑️ Portföyü Sıfırla", disabled=not records_writable):
                ok, msg, new_pf = reset_portfolio(st.session_state['portfolio_data'], reset_confirmed)
                if ok:
                    st.session_state['portfolio_data'] = new_pf
                    if safe_save_portfolio():
                        st.success(msg)
                        st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("Portföy boş.")
            st.metric("Mevcut Bakiye", f"${current_balance:,.2f}")
    # --- AĞIR PANELLER PORTFÖYDEN SONRA (spec 0009 AC10) ---
    # İleri takip ve kağıt ticaret ayrıntıları kayıt deposuna çok sorgu atar; portföy bunları beklemesin.
    # --- İLERİ DÖNEM SANAL TAKİP (spec 0005 Adım 7) ---
    with st.expander("📡 İleri Dönem Sanal Takip (ekran kapalıyken)", expanded=False):
        import forward_ui
        from datetime import datetime as _dt, timezone as _tz
        # Koşucunun sağlığı (son çalışma, bayatlık) iki ucuz sorguyla her zaman görünür (spec 0008 AC04);
        # ayrıntı, "göster" denmeden hesaplanmaz: uzak veritabanında her ekran çalıştırması gecikir (R02).
        forward_ui.render_forward_status(_dt.now(_tz.utc))
        if st.toggle("Ayrıntıları göster (takibi açıp kapatmaz)", key="forward_show"):
            forward_ui.render_forward_panel(_dt.now(_tz.utc))

    # --- KAĞIT TİCARET DOĞRULAMASI ---
    with st.expander("🧪 Kağıt Ticaret Doğrulaması (Canlı Sinyal Takibi)", expanded=False):
        st.info("Canlı sinyali (ML dahil) her gün kapanan mumda kaydeder ve backtest kurallarıyla "
                "sanal işlem yapar. Birkaç hafta sonra gerçek davranış ile backtest beklentisi "
                "karşılaştırılır. Günlük otomatik görev kuruluysa buton sadece kontrol içindir.")
        from paper_trading import run_paper_update, paper_report
        st.caption("Sermaye, tutar, adım ve maliyetler yukarıdaki **İşlem varsayımları** panelinden gelir.")
        if st.button("📸 Bugünü Kaydet / Güncelle", disabled=not trade_parsed.ok):
            pp_bar = st.progress(0.0)
            pp_txt = st.empty()
            paper_selection = {sel_c: symbol}
            paper_costs = trade_parsed.settings.costs
            paper_settings = {sel_c: {
                "capital": trade_parsed.capital,
                "trade_notional": trade_parsed.settings.notional,
                "quantity_step": trade_parsed.settings.quantity_step,
                "spread_bps": paper_costs.spread_bps,
                "slippage_bps": paper_costs.slippage_bps,
                "commission_pct": paper_costs.commission_pct,
                "currency": paper_currency,
            }}
            status = run_paper_update(
                paper_selection, src_pref, paper_settings=paper_settings,
                progress_callback=lambda p, n: (
                    pp_bar.progress(min(1.0, p)), pp_txt.text(f"Kaydediliyor: {n}")
                ),
            )
            pp_bar.empty(); pp_txt.empty()
            if status["errors"]:
                st.warning(f"{status['new_rows']} yeni kayıt. Güncellenemeyen: {', '.join(status['errors'])}")
            else:
                st.success(f"{status['new_rows']} yeni kayıt eklendi ({status['assets']} varlık).")
        paper_df, paper_totals = paper_report()
        if len(paper_df):
            st.dataframe(paper_df, width="stretch", hide_index=True)
            st.caption(f"Başlangıç: {paper_totals['başlangıç']} | Ort. getiri: %{paper_totals['toplam_getiri_pct']} "
                       f"| Al&Tut: %{paper_totals['al_tut_pct']} | Toplam kayıt: {paper_totals['kayıt']}")
        else:
            st.caption("Henüz kayıt yok — ilk kaydı almak için butona basın.")

else:
    # Boş durum (enstrümanda veri yok) ile hata durumu (ağ/işleme hatası)
    # ayrı gösterilir; kopan bağlantı boş enstrüman sanılmasın (spec 0001).
    reason = reasons.get(view_tf)
    if df_view is None and not is_empty_data_reason(reason):
        st.error("⚠️ Veri alınamadı. Lütfen bağlantıyı/kaynağı kontrol edip tekrar deneyin.")
    else:
        st.info("📭 Bu enstrüman için gösterilecek veri yok.")

# OTOMATİK KAPATMA / TELEGRAM
# Spec 0003 ile otomatik kapatma kaldırıldı: fiyat teması yalnız uyarı üretir,
# pozisyonu ve nakdi değiştirmez. Bu yol kayıtlara yazmadığı için portföyü
# yazan tek nokta safe_save_portfolio olarak kalır (spec 0004, QA F10).
if st.session_state.get('portfolio_data'):
    _, stop_alerts = check_active_positions_auto_close(
        st.session_state['portfolio_data'], st.session_state['coin_map'],
        prices=run_position_prices()
    )
    for trade in stop_alerts:
        st.sidebar.warning(
            f"🛑 {trade['coin']}: stop teması görüldü ({trade['observed_price']}). "
            "Gerçek satış yapıldıysa fiyat ve zamanı teyit edin."
        )

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

