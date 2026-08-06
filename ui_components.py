import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from config import PATTERN_INFO, DEFAULT_COIN_MAP, UIConfig
from technical_analysis import calculate_sr_advanced, detect_advanced_patterns, calculate_oracle_signal_v2, calculate_trade_setup
import theme

def render_sidebar_settings():
    st.sidebar.header("⚙️ Kontrol Paneli")
    with st.sidebar.expander("🔐 Bot & API Ayarları", expanded=False):
        st.caption("Telegram bildirimleri için gereklidir.")
        tg_token = st.text_input("Bot Token", value="", type="password")
        tg_chat = st.text_input("Chat ID", value="")
        st.caption("Bu ayarlar varsayılan olarak kapalıdır.")
    return tg_token, tg_chat

def render_asset_management(coin_map, save_assets_func):
    st.sidebar.divider()
    with st.sidebar.expander("➕ Varlık Yönetimi", expanded=False):
        st.info("Listeye yeni Coin, Hisse veya Emtia ekleyin.")
        
        with st.form("add_asset_form"):
            new_name = st.text_input("Görünen İsim (Örn: Pound)")
            new_symbol = st.text_input("Yahoo Kodu (Örn: GBPUSD=X)")
            submitted = st.form_submit_button("Listeye Ekle")
            
            if submitted:
                if new_name and new_symbol:
                    st.session_state['coin_map'][new_name] = new_symbol
                    save_assets_func(st.session_state['coin_map'])
                    st.success(f"{new_name} eklendi!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("İsim ve Sembol boş olamaz!")
        
        st.write("---")
        del_asset = st.selectbox("Silinecek Varlık", list(st.session_state['coin_map'].keys()), key="del_box")
        
        if st.button("Seçileni Sil"):
            if del_asset in st.session_state['coin_map']:
                if len(st.session_state['coin_map']) > 1:
                    del st.session_state['coin_map'][del_asset]
                    save_assets_func(st.session_state['coin_map'])
                    st.warning(f"{del_asset} silindi.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Listede en az 1 varlık kalmalı!")

        if st.button("🔄 Listeyi Sıfırla"):
            st.session_state['coin_map'] = DEFAULT_COIN_MAP.copy()
            save_assets_func(st.session_state['coin_map'])
            st.rerun()

def render_main_chart(df_view, view_tf, curr, f_dates, f_prices, ai_score, show_cloud, show_pred, show_ai, show_all_pats, f_wm, f_candle, f_advanced, items_raw, lines):
    fig = go.Figure()
    
    if show_cloud:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view['EMA_20'],
            line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
            name='EMA 20'
        ))
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view['EMA_50'],
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
            name='EMA 50 (Bulut Altı)'
        ))

    fig.add_trace(go.Candlestick(
        x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='Fiyat',
        increasing_line_color=theme.CANDLE_UP, decreasing_line_color=theme.CANDLE_DOWN
    ))
    fig.add_hline(y=curr, line_dash="dot", line_color="cyan", annotation_text=f" {curr:,.2f}", annotation_position="right")
    
    if show_pred and len(f_dates) > 0:
        fig.add_trace(go.Scatter(
            x=[df_view.index[-1]]+f_dates, 
            y=[df_view['Close'].iloc[-1]]+list(f_prices), 
            mode='lines', 
            line=dict(color='yellow', width=2, dash='dash'), 
            name=f'AI Tahmini (Güven: %{ai_score:.0f})'
        ))
        
        if ai_score > 70: st.success(f"🧠 **AI Güven Skoru:** %{ai_score:.1f} (Model bu coini çok iyi tanıyor!)")
        elif ai_score > 40: st.warning(f"🧠 **AI Güven Skoru:** %{ai_score:.1f} (Tahminler orta güvenilirlikte)")
        else: st.error(f"🧠 **AI Güven Skoru:** %{ai_score:.1f} (Piyasa çok belirsiz, AI zorlanıyor)")
             
    if show_ai and lines:
        for l in lines:
            fig.add_shape(type="line", x0=l['x0'], y0=l['y0'], x1=l['x1'], y1=l['y1'], line=dict(color=l['color'], width=2, dash='dot'))

    if show_all_pats:
        for i in items_raw:
            draw = False
            if i['type'] == 'box' and f_wm: draw = True
            if i['type'] == 'icon' and f_candle: draw = True
            if draw:
                if i['type'] == 'box':
                    fig.add_shape(
                        type="rect", 
                        x0=i['x0'], y0=i['y0'], 
                        x1=i['x1'], y1=i['y1'], 
                        line=dict(color=i['color'], width=2), 
                        fillcolor=i['color'], 
                        opacity=0.15
                    )
                    fig.add_hline(
                        y=i['target'], 
                        line_dash="dashdot", 
                        line_color="magenta", 
                        annotation_text="HEDEF"
                    )
                elif i['type'] == 'icon':
                    fig.add_annotation(
                        x=i['x'], y=i['y'], 
                        text=i['msg'], 
                        showarrow=False, 
                        yshift=15 if i.get('anchor')=='bottom' else -15
                    )
        
        if f_advanced:
            try:
                advanced_items = detect_advanced_patterns(df_view)
               
                for adv in advanced_items:
                    if adv['type'] == 'triangle':
                        # Draw BOTH boundary lines (resistance + support) so
                        # the actual triangle shape shows, not just one edge.
                        for line in adv.get('lines', []):
                            fig.add_shape(type="line", x0=line['x0'], y0=line['y0'], x1=line['x1'], y1=line['y1'],
                                          line=dict(color=adv['color'], width=2, dash='dot'))
                        fig.add_hline(y=adv['target'], line_dash="dashdot", line_color=adv['color'], annotation_text=f"🎯 {adv['name']}")

                    elif adv['type'] == 'harmonic':
                        points = adv['points']
                        for i in range(len(points) - 1):
                            fig.add_shape(type="line", x0=df_view.index[points[i]['idx']], y0=points[i]['price'], x1=df_view.index[points[i+1]['idx']], y1=points[i+1]['price'], line=dict(color=adv['color'], width=2))
                        fig.add_annotation(x=df_view.index[points[2]['idx']], y=points[2]['price'], text=adv['name'], showarrow=True, arrowhead=2, bgcolor=adv['color'], font=dict(color='white'))

                    elif adv['type'] == 'reversal':
                        # Draw the actual left-shoulder -> head -> right-shoulder
                        # shape (previously only the neckline/target were shown).
                        if 'head_x' in adv:
                            fig.add_shape(type="line", x0=adv['x0'], y0=adv['y0'], x1=adv['head_x'], y1=adv['head_y'], line=dict(color=adv['color'], width=2))
                            fig.add_shape(type="line", x0=adv['head_x'], y0=adv['head_y'], x1=adv['x1'], y1=adv['y1'], line=dict(color=adv['color'], width=2))
                            fig.add_annotation(x=adv['head_x'], y=adv['head_y'], text="Baş", showarrow=True, arrowhead=2, bgcolor=adv['color'], font=dict(color='white'))
                        fig.add_hline(y=adv['neckline'], line_dash="solid", line_color=adv['color'], annotation_text="Boyun Çizgisi")
                        fig.add_hline(y=adv['target'], line_dash="dot", line_color="red", annotation_text=f"🎯 {adv['name']}")

                    elif adv['type'] == 'continuation':
                        # Draw the flag/pole channel as a shaded box (previously
                        # this coordinate data was computed but discarded — only
                        # a text label was shown, which is the "one line" bug).
                        fig.add_shape(type="rect", x0=adv['x0'], y0=adv['y0'], x1=adv['x1'], y1=adv['y1'],
                                      line=dict(color=adv['color'], width=2), fillcolor=adv['color'], opacity=0.15)
                        fig.add_hline(y=adv['target'], line_dash="dashdot", line_color=adv['color'], annotation_text=f"🎯 {adv['name']}")
            except Exception as e:
                st.error(f"Gelişmiş formasyon çizim hatası: {e}")

    s_list, r_list = calculate_sr_advanced(df_view, view_tf)
    for s in [x for x in s_list if x < curr][-3:]:
        fig.add_hline(y=s, line_dash="dash", line_color="#00FF00", annotation_text=f"Dst: {s}")
    for r in [x for x in r_list if x > curr][:3]:
        fig.add_hline(y=r, line_dash="dash", line_color="#FF0000", annotation_text=f"Dir: {r}")

    import numpy as np 
    try:
        y_type = "log" if view_tf == "1wk" else "linear"
        zoom_count = 50 if view_tf == "1wk" else (80 if view_tf == "1d" else 100)
        
        if len(df_view) > zoom_count:
            visible_df = df_view.tail(zoom_count)
            zoom_start = visible_df.index[0]
            y_min_raw = visible_df['Low'].min()
            y_max_raw = visible_df['High'].max()
        else:
            zoom_start = df_view.index[0]
            y_min_raw = df_view['Low'].min()
            y_max_raw = df_view['High'].max()

        range_y = None 
        if y_type == "log":
            safe_min = max(y_min_raw, 0.000001) 
            range_y = [np.log10(safe_min * 0.90), np.log10(y_max_raw * 1.10)]
        else:
            range_y = [y_min_raw * 0.95, y_max_raw * 1.05]

        gap_multiplier = 3 if view_tf == "1wk" else 5
        if len(df_view) > 2:
            delta = df_view.index[-1] - df_view.index[-2]
            zoom_end = df_view.index[-1] + (delta * gap_multiplier)
        else:
            zoom_end = df_view.index[-1]

        spike_style = dict(
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikethickness=1, spikedash='solid',
            spikecolor='rgba(230,234,242,0.35)',
        )
        base_layout = theme.plotly_base_layout()
        base_layout.update(
            height=900,
            xaxis_rangeslider_visible=False,
            dragmode="pan",
            yaxis=dict(side="right", fixedrange=False, type=y_type, range=range_y, tickformat=".2f", exponentformat="none", **spike_style),
            xaxis=dict(range=[zoom_start, zoom_end], type="date", **spike_style),
            margin=dict(l=10, r=60, t=10, b=20),
            hovermode='x'
        )
        fig.update_layout(**base_layout)

        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'editable': False,
            'showAxisRangeEntryBoxes': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
            'displaylogo': False,
            'responsive': True
        }
        
        st.plotly_chart(fig, use_container_width=True, config=config, key="main_price_chart")
    except Exception as e:
        st.error(f"Grafik çizilirken hata oluştu: {e}")
        
    return s_list, r_list
