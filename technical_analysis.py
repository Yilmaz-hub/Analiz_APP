import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from config import IndicatorConfig, SignalConfig, Constants
from logger import logger

def calculate_sr_advanced(df, timeframe):
    supports, resistances = [], []
    n = 5 if timeframe == "4h" else (10 if timeframe == "1d" else 20)
    work_df = df.tail(500)
    
    for i in range(n, len(work_df) - n):
        l = work_df['Low'].iloc[i]
        h = work_df['High'].iloc[i]
        
        if l == work_df['Low'].iloc[i-n:i+n+1].min():
            supports.append(l)
        if h == work_df['High'].iloc[i-n:i+n+1].max():
            resistances.append(h)
    
    swing_high = work_df['High'].max()
    swing_low = work_df['Low'].min()
    diff = swing_high - swing_low
    
    fib_levels = [
        swing_low + diff * 0.236,
        swing_low + diff * 0.382,
        swing_low + diff * 0.5,
        swing_low + diff * 0.618,
        swing_low + diff * 0.786
    ]
    
    current_price = df['Close'].iloc[-1]
    
    for fib in fib_levels:
        if fib < current_price:
            supports.append(fib)
        else:
            resistances.append(fib)
    
    if 'Volume' in work_df.columns and work_df['Volume'].sum() > 0:
        price_bins = pd.cut(work_df['Close'], bins=50)
        volume_profile = work_df.groupby(price_bins)['Volume'].sum()
        top_volume_levels = volume_profile.nlargest(3).index
        for interval in top_volume_levels:
            mid_price = (interval.left + interval.right) / 2
            if mid_price < current_price:
                supports.append(mid_price)
            else:
                resistances.append(mid_price)
    
    supports = sorted(list(set([round(x, 2) for x in supports])))
    resistances = sorted(list(set([round(x, 2) for x in resistances])))
    
    return supports, resistances

def calculate_oracle_signal_v2(df, supports, resistances):
    if df is None or len(df) < 50: 
        return "Veri Yok", "gray", ""
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    rsi = last.get('RSI', 50)
    price = last['Close']
    ema_20 = last.get('EMA_20', price)
    ema_50 = last.get('EMA_50', price)
    bb_lower = last.get('BB_Lower', price * 0.9)
    bb_upper = last.get('BB_Upper', price * 1.1)
    
    macd_current = last.get('MACD')
    macd_prev = prev.get('MACD')
    macd_signal = last.get('MACD_Signal')

    macd_cross_up = False
    macd_cross_down = False
    if all(pd.notna([macd_current, macd_prev, macd_signal])):
        macd_cross_up = macd_prev < macd_signal and macd_current > macd_signal
        macd_cross_down = macd_prev > macd_signal and macd_current < macd_signal
    
    vol_ratio = 1
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol_avg = df['Volume'].rolling(IndicatorConfig.VOLUME_SMA_LENGTH).mean().iloc[-1]
        if vol_avg > 0:
            vol_ratio = last['Volume'] / vol_avg

    nearest_support = max([s for s in supports if s < price], default=0)
    nearest_resistance = min([r for r in resistances if r > price], default=price * 2)
    
    support_dist = ((price - nearest_support) / price * 100) if nearest_support > 0 else 100
    resist_dist = ((nearest_resistance - price) / price * 100) if nearest_resistance < price * 2 else 100
    
    score = SignalConfig.NEUTRAL_START_SCORE

    if rsi < SignalConfig.RSI_OVERSOLD:
        score += SignalConfig.RSI_OVERSOLD_SCORE
    elif rsi < SignalConfig.RSI_OVERSOLD_LIGHT:
        score += SignalConfig.RSI_OVERSOLD_LIGHT_SCORE
    elif rsi > SignalConfig.RSI_OVERBOUGHT:
        score += SignalConfig.RSI_OVERBOUGHT_SCORE
    elif rsi > SignalConfig.RSI_OVERBOUGHT_LIGHT:
        score += SignalConfig.RSI_OVERBOUGHT_LIGHT_SCORE
    
    if price > ema_20 > ema_50:
        score += SignalConfig.TREND_SCORE
    elif price < ema_20 < ema_50:
        score -= SignalConfig.TREND_SCORE

    if price < bb_lower:
        score += SignalConfig.BOLLINGER_SCORE
    elif price > bb_upper:
        score -= SignalConfig.BOLLINGER_SCORE

    if macd_cross_up:
        score += SignalConfig.MACD_CROSS_SCORE
    elif macd_cross_down:
        score -= SignalConfig.MACD_CROSS_SCORE

    if vol_ratio > IndicatorConfig.HIGH_VOLUME_THRESHOLD:
        score += SignalConfig.HIGH_VOLUME_SCORE
    elif vol_ratio < IndicatorConfig.LOW_VOLUME_THRESHOLD:
        score += SignalConfig.LOW_VOLUME_PENALTY

    if support_dist < SignalConfig.SUPPORT_DISTANCE_THRESHOLD:
        score += SignalConfig.SUPPORT_PROXIMITY_SCORE
    if resist_dist < SignalConfig.RESISTANCE_DISTANCE_THRESHOLD:
        score += SignalConfig.RESISTANCE_PROXIMITY_SCORE
    
    if score >= SignalConfig.STRONG_BUY_THRESHOLD:
        status, color = "GÜÇLÜ AL 🚀", "green"
        target_msg = f"📈 Hedef: ${nearest_resistance:,.2f} (+%{resist_dist:.1f})"
    elif score >= SignalConfig.BUY_THRESHOLD:
        status, color = "AL (Dikkatli) 📊", "blue"
        target_msg = f"Hedef: ${bb_upper:,.2f}"
    elif score <= SignalConfig.STRONG_SELL_THRESHOLD:
        status, color = "GÜÇLÜ SAT 📉", "red"
        target_msg = f"📉 Hedef: ${nearest_support:,.2f} (-%{support_dist:.1f})"
    elif score <= SignalConfig.SELL_THRESHOLD:
        status, color = "SAT (Kısmi) ⚠️", "orange"
        target_msg = f"Hedef: ${bb_lower:,.2f}"
    else:
        status, color = "NÖTR (BEKLE) 💤", "gray"
        target_msg = f"Skor: {score}/100 - Yön belirsiz"
    
    return status, color, target_msg

# === NEW: RSI Divergence Detection ===
def detect_rsi_divergence(df, lookback=20):
    """
    Detect RSI divergence:
    - Bullish: price makes lower low but RSI makes higher low
    - Bearish: price makes higher high but RSI makes lower high
    Returns: "BULLISH", "BEARISH", or None
    """
    if df is None or len(df) < lookback + 10 or 'RSI' not in df.columns:
        return None
    
    try:
        recent = df.tail(lookback)
        prices = recent['Close'].values
        rsi_vals = recent['RSI'].values
        
        if np.any(np.isnan(rsi_vals)):
            return None
        
        # Find local minima and maxima in price
        price_lows_idx = argrelextrema(prices, np.less, order=3)[0]
        price_highs_idx = argrelextrema(prices, np.greater, order=3)[0]
        
        # Bullish divergence: price lower low + RSI higher low
        if len(price_lows_idx) >= 2:
            i1, i2 = price_lows_idx[-2], price_lows_idx[-1]
            if prices[i2] < prices[i1] and rsi_vals[i2] > rsi_vals[i1]:
                return "BULLISH"
        
        # Bearish divergence: price higher high + RSI lower high
        if len(price_highs_idx) >= 2:
            i1, i2 = price_highs_idx[-2], price_highs_idx[-1]
            if prices[i2] > prices[i1] and rsi_vals[i2] < rsi_vals[i1]:
                return "BEARISH"
        
        return None
    except Exception as e:
        logger.debug(f"RSI divergence detection error: {e}")
        return None

# === NEW: Trend Strength Score ===
def calculate_trend_strength(df):
    """
    Calculate overall trend strength score from -100 (strong downtrend) to +100 (strong uptrend).
    Uses: EMA alignment, price vs EMA50, ADX, short-term slope.
    """
    if df is None or len(df) < 50:
        return 0
    
    try:
        last = df.iloc[-1]
        price = last['Close']
        ema_20 = last.get('EMA_20', price)
        ema_50 = last.get('EMA_50', price)
        
        score = 0
        
        # 1. EMA alignment (+/- 30)
        if price > ema_20 > ema_50:
            score += 30  # Perfect bullish alignment
        elif price < ema_20 < ema_50:
            score -= 30  # Perfect bearish alignment
        elif price > ema_50:
            score += 15  # Above long EMA
        elif price < ema_50:
            score -= 15  # Below long EMA
        
        # 2. Price distance from EMA50 (+/- 25)
        if ema_50 > 0:
            distance_pct = ((price - ema_50) / ema_50) * 100
            dist_score = max(-25, min(25, distance_pct * 5))
            score += dist_score
        
        # 3. ADX trend strength (+/- 25 or penalty)
        adx_cols = [c for c in df.columns if 'ADX' in c.upper()]
        adx_val = 25  # default neutral
        if adx_cols:
            adx_val = last.get(adx_cols[0], 25)
            if pd.isna(adx_val):
                adx_val = 25
        
        if adx_val > 40:
            # Strong trend — amplify existing direction
            direction = 1 if score > 0 else -1
            score += direction * 25
        elif adx_val > 25:
            direction = 1 if score > 0 else -1
            score += direction * 10
        
        # 4. Short-term slope (+/- 20)
        if len(df) >= 5:
            slope_5 = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
            slope_score = max(-20, min(20, slope_5 * 4))
            score += slope_score
        
        # Apply choppy market penalty AFTER all components are summed
        if adx_val < 20:
            score = score * 0.3  # Reduce total score by 70% in choppy markets
        
        return max(-100, min(100, score))
    except Exception as e:
        logger.debug(f"Trend strength calculation error: {e}")
        return 0

def calculate_trailing_stop(entry, current_price, atr, trailing_pct=0.05):
    initial_stop = entry - (atr * 1.5)
    
    if current_price > entry * (1 + trailing_pct):
        new_stop = current_price - (atr * 1.2)
        return max(initial_stop, new_stop)
    
    return initial_stop

def calculate_extended_trendlines(df, extend_candles=15):
    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index
    if len(dates) > 2: delta = dates[-1] - dates[-2]
    else: return []
    last_date = dates[-1]
    future_date = last_date + (delta * extend_candles)
    max_idxs = argrelextrema(highs, np.greater, order=10)[0]
    min_idxs = argrelextrema(lows, np.less, order=10)[0]
    lines = []
    if len(max_idxs) >= 2:
        p1, p2 = max_idxs[-2], max_idxs[-1]
        if highs[p2] < highs[p1]:
            slope = (highs[p2] - highs[p1]) / (p2 - p1)
            y_ext = highs[p1] + slope * (len(df)-1+extend_candles-p1)
            lines.append({"x0": dates[p1], "y0": highs[p1], "x1": future_date, "y1": y_ext, "color": "red"})
    if len(min_idxs) >= 2:
        p1, p2 = min_idxs[-2], min_idxs[-1]
        if lows[p2] > lows[p1]:
            slope = (lows[p2] - lows[p1]) / (p2 - p1)
            y_ext = lows[p1] + slope * (len(df)-1+extend_candles-p1)
            lines.append({"x0": dates[p1], "y0": lows[p1], "x1": future_date, "y1": y_ext, "color": "green"})
    return lines

def detect_patterns(df):
    patterns = []
    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    min_idxs = argrelextrema(lows, np.less, order=5)[0]
    max_idxs = argrelextrema(highs, np.greater, order=5)[0]

    if len(min_idxs) >= 2:
        i1, i2 = min_idxs[-2], min_idxs[-1]
        d1, d2 = lows[i1], lows[i2]
        if (abs(d1-d2)/d1 < 0.02) and ((i2-i1) > 5):
            neck = highs[i1:i2].max()
            patterns.append({"type": "box", "name": "İkili Dip (W)", "color": "green", "x0": dates[i1], "x1": dates[i2], "y0": min(d1,d2)*0.99, "y1": neck, "target": neck+(neck-(d1+d2)/2)})
    if len(max_idxs) >= 2:
        i1, i2 = max_idxs[-2], max_idxs[-1]
        t1, t2 = highs[i1], highs[i2]
        if (abs(t1-t2)/t1 < 0.02) and ((i2-i1) > 5):
            neck = lows[i1:i2].min()
            patterns.append({"type": "box", "name": "İkili Tepe (M)", "color": "red", "x0": dates[i1], "x1": dates[i2], "y0": neck, "y1": max(t1,t2)*1.01, "target": neck-((t1+t2)/2-neck)})
    for i in range(-5, 0):
        idx = i
        O, H, L, C = opens[idx], highs[idx], lows[idx], closes[idx]
        body = abs(C-O)
        if body < np.mean(np.abs(closes-opens))*0.1:
            patterns.append({"type": "icon", "name": "Doji", "color": "yellow", "x": dates[idx], "y": H, "msg": "⚠️", "anchor": "bottom"})
        if (min(O,C)-L) > 2*body and (H-max(O,C)) < 0.5*body:
            patterns.append({"type": "icon", "name": "Hammer", "color": "lime", "x": dates[idx], "y": L, "msg": "🔨", "anchor": "top"})
        if i < -1:
            if (closes[idx] > opens[idx]) and (closes[idx-1] < opens[idx-1]) and (closes[idx] > opens[idx-1]) and (opens[idx] < closes[idx-1]):
                 patterns.append({"type": "icon", "name": "Yutan Boğa", "color": "cyan", "x": dates[idx], "y": lows[idx], "msg": "🚀", "anchor": "top"})
    return patterns

def detect_advanced_patterns(df):
    patterns = []
    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    if len(df) < 50: return patterns
    
    try:
        window = min(60, len(df))
        work_slice = df.tail(window)
        peak_indices = argrelextrema(work_slice['High'].values, np.greater, order=5)[0]
        trough_indices = argrelextrema(work_slice['Low'].values, np.less, order=5)[0]
        
        if len(peak_indices) >= 2 and len(trough_indices) >= 2:
            resistance_slope = (work_slice['High'].iloc[peak_indices[-1]] - work_slice['High'].iloc[peak_indices[-2]]) / (peak_indices[-1] - peak_indices[-2])
            support_slope = (work_slice['Low'].iloc[trough_indices[-1]] - work_slice['Low'].iloc[trough_indices[-2]]) / (trough_indices[-1] - trough_indices[-2])
            current_price = df['Close'].iloc[-1]

            # A triangle has TWO boundary lines (resistance from the peaks,
            # support from the troughs) — both are always stored so the
            # chart can draw the full shape, not just whichever single pivot
            # pair happened to be used for the slope test below. Each line
            # is extrapolated to the last bar along its own slope so
            # converging/diverging lines actually look converging/diverging.
            last_pos = len(work_slice) - 1
            resistance_y_end = work_slice['High'].iloc[peak_indices[-1]] + resistance_slope * (last_pos - peak_indices[-1])
            support_y_end = work_slice['Low'].iloc[trough_indices[-1]] + support_slope * (last_pos - trough_indices[-1])
            resistance_line = {
                "x0": work_slice.index[peak_indices[-2]], "y0": work_slice['High'].iloc[peak_indices[-2]],
                "x1": work_slice.index[-1], "y1": resistance_y_end,
            }
            support_line = {
                "x0": work_slice.index[trough_indices[-2]], "y0": work_slice['Low'].iloc[trough_indices[-2]],
                "x1": work_slice.index[-1], "y1": support_y_end,
            }

            if abs(resistance_slope) < 0.001 and support_slope > 0.001:
                patterns.append({
                    "type": "triangle", "name": "Yükselen Üçgen ▲", "color": "green",
                    "lines": [resistance_line, support_line],
                    "direction": "BULLISH", "target": current_price * 1.08, "confidence": 75
                })
            elif abs(support_slope) < 0.001 and resistance_slope < -0.001:
                patterns.append({
                    "type": "triangle", "name": "Düşen Üçgen ▼", "color": "red",
                    "lines": [resistance_line, support_line],
                    "direction": "BEARISH", "target": current_price * 0.92, "confidence": 75
                })
            elif resistance_slope < -0.001 and support_slope > 0.001:
                patterns.append({
                    "type": "triangle", "name": "Simetrik Üçgen ◇", "color": "yellow",
                    "lines": [resistance_line, support_line],
                    "direction": "NEUTRAL", "target": current_price, "confidence": 60
                })
    except Exception as e:
        logger.debug(f"Triangle error: {e}")
        
    try:
        all_peaks = argrelextrema(highs, np.greater, order=10)[0]
        all_troughs = argrelextrema(lows, np.less, order=10)[0]
        
        pivots = []
        for i in range(len(df)):
            if i in all_peaks: pivots.append({'idx': i, 'price': highs[i], 'type': 'high'})
            elif i in all_troughs: pivots.append({'idx': i, 'price': lows[i], 'type': 'low'})
        
        if len(pivots) >= 4:
            A, B, C, D_cand = pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            if AB > 0:
                BC_ret = BC / AB
                if 0.382 <= BC_ret <= 0.886:
                    CD_proj = BC * 1.272
                    if A['type'] == 'low' and B['type'] == 'high':
                        target_D = C['price'] - CD_proj
                        patterns.append({
                            "type": "harmonic", "name": "ABCD Boğa 🦬", "color": "cyan",
                            "points": [A, B, C, D_cand], "direction": "BULLISH", "target": target_D, "confidence": 80
                        })
                    elif A['type'] == 'high' and B['type'] == 'low':
                        target_D = C['price'] + CD_proj
                        patterns.append({
                            "type": "harmonic", "name": "ABCD Ayı 🐻", "color": "magenta",
                            "points": [A, B, C, D_cand], "direction": "BEARISH", "target": target_D, "confidence": 80
                        })
                        
        if len(pivots) >= 5:
            X, A, B, C, D_cand = pivots[-5], pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            XA = abs(A['price'] - X['price'])
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D_cand['price'] - C['price'])
            if XA > 0 and AB > 0 and BC > 0:
                AB_ret = AB / XA
                BC_ret = BC / AB
                CD_ext = CD / BC
                XD_ext = abs(D_cand['price'] - X['price']) / XA
                if 0.75 <= AB_ret <= 0.82 and 0.35 <= BC_ret <= 0.90 and 1.5 <= CD_ext <= 2.7 and 1.2 <= XD_ext <= 1.65:
                    patterns.append({
                        "type": "harmonic", "name": "Kelebek 🦋", "color": "purple",
                        "points": [X, A, B, C, D_cand], "direction": "REVERSAL", "target": X['price'], "confidence": 85
                    })
    except Exception as e:
        logger.debug(f"Harmonic error: {e}")
        
    try:
        if len(all_peaks) >= 3 and len(all_troughs) >= 2:
            left_s = highs[all_peaks[-3]]
            head = highs[all_peaks[-2]]
            right_s = highs[all_peaks[-1]]
            left_t = lows[all_troughs[-2]]
            right_t = lows[all_troughs[-1]]
            
            if head > left_s and head > right_s and abs(left_s - right_s) / left_s < 0.10 and abs(left_t - right_t) / left_t < 0.05:
                neckline = (left_t + right_t) / 2
                target = neckline - (head - neckline)
                patterns.append({
                    "type": "reversal", "name": "Baş-Omuz 👤", "color": "red",
                    "x0": dates[all_peaks[-3]], "y0": left_s,
                    "head_x": dates[all_peaks[-2]], "head_y": head,
                    "x1": dates[all_peaks[-1]], "y1": right_s,
                    "neckline": neckline, "direction": "BEARISH", "target": target, "confidence": 90
                })
    except Exception as e:
        logger.debug(f"H&S error: {e}")
        
    try:
        recent_slice = df.tail(30)
        pole_high = recent_slice['High'].iloc[:-10].max()
        pole_low = recent_slice['Low'].iloc[:-10].min()
        flag_highs = recent_slice['High'].iloc[-10:]
        flag_lows = recent_slice['Low'].iloc[-10:]
        
        flag_w = (flag_highs.max() - flag_lows.min()) / pole_high
        pole_h = (pole_high - pole_low) / pole_low
        
        if pole_h > 0.05 and flag_w < 0.03:
            patterns.append({
                "type": "continuation", "name": "Boğa Bayrağı 🚩", "color": "lime",
                "x0": recent_slice.index[-10], "y0": flag_lows.min(), "x1": recent_slice.index[-1], "y1": flag_highs.max(),
                "direction": "BULLISH", "target": pole_high + pole_h * pole_high, "confidence": 70
            })
    except Exception as e:
        logger.debug(f"Flag error: {e}")
        
    return patterns

def calculate_trade_setup(df, signal_type):
    if df is None: return None
    last = df.iloc[-1]
    price = last['Close']
    atr = last.get('ATR', price * 0.02)
    if np.isnan(atr): atr = price * 0.02
    
    if "AL" in signal_type:
        return {
            'entry': price,
            'sl': price - (atr * 1.5),
            'tp': price + (atr * 3.0),
            'direction': 'LONG'
        }
    elif "SAT" in signal_type:
         return {
            'entry': price,
            'sl': price + (atr * 1.5),
            'tp': price - (atr * 3.0),
            'direction': 'SHORT'
        }
    return None

def calculate_regime_score(df):
    """
    Trend-regime quality score (0-100) used by the scanner to rank assets.
    The composite strategy only has an edge on cleanly trending assets, so
    capital should be steered toward high scores.

    Components (weights in RegimeConfig):
      - persistence: % of recent bars closing above the long-term MA
      - slope: long-term MA rise over SLOPE_LOOKBACK bars
      - ADX: trend strength, counted only while price is above the MA
      - alignment: EMA20 > EMA50 > MA stack

    Returns {'score': int, 'label': str, 'components': dict}, or None when
    there is not enough data.
    """
    from config import RegimeConfig as rc

    min_bars = rc.MA_PERIOD + rc.SLOPE_LOOKBACK
    if df is None or 'Close' not in getattr(df, 'columns', []) or len(df) < min_bars:
        return None

    close = df['Close']
    ma = close.rolling(rc.MA_PERIOD).mean()
    ma_now = ma.iloc[-1]

    # Persistence: share of recent bars above the MA (only where MA exists)
    ma_recent = ma.iloc[-rc.PERSISTENCE_LOOKBACK:]
    close_recent = close.iloc[-rc.PERSISTENCE_LOOKBACK:]
    valid = ma_recent.notna()
    persistence_ratio = float((close_recent[valid] > ma_recent[valid]).mean()) if valid.any() else 0.0
    persistence_pts = persistence_ratio * rc.PERSISTENCE_POINTS

    # Slope of the long-term MA
    ma_then = ma.iloc[-1 - rc.SLOPE_LOOKBACK]
    slope_pct = 0.0
    if not pd.isna(ma_then) and ma_then > 0 and not pd.isna(ma_now):
        slope_pct = (ma_now / ma_then - 1) * 100
    slope_pts = max(0.0, min(slope_pct / rc.SLOPE_FULL_SCORE_PCT, 1.0)) * rc.SLOPE_POINTS

    # ADX strength — directionless, so it only earns points in an uptrend
    above_ma = not pd.isna(ma_now) and close.iloc[-1] > ma_now
    adx_val = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 20.0
    adx_norm = max(0.0, min((adx_val - rc.ADX_FLOOR) / (rc.ADX_CEIL - rc.ADX_FLOOR), 1.0))
    adx_pts = adx_norm * rc.ADX_POINTS if above_ma else 0.0

    # EMA alignment
    ema20 = df['EMA_20'].iloc[-1] if 'EMA_20' in df.columns else close.ewm(span=20).mean().iloc[-1]
    ema50 = df['EMA_50'].iloc[-1] if 'EMA_50' in df.columns else close.ewm(span=50).mean().iloc[-1]
    if not pd.isna(ma_now) and ema20 > ema50 > ma_now:
        alignment_pts = rc.ALIGNMENT_POINTS
    elif ema20 > ema50:
        alignment_pts = rc.ALIGNMENT_POINTS * 0.5
    else:
        alignment_pts = 0.0

    score = int(round(persistence_pts + slope_pts + adx_pts + alignment_pts))
    score = max(0, min(score, 100))

    if score >= 70:
        label = "🟢 Güçlü Trend"
    elif score >= 55:
        label = "🟡 Trend Var"
    elif score >= 35:
        label = "⚪ Zayıf/Yatay"
    else:
        label = "🔴 Trend Yok/Düşüş"

    return {
        'score': score,
        'label': label,
        'components': {
            'persistence': round(persistence_pts, 1),
            'slope': round(slope_pts, 1),
            'adx': round(adx_pts, 1),
            'alignment': round(alignment_pts, 1),
        },
    }

def _update_trailing_stop(position, price, atr, direction, trail_breakeven, trail_lock_pct):
    """Advance one open position's trailing stop by one bar.
    direction: +1 for LONG, -1 for SHORT. Shared by both backtest branches
    below (they were previously ~30 lines of mirrored sign-flipped code)."""
    extreme_key = 'highest' if direction == 1 else 'lowest'
    position[extreme_key] = max(position[extreme_key], price) if direction == 1 else min(position[extreme_key], price)

    profit_distance = direction * (position[extreme_key] - position['entry'])
    if profit_distance > atr * 2.0:
        candidate = position['entry'] + direction * profit_distance * trail_lock_pct
        position['sl'] = max(position['sl'], candidate) if direction == 1 else min(position['sl'], candidate)
    elif profit_distance > atr * trail_breakeven:
        position['sl'] = max(position['sl'], position['entry']) if direction == 1 else min(position['sl'], position['entry'])


def _check_exit(position, price, state, direction):
    """Return a close reason ('SL'/'TP'/'Sinyal') or None, for either direction."""
    if direction * (position['sl'] - price) >= 0:
        return "SL"
    if direction * (price - position['tp']) >= 0:
        return "TP"
    if state != direction:
        return "Sinyal"
    return None


def run_strategy_backtest(df, initial_balance=10000, timeframe="1d", progress_callback=None,
                          sl_mult=None, tp_mult=None, entry_score=None, regime_ma_period=None,
                          fee_rate=None, entry_mode=None, pullback_tol=0.005,
                          pullback_min_score=10, rsi_cap=None, direction="long"):
    """
    Backtests the SAME signal the dashboard displays: the composite decision
    engine filtered through SignalStateMachine (confirmation bars + exit
    hysteresis), on closed candles only. The ML dimension is skipped for
    speed (training a model per bar is infeasible); its weight is
    redistributed, exactly as generate_stable_signal does on replay bars.

    Execution model: signal decided on bar i-1's close, executed at bar i's
    close — no lookahead.

    sl_mult / tp_mult / entry_score override the timeframe defaults — used
    for parameter tuning. regime_ma_period overrides the config's
    REGIME_MA_PERIOD (the SMA the signal engine's regime filter uses).
    fee_rate overrides BacktestConfig.FEE_RATE (fraction charged per side).

    Entry-quality research overrides (defaults reproduce shipped behavior):
      entry_mode: None/"breakout" = score >= min_entry_score (shipped);
                  "pullback" = confirmed uptrend + last closed bar at/below
                  EMA20*(1+pullback_tol) with score >= pullback_min_score;
                  "hybrid" = either condition.
      rsi_cap: skip NEW entries when the closed bar's RSI exceeds this.

    direction="short" mirrors the long logic for SAT-side validation:
    entries on confirmed short state (score <= -entry, regime <= 0,
    breakout-only — entry_mode is ignored), SL above / TP below entry,
    mirrored breakeven + profit-lock trailing, exit when the short state
    dies. Margin model: 95% of balance reserved, linear P&L, fee per side.
    """
    # Lazy import: signal_engine imports this module, so a top-level import
    # here would be circular.
    from signal_engine import _compute_bar_score, SignalStateMachine
    from config import DecisionEngineConfig as cfg
    from config import BacktestConfig

    if fee_rate is None:
        fee_rate = BacktestConfig.FEE_RATE

    balance = initial_balance
    position = None
    trades = []
    equity_curve = []
    cooldown = 0

    if len(df) < 100: return None

    # === TIMEFRAME-ADAPTIVE PARAMETERS ===
    # Sourced from BacktestConfig.TIMEFRAME_PARAMS — the single shared place
    # (also read by paper_trading.py) so the two can't silently drift apart.
    params = BacktestConfig.TIMEFRAME_PARAMS.get(timeframe, BacktestConfig.TIMEFRAME_PARAMS["1d"])
    # Weekly trends are clean enough to be strict on entry; other timeframes
    # use the shared entry-score threshold.
    min_entry_score = cfg.STRONG_BUY_THRESHOLD if timeframe == "1wk" else cfg.ENTRY_SCORE
    sl_multiplier = params["sl_mult"]
    tp_multiplier = params["tp_mult"]
    trail_breakeven = params["trail_breakeven"]
    trail_lock_pct = params["trail_lock_pct"]
    cooldown_bars = params["cooldown_bars"]

    # Tuning overrides
    if sl_mult is not None: sl_multiplier = sl_mult
    if tp_mult is not None: tp_multiplier = tp_mult
    if entry_score is not None: min_entry_score = entry_score

    # Custom regime MA for tuning (config's REGIME_MA_PERIOD applies otherwise
    # inside _compute_bar_score)
    custom_regime_ma = None
    if regime_ma_period:
        custom_regime_ma = df['Close'].rolling(regime_ma_period).mean()

    machine = SignalStateMachine(cfg)
    total_bars = len(df) - 60

    for i in range(60, len(df)):
        current_slice = df.iloc[:i]
        price = df['Close'].iloc[i]
        date = df.index[i]

        if progress_callback and (i - 60) % 25 == 0 and total_bars > 0:
            progress_callback((i - 60) / total_bars)

        bar = _compute_bar_score(current_slice, timeframe, include_ml=False)
        if bar is None:
            continue

        regime = bar.get("regime", 0)
        if custom_regime_ma is not None:
            ma_val = custom_regime_ma.iloc[i - 1]
            if pd.isna(ma_val):
                regime = 0
            else:
                regime = 1 if df['Close'].iloc[i - 1] > ma_val else -1

        state, _ = machine.update(bar["score"], bar["confidence"], bar["rsi"], bar["adx"], regime)

        atr = bar["atr"]
        score = bar["score"]

        # === ENTRY LOGIC (SHORT) ===
        if direction == "short" and position is None and balance > 0:
            if cooldown > 0:
                cooldown -= 1
            else:
                entry_ok = score <= -min_entry_score
                if rsi_cap is not None and bar["rsi"] < (100 - rsi_cap):
                    entry_ok = False  # mirrored: don't short into oversold
                if state == -1 and entry_ok and regime <= 0:
                    margin = balance * 0.95
                    qty = margin / price
                    entry_fee = qty * price * fee_rate
                    position = {
                        'entry': price, 'entry_date': date, 'qty': qty,
                        'type': 'SHORT', 'lowest': price, 'margin': margin,
                        'cost': margin + entry_fee,
                        'sl': price + (atr * sl_multiplier),
                        'tp': price - (atr * tp_multiplier),
                    }
                    balance -= (margin + entry_fee)

        # === EXIT LOGIC (SHORT) ===
        elif direction == "short" and position is not None:
            _update_trailing_stop(position, price, atr, -1, trail_breakeven, trail_lock_pct)
            close_reason = _check_exit(position, price, state, -1)

            if close_reason:
                exit_fee = position['qty'] * price * fee_rate
                pnl = position['qty'] * (position['entry'] - price) - exit_fee - (position['cost'] - position['margin'])
                balance += position['margin'] + position['qty'] * (position['entry'] - price) - exit_fee
                trades.append({
                    'entry': position['entry'], 'exit': price, 'entry_date': position['entry_date'],
                    'exit_date': date, 'pnl': pnl, 'pnl_pct': (pnl / position['cost']) * 100,
                    'reason': close_reason
                })
                if pnl < 0:
                    cooldown = cooldown_bars
                position = None

        # === ENTRY LOGIC ===
        elif position is None and balance > 0:
            if cooldown > 0:
                cooldown -= 1
            else:
                # Entry condition on the last CLOSED bar (i-1), like the signal
                breakout_ok = score >= min_entry_score
                if entry_mode in ("pullback", "hybrid"):
                    prev_close = df['Close'].iloc[i - 1]
                    ema20 = df['EMA_20'].iloc[i - 1] if 'EMA_20' in df.columns else prev_close
                    pullback_ok = (not pd.isna(ema20)
                                   and prev_close <= ema20 * (1 + pullback_tol)
                                   and score >= pullback_min_score)
                    entry_ok = pullback_ok if entry_mode == "pullback" else (breakout_ok or pullback_ok)
                else:
                    entry_ok = breakout_ok
                if rsi_cap is not None and bar["rsi"] > rsi_cap:
                    entry_ok = False

                if state == 1 and entry_ok and regime >= 0:
                    qty = (balance * 0.95) / price
                    cost = qty * price * (1 + fee_rate)  # entry fee paid on top
                    position = {
                        'entry': price, 'entry_date': date, 'qty': qty,
                        'type': 'LONG', 'highest': price, 'cost': cost,
                        'sl': price - (atr * sl_multiplier),
                        'tp': price + (atr * tp_multiplier),
                    }
                    balance -= cost

        # === EXIT LOGIC (Trailing Stop) ===
        elif position is not None:
            _update_trailing_stop(position, price, atr, 1, trail_breakeven, trail_lock_pct)
            close_reason = _check_exit(position, price, state, 1)

            if close_reason:
                proceeds = position['qty'] * price * (1 - fee_rate)  # exit fee
                pnl = proceeds - position['cost']
                balance += proceeds
                trades.append({
                    'entry': position['entry'], 'exit': price, 'entry_date': position['entry_date'],
                    'exit_date': date, 'pnl': pnl, 'pnl_pct': (pnl / position['cost']) * 100,
                    'reason': close_reason
                })
                if pnl < 0:
                    cooldown = cooldown_bars
                position = None
                
        if position is None:
            current_equity = balance
        elif position['type'] == 'SHORT':
            current_equity = balance + position['margin'] + position['qty'] * (position['entry'] - price)
        else:
            current_equity = balance + position['qty'] * price
        equity_curve.append({'date': date, 'equity': current_equity})

    if position is not None:
        final_price = df['Close'].iloc[-1]
        if position['type'] == 'SHORT':
            exit_fee = position['qty'] * final_price * fee_rate
            pnl = position['qty'] * (position['entry'] - final_price) - exit_fee - (position['cost'] - position['margin'])
            balance += position['margin'] + position['qty'] * (position['entry'] - final_price) - exit_fee
        else:
            proceeds = position['qty'] * final_price * (1 - fee_rate)
            pnl = proceeds - position['cost']
            balance += proceeds
        trades.append({
            'entry': position['entry'], 'exit': final_price, 'pnl': pnl,
            'pnl_pct': (pnl / position['cost']) * 100, 'reason': 'Final'
        })
        
    if not trades: return None
    
    total_return = ((balance - initial_balance) / initial_balance) * 100
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(winning) / len(trades)) * 100
    avg_win = np.mean([t['pnl'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl'] for t in losing]) if losing else 0
    profit_factor = abs(sum([t['pnl'] for t in winning]) / sum([t['pnl'] for t in losing])) if losing and sum([t['pnl'] for t in losing]) != 0 else 0
    
    return {
        'final_balance': balance, 'total_return': total_return, 'total_trades': len(trades),
        'winning_trades': len(winning), 'losing_trades': len(losing), 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor,
        'trades': trades, 'equity_curve': equity_curve
    }
