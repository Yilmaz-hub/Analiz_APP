"""
ADVANCED ANALYSIS MODULE
Elliott Wave, Ichimoku Cloud, Wyckoff Phase, Market Structure
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from logger import logger


# =============================================
# ELLIOTT WAVE ANALYSIS
# =============================================
def detect_elliott_wave(df, order=10):
    """
    Detect Elliott Wave patterns (5-wave impulse + ABC correction).
    Returns dict with wave_count, current_wave, direction, targets, labels.
    """
    result = {
        "detected": False,
        "type": None,        # "IMPULSE" or "CORRECTIVE"
        "current_wave": 0,
        "direction": "NEUTRAL",
        "confidence": 0,
        "targets": [],
        "labels": [],        # For chart overlay
        "description": ""
    }

    if df is None or len(df) < 80:
        return result

    try:
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        dates = df.index

        peak_idx = argrelextrema(highs, np.greater, order=order)[0]
        trough_idx = argrelextrema(lows, np.less, order=order)[0]

        # Merge and sort pivots
        pivots = []
        for i in peak_idx:
            pivots.append({"idx": int(i), "price": float(highs[i]), "type": "high", "date": dates[i]})
        for i in trough_idx:
            pivots.append({"idx": int(i), "price": float(lows[i]), "type": "low", "date": dates[i]})
        pivots.sort(key=lambda x: x["idx"])

        # Remove consecutive same-type pivots (keep extremes)
        filtered = []
        for p in pivots:
            if not filtered or filtered[-1]["type"] != p["type"]:
                filtered.append(p)
            else:
                if p["type"] == "high" and p["price"] > filtered[-1]["price"]:
                    filtered[-1] = p
                elif p["type"] == "low" and p["price"] < filtered[-1]["price"]:
                    filtered[-1] = p
        pivots = filtered

        if len(pivots) < 6:
            return result

        # --- Try 5-wave BULLISH impulse (last 6 pivots: trough-peak pattern) ---
        last6 = pivots[-6:]
        if last6[0]["type"] == "low":
            w1_start, w1_end = last6[0]["price"], last6[1]["price"]
            w2_end = last6[2]["price"]
            w3_end = last6[3]["price"]
            w4_end = last6[4]["price"]
            w5_end = last6[5]["price"] if last6[5]["type"] == "high" else None

            if w5_end and w1_end > w1_start and w2_end > w1_start and w3_end > w1_end:
                # Rule: Wave 3 not shortest
                w1_len = w1_end - w1_start
                w3_len = w3_end - w2_end
                w5_len = w5_end - w4_end if w5_end > w4_end else 0

                # Rule: Wave 4 doesn't overlap Wave 1
                wave4_ok = w4_end > w1_end * 0.99  # small tolerance

                if w3_len >= w1_len and w3_len >= w5_len and wave4_ok and w5_len > 0:
                    # Valid bullish impulse
                    result["detected"] = True
                    result["type"] = "IMPULSE"
                    result["direction"] = "BULLISH"
                    result["current_wave"] = 5
                    result["confidence"] = 75

                    # Fibonacci extension targets for correction
                    full_move = w5_end - w1_start
                    result["targets"] = [
                        round(w5_end - full_move * 0.382, 2),
                        round(w5_end - full_move * 0.500, 2),
                        round(w5_end - full_move * 0.618, 2)
                    ]
                    result["labels"] = [
                        {"wave": "1", "date": last6[1]["date"], "price": w1_end},
                        {"wave": "2", "date": last6[2]["date"], "price": w2_end},
                        {"wave": "3", "date": last6[3]["date"], "price": w3_end},
                        {"wave": "4", "date": last6[4]["date"], "price": w4_end},
                        {"wave": "5", "date": last6[5]["date"], "price": w5_end},
                    ]
                    result["description"] = f"Boğa İmpuls Dalgası tamamlandı. Dalga 5 = ${w5_end:,.2f}. Düzeltme bekleniyor."
                    return result

        # --- Try 5-wave BEARISH impulse ---
        if last6[0]["type"] == "high":
            w1_start, w1_end = last6[0]["price"], last6[1]["price"]
            w2_end = last6[2]["price"]
            w3_end = last6[3]["price"]
            w4_end = last6[4]["price"]
            w5_end = last6[5]["price"] if last6[5]["type"] == "low" else None

            if w5_end and w1_end < w1_start and w2_end < w1_start and w3_end < w1_end:
                w1_len = abs(w1_start - w1_end)
                w3_len = abs(w2_end - w3_end)
                w5_len = abs(w4_end - w5_end) if w5_end < w4_end else 0
                wave4_ok = w4_end < w1_end * 1.01

                if w3_len >= w1_len and w3_len >= w5_len and wave4_ok and w5_len > 0:
                    result["detected"] = True
                    result["type"] = "IMPULSE"
                    result["direction"] = "BEARISH"
                    result["current_wave"] = 5
                    result["confidence"] = 75
                    full_move = abs(w1_start - w5_end)
                    result["targets"] = [
                        round(w5_end + full_move * 0.382, 2),
                        round(w5_end + full_move * 0.500, 2),
                        round(w5_end + full_move * 0.618, 2)
                    ]
                    result["labels"] = [
                        {"wave": "1", "date": last6[1]["date"], "price": w1_end},
                        {"wave": "2", "date": last6[2]["date"], "price": w2_end},
                        {"wave": "3", "date": last6[3]["date"], "price": w3_end},
                        {"wave": "4", "date": last6[4]["date"], "price": w4_end},
                        {"wave": "5", "date": last6[5]["date"], "price": w5_end},
                    ]
                    result["description"] = f"Ayı İmpuls Dalgası tamamlandı. Dalga 5 = ${w5_end:,.2f}. Toparlanma bekleniyor."
                    return result

        # --- Try ABC Corrective (last 4 pivots) ---
        if len(pivots) >= 4:
            last4 = pivots[-4:]
            A_start, A_end = last4[0]["price"], last4[1]["price"]
            B_end = last4[2]["price"]
            C_end = last4[3]["price"]

            AB = abs(A_end - A_start)
            BC = abs(B_end - A_end)
            CD = abs(C_end - B_end)

            if AB > 0 and 0.3 <= BC / AB <= 0.8:
                is_bearish_abc = A_end < A_start and C_end < B_end
                is_bullish_abc = A_end > A_start and C_end > B_end

                if is_bearish_abc or is_bullish_abc:
                    result["detected"] = True
                    result["type"] = "CORRECTIVE"
                    result["direction"] = "BEARISH" if is_bearish_abc else "BULLISH"
                    result["current_wave"] = 3  # Wave C
                    result["confidence"] = 60
                    result["labels"] = [
                        {"wave": "A", "date": last4[1]["date"], "price": A_end},
                        {"wave": "B", "date": last4[2]["date"], "price": B_end},
                        {"wave": "C", "date": last4[3]["date"], "price": C_end},
                    ]
                    direction_tr = "Düşüş" if is_bearish_abc else "Yükseliş"
                    result["description"] = f"ABC Düzeltme Dalgası ({direction_tr}). Dalga C = ${C_end:,.2f}."

    except Exception as e:
        logger.debug(f"Elliott Wave detection error: {e}")

    return result


# =============================================
# ICHIMOKU CLOUD ANALYSIS
# =============================================
def analyze_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    """Ichimoku Cloud analysis: TK cross, cloud position, Chikou confirmation."""
    result = {
        "signal": "NEUTRAL",
        "cloud_status": "",
        "tk_cross": "",
        "chikou": "",
        "score": 0,
        "description": ""
    }

    if df is None or len(df) < senkou_b + 26:
        return result

    try:
        high = df['High']
        low = df['Low']
        close = df['Close']

        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        chikou = close.shift(-kijun)

        price = close.iloc[-1]
        sa = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else price
        sb = senkou_b_line.iloc[-1] if not pd.isna(senkou_b_line.iloc[-1]) else price
        tk = tenkan_sen.iloc[-1] if not pd.isna(tenkan_sen.iloc[-1]) else price
        kj = kijun_sen.iloc[-1] if not pd.isna(kijun_sen.iloc[-1]) else price

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        score = 0
        reasons = []

        # Cloud position
        if price > cloud_top:
            score += 30
            result["cloud_status"] = "Bulutun ÜSTÜNDE (Boğa)"
            reasons.append("Fiyat bulutun üstünde")
        elif price < cloud_bottom:
            score -= 30
            result["cloud_status"] = "Bulutun ALTINDA (Ayı)"
            reasons.append("Fiyat bulutun altında")
        else:
            result["cloud_status"] = "Bulut İÇİNDE (Kararsız)"
            reasons.append("Fiyat bulut içinde")

        # TK Cross
        tk_prev = tenkan_sen.iloc[-2] if not pd.isna(tenkan_sen.iloc[-2]) else tk
        kj_prev = kijun_sen.iloc[-2] if not pd.isna(kijun_sen.iloc[-2]) else kj

        if tk_prev < kj_prev and tk > kj:
            score += 25
            result["tk_cross"] = "Boğa Kesişimi (Tenkan > Kijun)"
            reasons.append("TK boğa kesişimi")
        elif tk_prev > kj_prev and tk < kj:
            score -= 25
            result["tk_cross"] = "Ayı Kesişimi (Tenkan < Kijun)"
            reasons.append("TK ayı kesişimi")
        elif tk > kj:
            score += 10
            result["tk_cross"] = "Tenkan > Kijun"
        else:
            score -= 10
            result["tk_cross"] = "Tenkan < Kijun"

        # Chikou confirmation
        chikou_idx = -kijun - 1
        if abs(chikou_idx) < len(close):
            chikou_price = close.iloc[chikou_idx]
            if price > chikou_price:
                score += 15
                result["chikou"] = "Chikou onayı (Boğa)"
            else:
                score -= 15
                result["chikou"] = "Chikou reddi (Ayı)"

        # Cloud thickness = trend strength
        cloud_thickness = abs(sa - sb) / price * 100
        if cloud_thickness > 3:
            direction = 1 if sa > sb else -1
            score += direction * 10

        result["score"] = max(-100, min(100, score))
        result["signal"] = "BULLISH" if score > 20 else ("BEARISH" if score < -20 else "NEUTRAL")
        result["description"] = " | ".join(reasons) if reasons else "Ichimoku nötr"

    except Exception as e:
        logger.debug(f"Ichimoku error: {e}")

    return result


# =============================================
# WYCKOFF PHASE DETECTION
# =============================================
def detect_wyckoff_phase(df, lookback=60):
    """Detect Wyckoff accumulation/distribution phases."""
    result = {
        "phase": "UNKNOWN",
        "signal": "NEUTRAL",
        "score": 0,
        "description": ""
    }

    if df is None or len(df) < lookback:
        return result

    try:
        recent = df.tail(lookback)
        prices = recent['Close'].values
        highs = recent['High'].values
        lows = recent['Low'].values

        has_volume = 'Volume' in recent.columns and recent['Volume'].sum() > 0
        if has_volume:
            volumes = recent['Volume'].values
            avg_vol = np.mean(volumes)

        price_range = (np.max(highs) - np.min(lows)) / np.mean(prices) * 100
        half = lookback // 2
        first_half_range = (np.max(highs[:half]) - np.min(lows[:half])) / np.mean(prices[:half]) * 100
        second_half_range = (np.max(highs[half:]) - np.min(lows[half:])) / np.mean(prices[half:]) * 100

        price_trend = (prices[-1] - prices[0]) / prices[0] * 100
        recent_trend = (prices[-1] - prices[-10]) / prices[-10] * 100

        score = 0

        # ACCUMULATION: Range-bound with declining volume, then spring (price dip below range then recovery)
        if price_range < 15 and abs(price_trend) < 5:
            min_price = np.min(lows)
            last_10_min = np.min(lows[-10:])

            if last_10_min <= min_price * 1.01 and recent_trend > 0:
                # Spring detected
                result["phase"] = "ACCUMULATION (Spring)"
                result["signal"] = "BULLISH"
                score = 60
                result["description"] = "Wyckoff Birikim: Spring testi — dip yapıp toparlandı. Yükseliş beklentisi."

                if has_volume and np.mean(volumes[-5:]) > avg_vol * 1.2:
                    score += 15
                    result["description"] += " Hacim artışı onaylıyor."

            elif has_volume and np.mean(volumes[half:]) < np.mean(volumes[:half]) * 0.8:
                result["phase"] = "ACCUMULATION"
                result["signal"] = "BULLISH"
                score = 35
                result["description"] = "Wyckoff Birikim: Yatay bant + azalan hacim. Kırılım yakın olabilir."
            else:
                result["phase"] = "RANGING"
                result["description"] = "Yatay bant — Wyckoff fazı netleşmedi."

        # DISTRIBUTION: Range-bound at highs with UTAD
        elif price_range < 15 and abs(price_trend) < 5:
            max_price = np.max(highs)
            last_10_max = np.max(highs[-10:])

            if last_10_max >= max_price * 0.99 and recent_trend < 0:
                result["phase"] = "DISTRIBUTION (UTAD)"
                result["signal"] = "BEARISH"
                score = -60
                result["description"] = "Wyckoff Dağıtım: UTAD testi — tepe yapıp geri çekildi. Düşüş beklentisi."
            else:
                result["phase"] = "DISTRIBUTION"
                result["signal"] = "BEARISH"
                score = -30
                result["description"] = "Wyckoff Dağıtım: Tepe bölgesinde yatay hareket."

        # MARKUP: Strong uptrend with expanding volume
        elif price_trend > 10 and recent_trend > 2:
            result["phase"] = "MARKUP"
            result["signal"] = "BULLISH"
            score = 40
            result["description"] = f"Wyckoff Yükseliş Fazı: +%{price_trend:.1f} trend."
            if has_volume and np.mean(volumes[-10:]) > avg_vol:
                score += 15

        # MARKDOWN: Strong downtrend
        elif price_trend < -10 and recent_trend < -2:
            result["phase"] = "MARKDOWN"
            result["signal"] = "BEARISH"
            score = -40
            result["description"] = f"Wyckoff Düşüş Fazı: %{price_trend:.1f} trend."

        result["score"] = max(-100, min(100, score))

    except Exception as e:
        logger.debug(f"Wyckoff error: {e}")

    return result


# =============================================
# MARKET STRUCTURE ANALYSIS
# =============================================
def analyze_market_structure(df, order=5):
    """Detect HH/HL (uptrend), LH/LL (downtrend), BOS, CHoCH."""
    result = {
        "structure": "UNKNOWN",
        "signal": "NEUTRAL",
        "bos": False,
        "choch": False,
        "score": 0,
        "description": ""
    }

    if df is None or len(df) < 40:
        return result

    try:
        highs = df['High'].values
        lows = df['Low'].values

        peak_idx = argrelextrema(highs, np.greater, order=order)[0]
        trough_idx = argrelextrema(lows, np.less, order=order)[0]

        if len(peak_idx) < 3 or len(trough_idx) < 3:
            return result

        # Last 3 swing highs and lows
        last_highs = [highs[i] for i in peak_idx[-3:]]
        last_lows = [lows[i] for i in trough_idx[-3:]]

        hh_count = sum(1 for i in range(1, len(last_highs)) if last_highs[i] > last_highs[i-1])
        hl_count = sum(1 for i in range(1, len(last_lows)) if last_lows[i] > last_lows[i-1])
        lh_count = sum(1 for i in range(1, len(last_highs)) if last_highs[i] < last_highs[i-1])
        ll_count = sum(1 for i in range(1, len(last_lows)) if last_lows[i] < last_lows[i-1])

        score = 0

        if hh_count >= 1 and hl_count >= 1:
            result["structure"] = "UPTREND (HH/HL)"
            result["signal"] = "BULLISH"
            score = 40
            result["description"] = "Piyasa yapısı: Yükselen tepeler ve yükselen dipler."
        elif lh_count >= 1 and ll_count >= 1:
            result["structure"] = "DOWNTREND (LH/LL)"
            result["signal"] = "BEARISH"
            score = -40
            result["description"] = "Piyasa yapısı: Alçalan tepeler ve alçalan dipler."
        else:
            result["structure"] = "RANGING"
            result["description"] = "Piyasa yapısı: Yatay / kararsız."

        # BOS: Break of Structure (last swing broken)
        current_price = df['Close'].iloc[-1]
        last_swing_high = highs[peak_idx[-1]]
        last_swing_low = lows[trough_idx[-1]]

        if result["signal"] == "BULLISH" and current_price > last_swing_high:
            result["bos"] = True
            score += 20
            result["description"] += " BOS: Son direnç kırıldı!"
        elif result["signal"] == "BEARISH" and current_price < last_swing_low:
            result["bos"] = True
            score -= 20
            result["description"] += " BOS: Son destek kırıldı!"

        # CHoCH: Change of Character (structure changed direction)
        if len(peak_idx) >= 4 and len(trough_idx) >= 4:
            prev_highs = [highs[i] for i in peak_idx[-4:-1]]
            prev_lows = [lows[i] for i in trough_idx[-4:-1]]

            was_uptrend = prev_highs[-1] > prev_highs[-2] and prev_lows[-1] > prev_lows[-2]
            was_downtrend = prev_highs[-1] < prev_highs[-2] and prev_lows[-1] < prev_lows[-2]

            if was_uptrend and result["signal"] == "BEARISH":
                result["choch"] = True
                score -= 15
                result["description"] += " ⚠️ CHoCH: Yükselişten düşüşe dönüş!"
            elif was_downtrend and result["signal"] == "BULLISH":
                result["choch"] = True
                score += 15
                result["description"] += " ⚠️ CHoCH: Düşüşten yükselişe dönüş!"

        result["score"] = max(-100, min(100, score))

    except Exception as e:
        logger.debug(f"Market structure error: {e}")

    return result


# =============================================
# COMBINED ADVANCED SCORE (for signal_engine)
# =============================================
def calculate_advanced_score(df, timeframe="1d"):
    """
    Combine all advanced analyses into a single -100 to +100 score.
    Returns (score, reasons_list).
    """
    score = 0
    reasons = []

    # Elliott Wave (weight: 30%)
    ew = detect_elliott_wave(df)
    if ew["detected"]:
        ew_score = 0
        if ew["type"] == "IMPULSE":
            if ew["direction"] == "BULLISH":
                ew_score = -30  # Impulse complete = expect correction
                reasons.append(f"🌊 Elliott: Boğa impulsu tamamlandı — düzeltme bekleniyor")
            else:
                ew_score = 30
                reasons.append(f"🌊 Elliott: Ayı impulsu tamamlandı — toparlanma bekleniyor")
        elif ew["type"] == "CORRECTIVE":
            if ew["direction"] == "BEARISH":
                ew_score = 25  # Correction ending = buy
                reasons.append(f"🌊 Elliott: ABC düzeltme ({ew['direction']}) — dip fırsatı")
            else:
                ew_score = -25
                reasons.append(f"🌊 Elliott: ABC düzeltme ({ew['direction']}) — tepe riski")
        score += int(ew_score * 0.30)

    # Ichimoku (weight: 30%)
    ich = analyze_ichimoku(df)
    if ich["score"] != 0:
        score += int(ich["score"] * 0.30)
        if ich["signal"] == "BULLISH":
            reasons.append(f"☁️ Ichimoku: {ich['cloud_status']}")
        elif ich["signal"] == "BEARISH":
            reasons.append(f"☁️ Ichimoku: {ich['cloud_status']}")
        if ich["tk_cross"] and ("Kesişim" in ich["tk_cross"]):
            reasons.append(f"☁️ {ich['tk_cross']}")

    # Wyckoff (weight: 20%)
    wyck = detect_wyckoff_phase(df)
    if wyck["score"] != 0:
        score += int(wyck["score"] * 0.20)
        if wyck["phase"] != "UNKNOWN":
            reasons.append(f"📦 Wyckoff: {wyck['phase']}")

    # Market Structure (weight: 20%)
    ms = analyze_market_structure(df)
    if ms["score"] != 0:
        score += int(ms["score"] * 0.20)
        reasons.append(f"📐 {ms['description'][:60]}")
        if ms["bos"]:
            reasons.append("💥 Break of Structure tespit edildi")
        if ms["choch"]:
            reasons.append("⚠️ Change of Character tespit edildi")

    return max(-100, min(100, score)), reasons
