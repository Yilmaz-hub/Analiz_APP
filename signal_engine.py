"""
COMPOSITE DECISION ENGINE (Karar Motoru)
Combines 6 independent signal dimensions into a single clear verdict:
  GÜÇLÜ AL / AL / BEKLE / SAT / GÜÇLÜ SAT

Each dimension scores from -100 (strongly bearish) to +100 (strongly bullish).
Final verdict is a weighted average with confidence based on dimension agreement.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config import DecisionEngineConfig, IndicatorConfig, SignalConfig
from technical_analysis import (
    detect_rsi_divergence,
    calculate_trend_strength,
    calculate_sr_advanced,
    detect_patterns,
    detect_advanced_patterns
)
from advanced_analysis import calculate_advanced_score
from logger import logger


@dataclass
class CompositeSignal:
    verdict: str = "BEKLE"           # GÜÇLÜ AL | AL | BEKLE | SAT | GÜÇLÜ SAT
    confidence: float = 0.0          # 0-100%
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0       # Conservative (1.5:1 R:R)
    take_profit_2: float = 0.0       # Aggressive (3:1 R:R)
    risk_reward: float = 0.0
    risk_amount_pct: float = 0.0     # % risk from entry to SL
    reasons: list = field(default_factory=list)
    dimension_scores: dict = field(default_factory=dict)
    final_score: float = 0.0         # -100 to +100
    color: str = "gray"
    emoji: str = "⚪"
    timeframe: str = ""


# =============================================
# DIMENSION 1: TREND SCORE (-100 to +100)
# =============================================
def _score_trend(df):
    """EMA alignment, price vs EMA50, ADX, slope"""
    from technical_analysis import calculate_trend_strength
    score = calculate_trend_strength(df)
    reasons = []

    last = df.iloc[-1]
    price = last['Close']
    ema_20 = last.get('EMA_20', price)
    ema_50 = last.get('EMA_50', price)

    if price > ema_20 > ema_50:
        reasons.append("EMA sıralaması yükseliş yönünde (Fiyat > EMA20 > EMA50)")
    elif price < ema_20 < ema_50:
        reasons.append("EMA sıralaması düşüş yönünde (Fiyat < EMA20 < EMA50)")

    if score > 50:
        reasons.append(f"Güçlü yükseliş trendi (skor: {score:.0f})")
    elif score < -50:
        reasons.append(f"Güçlü düşüş trendi (skor: {score:.0f})")
    elif abs(score) < 15:
        reasons.append("Trend yatay / kararsız")

    return score, reasons


# =============================================
# DIMENSION 2: MOMENTUM SCORE (-100 to +100)
# =============================================
def _score_momentum(df):
    """RSI, RSI divergence, StochRSI, CCI, MACD"""
    if df is None or len(df) < 50:
        return 0, []

    score = 0
    reasons = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # --- RSI ---
    rsi = last.get('RSI', 50)
    if pd.isna(rsi):
        rsi = 50

    if rsi < 25:
        score += 40
        reasons.append(f"RSI aşırı satım bölgesinde ({rsi:.0f})")
    elif rsi < 35:
        score += 25
        reasons.append(f"RSI satım bölgesinde ({rsi:.0f})")
    elif rsi > 75:
        score -= 40
        reasons.append(f"RSI aşırı alım bölgesinde ({rsi:.0f})")
    elif rsi > 65:
        score -= 25
        reasons.append(f"RSI alım bölgesinde ({rsi:.0f})")

    # --- RSI Divergence ---
    divergence = detect_rsi_divergence(df)
    if divergence == "BULLISH":
        score += 30
        reasons.append("🔀 RSI Boğa Uyumsuzluğu (dip sinyali)")
    elif divergence == "BEARISH":
        score -= 30
        reasons.append("🔀 RSI Ayı Uyumsuzluğu (tepe sinyali)")

    # --- MACD ---
    macd_current = last.get('MACD')
    macd_prev = prev.get('MACD')
    macd_signal = last.get('MACD_Signal')

    if all(pd.notna([macd_current, macd_prev, macd_signal])):
        if macd_prev < macd_signal and macd_current > macd_signal:
            score += 25
            reasons.append("MACD yukarı kesişim ✂️")
        elif macd_prev > macd_signal and macd_current < macd_signal:
            score -= 25
            reasons.append("MACD aşağı kesişim ✂️")

        # MACD histogram direction
        hist_curr = macd_current - macd_signal
        hist_prev = macd_prev - prev.get('MACD_Signal', macd_signal)
        if not pd.isna(hist_prev):
            if hist_curr > hist_prev and hist_curr > 0:
                score += 10
            elif hist_curr < hist_prev and hist_curr < 0:
                score -= 10

    # --- Bollinger Bands ---
    bb_lower = last.get('BB_Lower', None)
    bb_upper = last.get('BB_Upper', None)
    price = last['Close']

    if bb_lower is not None and not pd.isna(bb_lower) and price < bb_lower:
        score += 15
        reasons.append("Fiyat Bollinger alt bandının altında")
    elif bb_upper is not None and not pd.isna(bb_upper) and price > bb_upper:
        score -= 15
        reasons.append("Fiyat Bollinger üst bandının üstünde")

    return max(-100, min(100, score)), reasons


# =============================================
# DIMENSION 3: VOLUME SCORE (-100 to +100)
# =============================================
def _score_volume(df):
    """Volume ratio, OBV direction, volume-price confirmation"""
    if df is None or len(df) < 20:
        return 0, []

    score = 0
    reasons = []

    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return 0, ["Hacim verisi yok"]

    last = df.iloc[-1]
    price = last['Close']
    prev_price = df['Close'].iloc[-2]

    # Volume ratio
    vol_avg = df['Volume'].rolling(IndicatorConfig.VOLUME_SMA_LENGTH).mean().iloc[-1]
    if pd.isna(vol_avg) or vol_avg == 0:
        return 0, []

    vol_ratio = last['Volume'] / vol_avg

    if vol_ratio > 2.0:
        # Very high volume — direction matters
        if price > prev_price:
            score += 40
            reasons.append(f"Çok yüksek hacimli alım (x{vol_ratio:.1f})")
        else:
            score -= 40
            reasons.append(f"Çok yüksek hacimli satış (x{vol_ratio:.1f})")
    elif vol_ratio > 1.5:
        if price > prev_price:
            score += 25
            reasons.append(f"Yüksek hacimli hareket (x{vol_ratio:.1f})")
        else:
            score -= 25
            reasons.append(f"Yüksek hacimli düşüş (x{vol_ratio:.1f})")
    elif vol_ratio < 0.5:
        score -= 10
        reasons.append("Düşük hacim — güvenilirlik azalır")

    # OBV direction (last 5 bars)
    try:
        obv = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).cumsum()
        obv_recent = obv.iloc[-5:]
        obv_slope = obv_recent.iloc[-1] - obv_recent.iloc[0]
        if obv_slope > 0:
            score += 15
        elif obv_slope < 0:
            score -= 15
    except Exception:
        pass

    return max(-100, min(100, score)), reasons


# =============================================
# DIMENSION 4: PATTERN SCORE (-100 to +100)
# =============================================
def _score_patterns(df):
    """Active chart patterns — W/M, candles, advanced formations"""
    if df is None or len(df) < 20:
        return 0, []

    score = 0
    reasons = []

    try:
        basic_patterns = detect_patterns(df)
        for pat in basic_patterns:
            name = pat.get('name', '')
            if name in ('İkili Dip (W)', 'Yutan Boğa', 'Hammer'):
                score += 30
                reasons.append(f"📊 {name} formasyonu (yükseliş)")
            elif name == 'İkili Tepe (M)':
                score -= 30
                reasons.append(f"📊 {name} formasyonu (düşüş)")
            elif name == 'Doji':
                reasons.append("⚠️ Doji — kararsızlık mumu")
    except Exception:
        pass

    try:
        advanced = detect_advanced_patterns(df)
        for pat in advanced:
            direction = pat.get('direction', 'NEUTRAL')
            confidence = pat.get('confidence', 50)
            name = pat.get('name', '')

            weight = confidence / 100 * 40  # Max 40 from advanced patterns

            if direction == 'BULLISH':
                score += weight
                reasons.append(f"📐 {name} (Boğa, güven: %{confidence})")
            elif direction == 'BEARISH':
                score -= weight
                reasons.append(f"📐 {name} (Ayı, güven: %{confidence})")
            elif direction == 'REVERSAL':
                # Reversal direction depends on current trend
                if len(df) >= 50:
                    trend_dir = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-20] else -1
                    score -= trend_dir * weight  # Reversal opposes current trend
                    reasons.append(f"📐 {name} (Dönüş, güven: %{confidence})")
    except Exception:
        pass

    return max(-100, min(100, score)), reasons


# =============================================
# DIMENSION 6: ADVANCED SCORE (-100 to +100)
# =============================================
def _score_advanced(df, timeframe="1d"):
    """Elliott Wave, Ichimoku Cloud, Wyckoff Phase, Market Structure"""
    try:
        return calculate_advanced_score(df, timeframe)
    except Exception as e:
        logger.debug(f"Advanced scoring error: {e}")
        return 0, ["Gelişmiş analiz hesaplanamadı"]


# =============================================
# DIMENSION 5: ML SCORE (-100 to +100)
# =============================================
def _score_ml(df):
    """ML model direction prediction"""
    if df is None or len(df) < 150:
        return 0, ["ML: Yetersiz veri"]

    try:
        from ml_models import calculate_ml_direction_signal
        ml_result = calculate_ml_direction_signal(df)

        if ml_result is None:
            return 0, ["ML: Model hesaplanamadı"]

        direction = ml_result['direction']
        confidence = ml_result['confidence']
        change_pct = ml_result['predicted_change_pct']

        # Convert ML output to -100..+100 score
        if direction == "BULLISH":
            score = min(100, confidence * 1.0)
            reasons = [f"🤖 AI: Yükseliş tahmini (%{confidence:.0f} güven, +%{change_pct:.2f})"]
        elif direction == "BEARISH":
            score = max(-100, -confidence * 1.0)
            reasons = [f"🤖 AI: Düşüş tahmini (%{confidence:.0f} güven, %{change_pct:.2f})"]
        else:
            score = 0
            reasons = [f"🤖 AI: Yön belirsiz (%{confidence:.0f} güven)"]

        return score, reasons

    except Exception as e:
        logger.debug(f"ML scoring error: {e}")
        return 0, ["ML: Hesaplama hatası"]


# =============================================
# COMPOSITE SIGNAL GENERATOR
# =============================================
def generate_composite_signal(df, timeframe="1d", supports=None, resistances=None):
    """
    Main entry point — generates a single CompositeSignal with clear verdict.
    """
    signal = CompositeSignal(timeframe=timeframe)

    if df is None or len(df) < 50:
        signal.verdict = "BEKLE"
        signal.reasons = ["Yetersiz veri"]
        return signal

    # Calculate support/resistance if not provided
    if supports is None or resistances is None:
        supports, resistances = calculate_sr_advanced(df, timeframe)

    last = df.iloc[-1]
    price = last['Close']
    atr = last.get('ATR', price * 0.02)
    if pd.isna(atr) or atr == 0:
        atr = price * 0.02

    # === SCORE ALL 6 DIMENSIONS ===
    trend_score, trend_reasons = _score_trend(df)
    momentum_score, momentum_reasons = _score_momentum(df)
    volume_score, volume_reasons = _score_volume(df)
    pattern_score, pattern_reasons = _score_patterns(df)
    ml_score, ml_reasons = _score_ml(df)
    advanced_score, advanced_reasons = _score_advanced(df, timeframe)

    signal.dimension_scores = {
        "trend": trend_score,
        "momentum": momentum_score,
        "volume": volume_score,
        "pattern": pattern_score,
        "ml": ml_score,
        "advanced": advanced_score
    }

    # === ADAPTIVE WEIGHTED COMPOSITE ===
    cfg = DecisionEngineConfig

    dim_weights = {
        "trend": (trend_score, cfg.TREND_WEIGHT),
        "momentum": (momentum_score, cfg.MOMENTUM_WEIGHT),
        "volume": (volume_score, cfg.VOLUME_WEIGHT),
        "pattern": (pattern_score, cfg.PATTERN_WEIGHT),
        "ml": (ml_score, cfg.ML_WEIGHT),
        "advanced": (advanced_score, cfg.ADVANCED_WEIGHT),
    }

    # Separate active (has meaningful data) vs inactive dimensions.
    # A dimension is "inactive" only if it returns exactly 0 AND its reasons
    # indicate data issues (not a genuine neutral score).
    active_dims = {}
    inactive_weight = 0.0

    for key, (score, weight) in dim_weights.items():
        if key == "ml" and score == 0 and any("Yetersiz" in r or "Hesaplama" in r or "hesaplanamadı" in r for r in ml_reasons):
            inactive_weight += weight
        elif key == "volume" and score == 0 and any("Hacim verisi yok" in r for r in volume_reasons):
            inactive_weight += weight
        else:
            active_dims[key] = (score, weight)

    # Redistribute inactive weight proportionally
    total_active_weight = sum(w for _, w in active_dims.values())
    if total_active_weight > 0:
        redistribution_factor = (total_active_weight + inactive_weight) / total_active_weight
    else:
        redistribution_factor = 1.0

    final_score = sum(score * weight * redistribution_factor for score, weight in active_dims.values())
    signal.final_score = final_score

    # === CONFIDENCE ===
    # Only count dimensions that have data (active dimensions)
    active_scores = [score for score, _ in active_dims.values()]
    non_zero = [s for s in active_scores if abs(s) > 5]

    if len(non_zero) > 0:
        same_direction = sum(1 for s in non_zero if (s > 0) == (final_score > 0))
        agreement_ratio = same_direction / len(non_zero)
    else:
        agreement_ratio = 0.5  # No strong signals = moderate confidence

    # Average magnitude of active dimensions only
    avg_magnitude = np.mean([abs(s) for s in active_scores]) if active_scores else 0

    confidence = (agreement_ratio * 60) + (min(avg_magnitude, 60) / 60 * 40)
    signal.confidence = max(0, min(100, confidence))

    # === COLLECT ALL REASONS ===
    all_reasons = trend_reasons + momentum_reasons + volume_reasons + pattern_reasons + ml_reasons + advanced_reasons
    signal.reasons = all_reasons

    # === VERDICT ===
    if signal.confidence < cfg.MIN_CONFIDENCE_TO_TRADE:
        signal.verdict = "BEKLE"
        signal.color = "gray"
        signal.emoji = "⚪"
        signal.reasons.insert(0, f"Güven düşük (%{signal.confidence:.0f}), işlem önerilmez")
    elif final_score >= cfg.STRONG_BUY_THRESHOLD:
        signal.verdict = "GÜÇLÜ AL"
        signal.color = "green"
        signal.emoji = "🟢"
    elif final_score >= cfg.BUY_THRESHOLD:
        signal.verdict = "AL"
        signal.color = "lightgreen"
        signal.emoji = "🟢"
    elif final_score <= cfg.STRONG_SELL_THRESHOLD:
        signal.verdict = "GÜÇLÜ SAT"
        signal.color = "red"
        signal.emoji = "🔴"
    elif final_score <= cfg.SELL_THRESHOLD:
        signal.verdict = "SAT"
        signal.color = "orange"
        signal.emoji = "🔴"
    else:
        signal.verdict = "BEKLE"
        signal.color = "gray"
        signal.emoji = "⚪"

    # === TRADE SETUP ===
    signal.entry_price = price

    if "AL" in signal.verdict:
        signal.stop_loss = price - (atr * cfg.ATR_SL_MULTIPLIER)
        risk = price - signal.stop_loss
        signal.take_profit_1 = price + (risk * cfg.CONSERVATIVE_RR)
        signal.take_profit_2 = price + (risk * cfg.AGGRESSIVE_RR)
        signal.risk_reward = cfg.CONSERVATIVE_RR
        signal.risk_amount_pct = (risk / price) * 100

    elif "SAT" in signal.verdict:
        signal.stop_loss = price + (atr * cfg.ATR_SL_MULTIPLIER)
        risk = signal.stop_loss - price
        signal.take_profit_1 = price - (risk * cfg.CONSERVATIVE_RR)
        signal.take_profit_2 = price - (risk * cfg.AGGRESSIVE_RR)
        signal.risk_reward = cfg.CONSERVATIVE_RR
        signal.risk_amount_pct = (risk / price) * 100

    return signal
