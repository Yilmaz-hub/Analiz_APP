"""
COMPOSITE DECISION ENGINE (Karar Motoru)
Combines 6 independent signal dimensions into a single clear verdict:
  GÜÇLÜ AL / AL / BEKLE / SAT / GÜÇLÜ SAT

Each dimension scores from -100 (strongly bearish) to +100 (strongly bullish).
Final verdict is a weighted average with confidence based on dimension agreement.
"""

import threading
from collections import OrderedDict

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config import DecisionEngineConfig, IndicatorConfig
from technical_analysis import (
    detect_rsi_divergence,
    calculate_trend_strength,
    detect_patterns,
    detect_advanced_patterns
)
from advanced_analysis import calculate_advanced_score
from logger import logger


class _BoundedCache:
    """Thread-safe, fixed-size LRU cache. Evicts only the single
    least-recently-used entry on overflow — unlike a plain dict manually
    cleared in full once it passes a size threshold, which drops every
    cached signal at once and forces every concurrent user's next render to
    recompute simultaneously."""

    def __init__(self, maxsize=256):
        self._maxsize = maxsize
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


def _weights_fingerprint(weights):
    """Turn a weights dict into a hashable tuple for cache keys, or None
    when weights is None (the default-config case) — so a cache entry
    computed under one weight vector is never returned for another."""
    if weights is None:
        return None
    return tuple(round(float(weights[k]), 6) for k in
                 ("trend", "momentum", "volume", "pattern", "ml", "advanced"))


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
    raw_verdict: str = "BEKLE"       # Unfiltered verdict of the last closed bar (informational)
    bars_held: int = 0               # How many closed bars the confirmed verdict has been stable


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
# PER-BAR SCORING (shared by live signal & backtest)
# =============================================
def _compute_bar_score(df, timeframe="1d", include_ml=True, weights=None):
    """
    Score the LAST bar of `df` across all 6 dimensions and combine them into
    a weighted composite score + confidence. Pure computation — no verdict
    mapping, no state. Returns None when there is not enough data.

    include_ml=False skips the RandomForest dimension (its weight is
    redistributed to the other dimensions) — required for bar-by-bar
    backtesting where training a model per bar would be far too slow.

    weights: optional dict with keys trend/momentum/volume/pattern/ml/advanced
    overriding DecisionEngineConfig's fixed weights (e.g. a per-asset-class
    profile from weight_profiles.py). None (default) uses the config values,
    unchanged from today's behavior.
    """
    if df is None or len(df) < 50:
        return None

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
    if include_ml:
        ml_score, ml_reasons = _score_ml(df)
    else:
        ml_score, ml_reasons = 0, ["ML: Devre dışı"]
    advanced_score, advanced_reasons = _score_advanced(df, timeframe)

    dimension_scores = {
        "trend": trend_score,
        "momentum": momentum_score,
        "volume": volume_score,
        "pattern": pattern_score,
        "ml": ml_score,
        "advanced": advanced_score
    }

    # === ADAPTIVE WEIGHTED COMPOSITE ===
    cfg = DecisionEngineConfig
    w = weights or {
        "trend": cfg.TREND_WEIGHT, "momentum": cfg.MOMENTUM_WEIGHT,
        "volume": cfg.VOLUME_WEIGHT, "pattern": cfg.PATTERN_WEIGHT,
        "ml": cfg.ML_WEIGHT, "advanced": cfg.ADVANCED_WEIGHT,
    }

    dim_weights = {
        "trend": (trend_score, w["trend"]),
        "momentum": (momentum_score, w["momentum"]),
        "volume": (volume_score, w["volume"]),
        "pattern": (pattern_score, w["pattern"]),
        "ml": (ml_score, w["ml"]),
        "advanced": (advanced_score, w["advanced"]),
    }

    # Separate active (has meaningful data) vs inactive dimensions.
    # A dimension is "inactive" only if it returns exactly 0 AND its reasons
    # indicate data issues (not a genuine neutral score).
    active_dims = {}
    inactive_weight = 0.0

    for key, (score, weight) in dim_weights.items():
        if key == "ml" and score == 0 and any("Yetersiz" in r or "Hesaplama" in r or "hesaplanamadı" in r or "Devre dışı" in r for r in ml_reasons):
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
    confidence = max(0, min(100, confidence))

    all_reasons = trend_reasons + momentum_reasons + volume_reasons + pattern_reasons + ml_reasons + advanced_reasons

    rsi_val = last.get('RSI', 50)
    if pd.isna(rsi_val): rsi_val = 50
    adx_val = last.get('ADX', 25)
    if pd.isna(adx_val): adx_val = 25

    # Regime: +1 above the long-term SMA, -1 below, 0 unknown/disabled.
    # The state machine only allows long entries at +1 and shorts at -1.
    regime = 0
    ma_period = getattr(cfg, "REGIME_MA_PERIOD", 0)
    if ma_period and len(df) >= ma_period:
        regime_ma = df['Close'].iloc[-ma_period:].mean()
        if not pd.isna(regime_ma):
            regime = 1 if price > regime_ma else -1

    return {
        "score": final_score,
        "confidence": confidence,
        "rsi": rsi_val,
        "adx": adx_val,
        "price": price,
        "atr": atr,
        "regime": regime,
        "dimension_scores": dimension_scores,
        "reasons": all_reasons,
    }


def _apply_trade_setup(signal, price, atr, cfg):
    """Fill entry/SL/TP fields on an actionable (AL/SAT) signal."""
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


# =============================================
# SIGNAL STATE MACHINE (anti-whipsaw core)
# =============================================
class SignalStateMachine:
    """
    Whipsaw filter shared by the live signal and the backtest, so backtest
    results reflect exactly what the dashboard will do. Feed one CLOSED bar
    at a time via update(). State: +1 long / 0 flat / -1 short.

    Three mechanisms (all backtest-validated together, 2026-07-15):
      1. Confirmation bars — the direction state only changes after the raw
         direction disagrees with it for CONFIRMATION_BARS consecutive bars
         (raw direction uses the BUY/SELL ±15 thresholds).
      2. Exit hysteresis — once in a direction, the state is held while the
         score stays inside the hold band (threshold minus EXIT_SCORE_BUFFER),
         even if fresh-entry conditions lapsed.
      3. Arming latch — a confirmed direction only becomes an actionable
         signal (`.signal`) once a bar reaches |score| >= ENTRY_SCORE with the
         regime filter agreeing (+1 above the long-term SMA for longs). Once
         armed it stays armed until the direction state changes, so the
         published verdict doesn't flicker.

    Entry guards (chop filter, RSI overextension, low confidence, entry
    score, regime) only block NEW signals; they never force an exit.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or DecisionEngineConfig
        self.state = 0          # confirmed direction: +1 / 0 / -1
        self.armed = False      # direction reached ENTRY_SCORE + regime OK
        self.bars_held = 0
        self._candidate = 0
        self._candidate_bars = 0

    def _raw_direction(self, score, confidence, rsi, adx):
        cfg = self.cfg
        if confidence < cfg.MIN_CONFIDENCE_TO_TRADE:
            return 0
        if adx < cfg.CHOP_ADX_LIMIT and abs(score) < cfg.CHOP_SCORE_OVERRIDE:
            return 0  # choppy market — don't open fresh positions
        if score >= cfg.BUY_THRESHOLD and rsi <= 70:
            return 1
        if score <= cfg.SELL_THRESHOLD and rsi >= 30:
            return -1
        return 0

    def update(self, score, confidence, rsi, adx, regime=0):
        """Returns (confirmed_state, raw_direction) after this bar."""
        raw = self._raw_direction(score, confidence, rsi, adx)

        # Exit hysteresis: hold an existing signal while the score is still
        # inside the hold band (raw only went flat, not opposite). Entry-only
        # guards (chop, RSI) don't evict a held signal.
        if self.state == 1 and raw == 0 and score > self.cfg.BUY_THRESHOLD - self.cfg.EXIT_SCORE_BUFFER:
            raw = 1
        elif self.state == -1 and raw == 0 and score < self.cfg.SELL_THRESHOLD + self.cfg.EXIT_SCORE_BUFFER:
            raw = -1

        if raw == self.state:
            self.bars_held += 1
            self._candidate = self.state
            self._candidate_bars = 0
        else:
            if raw == self._candidate:
                self._candidate_bars += 1
            else:
                self._candidate = raw
                self._candidate_bars = 1
            if self._candidate_bars >= self.cfg.CONFIRMATION_BARS:
                self.state = self._candidate
                self.bars_held = self._candidate_bars
                self._candidate_bars = 0
                self.armed = False  # new direction must re-arm

        # Arming: the confirmed direction becomes actionable once one bar
        # clears the entry score with the regime agreeing. Latched until the
        # direction state changes.
        entry_score = getattr(self.cfg, "ENTRY_SCORE", self.cfg.BUY_THRESHOLD)
        if self.state == 1 and score >= entry_score and regime >= 0:
            self.armed = True
        elif self.state == -1 and score <= -entry_score and regime <= 0:
            self.armed = True
        elif self.state == 0:
            self.armed = False

        return self.state, raw

    @property
    def signal(self):
        """Actionable direction: state if armed, else 0."""
        return self.state if self.armed else 0

    @property
    def pending_bars(self):
        """Confirmation progress of a pending direction change (0 if none)."""
        return self._candidate_bars


# =============================================
# COMPOSITE SIGNAL GENERATOR (raw, single-bar)
# =============================================
def generate_composite_signal(df, timeframe="1d", supports=None, resistances=None, include_ml=True, weights=None):
    """
    Raw single-bar verdict — scores the latest bar with no persistence.
    Kept for compatibility/diagnostics; the UI should prefer
    generate_stable_signal(), which adds the anti-whipsaw layer.
    (supports/resistances are accepted for backward compatibility but the
    dimension scores never used them.)
    """
    signal = CompositeSignal(timeframe=timeframe)
    cfg = DecisionEngineConfig

    bar = _compute_bar_score(df, timeframe, include_ml, weights=weights)
    if bar is None:
        signal.verdict = "BEKLE"
        signal.reasons = ["Yetersiz veri"]
        return signal

    final_score = bar["score"]
    signal.final_score = final_score
    signal.confidence = bar["confidence"]
    signal.dimension_scores = bar["dimension_scores"]
    signal.reasons = list(bar["reasons"])

    price = bar["price"]
    atr = bar["atr"]
    rsi_val = bar["rsi"]
    adx_val = bar["adx"]

    if signal.confidence < cfg.MIN_CONFIDENCE_TO_TRADE:
        signal.verdict = "BEKLE"
        signal.color = "gray"
        signal.emoji = "⚪"
        signal.reasons.insert(0, f"Güven düşük (%{signal.confidence:.0f}), işlem önerilmez")
    elif adx_val < 20 and abs(final_score) < 60:
        # Choppy market filter: Ignore weak signals if ADX is very low
        signal.verdict = "BEKLE"
        signal.color = "gray"
        signal.emoji = "⚪"
        signal.reasons.insert(0, f"Yatay piyasa tespit edildi (ADX: {adx_val:.1f}), sahte kırılım (whipsaw) riski!")
    elif final_score >= cfg.STRONG_BUY_THRESHOLD:
        if rsi_val > 75:
            signal.verdict = "BEKLE"
            signal.color = "gray"
            signal.emoji = "⚪"
            signal.reasons.insert(0, f"Güçlü Alım sinyali var ancak fiyat çok şişmiş (RSI: {rsi_val:.1f}), düzeltme beklenebilir.")
        else:
            signal.verdict = "GÜÇLÜ AL"
            signal.color = "green"
            signal.emoji = "🟢"
    elif final_score >= cfg.BUY_THRESHOLD:
        if rsi_val > 70:
            signal.verdict = "BEKLE"
            signal.color = "gray"
            signal.emoji = "⚪"
            signal.reasons.insert(0, f"Alım sinyali var ancak fiyat şişmiş (RSI: {rsi_val:.1f}), riskli giriş.")
        else:
            signal.verdict = "AL"
            signal.color = "lightgreen"
            signal.emoji = "🟢"
    elif final_score <= cfg.STRONG_SELL_THRESHOLD:
        if rsi_val < 25:
            signal.verdict = "BEKLE"
            signal.color = "gray"
            signal.emoji = "⚪"
            signal.reasons.insert(0, f"Güçlü Satış sinyali var ancak fiyat çok düşmüş (RSI: {rsi_val:.1f}), tepki yükselişi gelebilir.")
        else:
            signal.verdict = "GÜÇLÜ SAT"
            signal.color = "red"
            signal.emoji = "🔴"
    elif final_score <= cfg.SELL_THRESHOLD:
        if rsi_val < 30:
            signal.verdict = "BEKLE"
            signal.color = "gray"
            signal.emoji = "⚪"
            signal.reasons.insert(0, f"Satış sinyali var ancak fiyat dipte (RSI: {rsi_val:.1f}), riskli açığa satış.")
        else:
            signal.verdict = "SAT"
            signal.color = "orange"
            signal.emoji = "🔴"
    else:
        signal.verdict = "BEKLE"
        signal.color = "gray"
        signal.emoji = "⚪"

    # === TRADE SETUP ===
    signal.raw_verdict = signal.verdict
    _apply_trade_setup(signal, price, atr, cfg)
    return signal


# =============================================
# STABLE SIGNAL (public entry point for UI/scanner)
# =============================================
_stable_cache = _BoundedCache(maxsize=256)

# Per-bar dimension-score cache. STABILITY_LOOKBACK bars are replayed on
# every generate_stable_signal() call; as new candles close, that window
# slides forward by one bar but mostly re-scores the SAME historical bars
# it scored on the previous call. This reuses those instead of rerunning
# the full 6-dimension analysis (pattern detection, Elliott/Ichimoku/
# Wyckoff/market-structure, etc.) for bars whose data hasn't changed.
# Sized larger than _stable_cache since entries are per-(asset, timeframe,
# bar) rather than per-(asset, timeframe) final result.
_bar_score_cache = _BoundedCache(maxsize=4096)


def _cached_bar_score(df_slice, timeframe, include_ml, weights=None):
    try:
        key = (timeframe, str(df_slice.index[-1]),
               round(float(df_slice['Close'].iloc[-1]), 8), len(df_slice), include_ml,
               _weights_fingerprint(weights))
    except Exception:
        key = None

    if key is not None:
        cached = _bar_score_cache.get(key)
        if cached is not None:
            return cached

    result = _compute_bar_score(df_slice, timeframe, include_ml=include_ml, weights=weights)
    if key is not None and result is not None:
        _bar_score_cache.set(key, result)
    return result


def generate_stable_signal(df, timeframe="1d", supports=None, resistances=None, include_ml=True, weights=None):
    """
    Whipsaw-resistant signal — this is what the UI should display and what
    run_strategy_backtest() trades, so backtest numbers match live behavior.

      1. Scores CLOSED candles only (drops the still-forming last bar), so
         the verdict cannot flip intraday from a half-formed candle.
      2. Replays the last STABILITY_LOOKBACK closed bars through
         SignalStateMachine: a direction change needs CONFIRMATION_BARS of
         agreement, and exits have score hysteresis.

    Deterministic between candle closes — Streamlit reruns always reproduce
    the same verdict until a new candle closes.

    ML runs only on the final bar (replay bars skip it for speed; its 10%
    weight is redistributed there).

    weights: optional per-asset-class dimension weights (see
    _compute_bar_score). None uses DecisionEngineConfig defaults.
    """
    cfg = DecisionEngineConfig
    signal = CompositeSignal(timeframe=timeframe)

    work = df
    if cfg.DROP_UNCLOSED_CANDLE and df is not None and len(df) > 1:
        work = df.iloc[:-1]

    if work is None or len(work) < 55:
        signal.verdict = "BEKLE"
        signal.reasons = ["Yetersiz veri"]
        return signal

    # Output only changes when a new candle closes — cache on the last bar.
    try:
        cache_key = (timeframe, str(work.index[-1]),
                     round(float(work['Close'].iloc[-1]), 8), len(work), include_ml,
                     _weights_fingerprint(weights))
    except Exception:
        cache_key = None
    if cache_key is not None:
        cached_signal = _stable_cache.get(cache_key)
        if cached_signal is not None:
            return cached_signal

    machine = SignalStateMachine(cfg)
    lookback = min(cfg.STABILITY_LOOKBACK, len(work) - 50)
    bar = None
    raw_dir = 0

    for k in range(lookback - 1, -1, -1):
        s = work.iloc[:len(work) - k]
        is_final = (k == 0)
        b = _cached_bar_score(s, timeframe, include_ml=(include_ml and is_final), weights=weights)
        if b is None:
            continue
        _, raw_dir = machine.update(b["score"], b["confidence"], b["rsi"], b["adx"], b.get("regime", 0))
        if is_final:
            bar = b

    if bar is None:
        signal.verdict = "BEKLE"
        signal.reasons = ["Yetersiz veri"]
        return signal

    signal.final_score = bar["score"]
    signal.confidence = bar["confidence"]
    signal.dimension_scores = bar["dimension_scores"]
    signal.reasons = list(bar["reasons"])
    signal.bars_held = machine.bars_held

    raw_map = {1: "AL", 0: "BEKLE", -1: "SAT"}
    signal.raw_verdict = raw_map[raw_dir]

    # Actionable verdict = confirmed direction AND armed (entry score + regime
    # reached at least once). Strength label from latest score.
    if machine.signal == 1:
        strong = bar["score"] >= cfg.STRONG_BUY_THRESHOLD
        signal.verdict = "GÜÇLÜ AL" if strong else "AL"
        signal.color = "green" if strong else "lightgreen"
        signal.emoji = "🟢"
    elif machine.signal == -1:
        strong = bar["score"] <= cfg.STRONG_SELL_THRESHOLD
        signal.verdict = "GÜÇLÜ SAT" if strong else "SAT"
        signal.color = "red" if strong else "orange"
        signal.emoji = "🔴"
    else:
        signal.verdict = "BEKLE"
        signal.color = "gray"
        signal.emoji = "⚪"

    # Explain the anti-whipsaw layer's decision so the lag is understandable
    entry_score = getattr(cfg, "ENTRY_SCORE", cfg.BUY_THRESHOLD)
    if machine.state != 0 and not machine.armed:
        if bar.get("regime", 0) * machine.state < 0:
            neden = "fiyat uzun vadeli ortalamanın ters tarafında (rejim filtresi)"
        else:
            neden = f"skor giriş eşiğine (±{entry_score}) ulaşmadı"
        signal.reasons.insert(
            0, f"⏳ Yön {raw_map[machine.state]} teyitli ama işlem sinyali yok: {neden}")
    elif raw_dir != machine.state:
        if machine.state == 0:
            signal.reasons.insert(
                0, f"⏳ Ham sinyal {raw_map[raw_dir]}, onay bekleniyor "
                   f"({machine.pending_bars}/{cfg.CONFIRMATION_BARS} kapanmış bar)")
        else:
            signal.reasons.insert(
                0, f"🔒 {signal.verdict} korunuyor — ham sinyal {raw_map[raw_dir]} oldu ama "
                   f"{cfg.CONFIRMATION_BARS} bar teyit gerekiyor (histerezis)")
    elif machine.state != 0:
        signal.reasons.insert(0, f"✅ Sinyal {machine.bars_held} kapanmış bardır stabil")

    _apply_trade_setup(signal, bar["price"], bar["atr"], cfg)

    if cache_key is not None:
        _stable_cache.set(cache_key, signal)
    return signal
