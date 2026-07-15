#CONFIG.PY
"""
ALL MAGIC NUMBERS AND TUNABLE PARAMETERS IN ONE PLACE FOR EASY MAINTENANCE AND TUNING
"""
#====================
#Data fetching and processing
#====================       


class DataFetchConfig:
    BINANCE_LIMIT = 1000
    OKX_LIMIT = 300
    YAHOO_PERIODS = {
        '1wk': '10y',
        '1d': '4y',
        '4h': '1mo'
    }
    REQUEST_TIMEOUT = 5 #Seconds    
    CACHE_TTL = 60 #Seconds,
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }   

class IndicatorConfig:
    RSI_LENGTH = 14
    EMA_SHORT = 20
    EMA_LONG = 50
    BOLLINGER_LENGTH = 20
    BOLLINGER_STD = 2
    ATR_LENGTH = 14 
    CCI_LENGTH = 20
    ADX_LENGTH = 14

    #MACD
    MACD_FAST = 12  
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    #Stochastic
    STOCH_RSI_LENGTH = 14
    STOCH_RSI_SMOOTH_K = 3
    STOCH_RSI_SMOOTH_D = 3

    #VOLUME
    VOLUME_SMA_LENGTH = 20
    HIGH_VOLUME_THRESHOLD = 1.5 #Times the average volume   
    LOW_VOLUME_THRESHOLD = 0.7 #Times the average volume

    #RISK MANAGEMENT    

class RiskConfig:
    ATR_SL_MULTIPLIER = 1.5
    ATR_TP_MULTIPLIER = 2.5
    ATR_FALLBACK_MULTIPLIER = 0.02  #2% of price if ATR is unavailable

    MAX_POSITION_SIZE = 0.4  #40% of account balance
    MAX_TOTAL_EXPOSURE = 0.8  #80% of account balance across all positions

    TRAILING_STOP_PCT = 0.05  #2% trailing stop
    TRAILING_ATR_MULTIPLIER = 1.2  #1.2 ATR trailing stop

class MLConfig:

    #train/test split
    TEST_SIZE_RATIO = 0.25

    #RANDOM FOREST
    RF_BACKTEST_ESTIMATORS = 250
    RF_BACKTEST_MAX_DEPTH = 10
    RF_BACKTEST_MIN_SAMPLES_SPLIT = 5

    #RANDOM FOREST PRODUCTION   
    RF_PROD_ESTIMATORS = 300
    RF_PROD_MAX_DEPTH = 12
    RF_PROD_MIN_SAMPLES_SPLIT = 5

    #ENSEMBLE
    RF_WEIGHT = 0.7  #Weight for Random Forest in ensemble models
    LR_WEIGHT = 0.3  #Weight for Logistic Regression in ensemble models

    #PREDICTION
    DEFAULT_FORECAST_PERIODS = 15  #Default number of periods to forecast
    CONFIDENCE_DECAY = 0.95  #Decay factor for confidence scores over time

    #FEATURES
    LAG_PERIODS = [1, 2, 3, 5]  #Lag periods for feature engineering
    RETURN_LOOKBACK = 5  #Number of periods to look back for return calculations
    VOLATILITY_WINDOW = 20  #Number of periods to look back for volatility

    #ACCURACY SCORES
    DIRECTION_WEIGHT = 0.6  #Weight for direction accuracy in overall score
    VOLATILITY_WEIGHT = 0.4  #Weight for volatility accuracy in overall score

    #SIMULATION WEIGHTS
    RSI_MAX = 85
    RSI_MIN = 15
    CCI_MAX = 200
    CCI_MIN = -200
    STOCH_MAX = 100
    STOCH_MIN = 0
    ADX_MIN = 10
    ADX_DECAY = 0.9
    VOLATILITY_SMOOTHING = 0.05
    MACD_UPDATE_FACTOR = 0.5
    MIN_CHANGE_THRESHOLD = 0.005  #Minimum change threshold for indicators to affect confidence scores

#SIGNAL GENERATION
class SignalConfig:
    #SCORE THRESHOLDS
    STRONG_BUY_THRESHOLD = 75
    BUY_THRESHOLD = 60
    STRONG_SELL_THRESHOLD = 25
    SELL_THRESHOLD = 40
    NEUTRAL_START_SCORE = 50

    #RSI ZONES
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    RSI_OVERSOLD_LIGHT = 40
    RSI_OVERBOUGHT_LIGHT = 60

    #SCORE ADJUSTMENTS
    RSI_OVERSOLD_SCORE = 30
    RSI_OVERBOUGHT_SCORE = -30
    RSI_OVERSOLD_LIGHT_SCORE = 15
    RSI_OVERBOUGHT_LIGHT_SCORE = -15    

    TREND_SCORE = 20
    BOLLINGER_SCORE = 15
    MACD_CROSS_SCORE = 10
    HIGH_VOLUME_SCORE = 10
    LOW_VOLUME_PENALTY = -5
    SUPPORT_PROXIMITY_SCORE = 20
    RESISTANCE_PROXIMITY_SCORE = -20

    #PROXIMITY THRESHOLDS
    SUPPORT_DISTANCE_THRESHOLD = 2.0  #2% distance from support level
    RESISTANCE_DISTANCE_THRESHOLD = 2.0  #2% distance from resistance level

# DECISION ENGINE (Composite Signal)
class DecisionEngineConfig:
    # Dimension weights (must sum to 1.0)
    TREND_WEIGHT = 0.25
    MOMENTUM_WEIGHT = 0.20
    VOLUME_WEIGHT = 0.10
    PATTERN_WEIGHT = 0.10
    ML_WEIGHT = 0.10
    ADVANCED_WEIGHT = 0.25  # Elliott Wave, Ichimoku, Wyckoff, Market Structure

    # Verdict thresholds (on -100 to +100 scale)
    STRONG_BUY_THRESHOLD = 35
    BUY_THRESHOLD = 15
    SELL_THRESHOLD = -15
    STRONG_SELL_THRESHOLD = -35

    # Risk management for trade setups
    CONSERVATIVE_RR = 1.5   # Risk:Reward for TP1
    AGGRESSIVE_RR = 3.0     # Risk:Reward for TP2
    ATR_SL_MULTIPLIER = 2.5 # ATR multiplier for stop-loss distance (1.5 -> 2.5: grid showed tight stops bled money on daily crypto)

    # RSI divergence detection
    DIVERGENCE_LOOKBACK = 20
    DIVERGENCE_MIN_SWING = 3  # Minimum bars between swings

    # Confidence calculation
    MIN_CONFIDENCE_TO_TRADE = 30  # Below this, always BEKLE

    # --- SIGNAL STABILITY / ANTI-WHIPSAW ---
    # The published verdict only changes direction after the raw signal
    # holds the new direction for CONFIRMATION_BARS consecutive CLOSED bars.
    CONFIRMATION_BARS = 2
    # How many recent closed bars to replay through the state machine
    # when producing the live (stable) signal. Must be > CONFIRMATION_BARS.
    STABILITY_LOOKBACK = 15
    # Hysteresis: once in AL, the score must fall below
    # BUY_THRESHOLD - EXIT_SCORE_BUFFER before the signal starts dropping
    # (mirrored for SAT). Prevents flip-flop around the entry threshold.
    EXIT_SCORE_BUFFER = 10
    # Score the last CLOSED candle only — the still-forming candle's
    # indicators change continuously and cause intraday verdict flips.
    DROP_UNCLOSED_CANDLE = True
    # Choppy-market guard used by the state machine for NEW entries
    CHOP_ADX_LIMIT = 20
    CHOP_SCORE_OVERRIDE = 60  # |score| above this ignores the chop guard

    # --- ENTRY QUALITY (backtest-validated 2026-07-15, 5 crypto assets 1d) ---
    # Asymmetric thresholds: a NEW signal needs |score| >= ENTRY_SCORE, but an
    # existing signal is held/exited via BUY/SELL_THRESHOLD ± EXIT_SCORE_BUFFER.
    # Grid result: entry 15 -> 25 improved every asset tested.
    ENTRY_SCORE = 25
    # Regime filter: long entries only when the last closed bar is above this
    # SMA (shorts only below). 0 disables. Backtest: improved every losing
    # asset, never hurt a winning one.
    REGIME_MA_PERIOD = 100

# BACKTEST EXECUTION MODEL
class BacktestConfig:
    # Trading fee charged per side (entry AND exit), as a fraction of trade
    # value. 0.001 = 0.1% (Binance spot taker). Without this the backtest
    # overstates returns by ~0.2% per round trip.
    FEE_RATE = 0.001

# TREND REGIME SCORING (scanner asset ranking)
class RegimeConfig:
    # The composite strategy only has an edge on trending assets (BTC/XRP
    # profile); choppy ones bleed. This 0-100 score ranks assets by trend
    # quality so the scanner can steer capital toward the former.
    MA_PERIOD = 100            # long-term SMA (matches DecisionEngineConfig.REGIME_MA_PERIOD)
    PERSISTENCE_LOOKBACK = 50  # bars used for "% of time above the MA"
    SLOPE_LOOKBACK = 20        # bars used for the MA slope
    SLOPE_FULL_SCORE_PCT = 5.0 # MA slope (%) over SLOPE_LOOKBACK that earns full slope points
    ADX_FLOOR = 15             # ADX at/below this earns 0 trend-strength points
    ADX_CEIL = 35              # ADX at/above this earns full trend-strength points

    # Component weights (must sum to 100)
    PERSISTENCE_POINTS = 40
    SLOPE_POINTS = 25
    ADX_POINTS = 20
    ALIGNMENT_POINTS = 15      # EMA20 > EMA50 > MA100 stack

    # Scanner behavior. Per-trade validation (2026-07-15, 5 assets, 72 trades):
    # entries below regime 35 netted ~0% combined; gating HIGHER than this
    # (e.g. 55) drops big early-trend winners because the score lags new
    # trends — do not raise it without re-running the per-trade analysis.
    MIN_TRADEABLE_SCORE = 35   # below this, no allocation is recommended
    # Opportunity rank = RANK_SIGNAL_WEIGHT * decision score
    #                  + RANK_REGIME_WEIGHT * (regime score mapped to -100..+100)
    RANK_SIGNAL_WEIGHT = 0.6
    RANK_REGIME_WEIGHT = 0.4

# POSITION SIZING (scanner allocation advice)
class SizingConfig:
    # Inverse-volatility sizing, validated 2026-07-15 on 3 samples (2 universes
    # x 2 windows): sizing positions by 1/vol beat equal-weight on return AND
    # drawdown in both recent samples and matched risk-adjusted return in the
    # 2019-2023 bull. 30-bar lookback beat 90-bar.
    VOL_LOOKBACK = 30      # bars of daily returns for realized volatility
    ADJ_MIN = 0.5          # calmest/wildest asset gets at most/least this
    ADJ_MAX = 1.5          #   multiple of the base allocation recommendation

# ADVANCED ANALYSIS
class AdvancedAnalysisConfig:
    # Elliott Wave
    ELLIOTT_PIVOT_ORDER = 10
    ELLIOTT_MIN_BARS = 80

    # Ichimoku Cloud
    ICHIMOKU_TENKAN = 9
    ICHIMOKU_KIJUN = 26
    ICHIMOKU_SENKOU_B = 52

    # Wyckoff
    WYCKOFF_LOOKBACK = 60

    # Market Structure
    MARKET_STRUCTURE_ORDER = 5

#SUPPORT AND RESISTANCE
class SRConfig:
    PIVOT_WINDOW = {   
        '4h': 5,  #5 days for daily data
        '1d': 28,  #28 periods for 4-hour data (equivalent to 14 days)
        '1wk': 52   #52 weeks for weekly data (equivalent to 1 year)

    }
    PIVOT_LOOKBACK = 500  #Look back at least 500 periods to find pivots

    #FIBONACCI
    FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]  #Common Fibonacci retracement levels

    #VOLUME PROFILE
    VOLUME_BINS = 50  #Number of price bins for volume profile
    TOP_VOLUME_LEVELS = 3  #Number of top volume levels to identify as support/resistance


#CONSTANTS
class Constants:
    OUNCE_TO_GRAMS = 31.1035
    DEFAULT_PORTFOLIO_BALANCE = 1000  #Default portfolio balance for backtesting and simulations
    DEFAULT_GRAM_GOLD_VOLUME = 1000  #Default value of 1 gram of gold in USD for simulations

    #PATTERN DETECTION
    W_M_PATTERN_TOLERANCE = 0.02  #2% tolerance for W and M pattern detection
    W_M_DISTANCE = 5  #Maximum distance in periods between the two peaks/troughs in W and M patterns
    DOJI_BODY_RATIO = 0.1  #Maximum body size as a percentage of the total candle range to be considered a doji
    HAMMER_WICK_RATIO = 2.0  #Minimum wick size as a multiple of the body size to be considered a hammer
    HAMMER_BODY_RATIO = 0.5  #Maximum body size as a percentage of the total candle range to be considered a hammer

    #PRICE ADJUSTMENT
    # 
    # Price adjustment factor to account for gold's unique characteristics in the model. This can be used to scale features or predictions to better fit gold's price behavior compared to other assets.
    GRAM_GOLD_HIGH_FACTOR = 1.002  #Factor to adjust high price for gram gold
    GRAM_GOLD_LOW_FACTOR = 0.998   #Factor to adjust low price for gram gold

#FILES
class FileConfig:
    PORTFOLIO_FILE = 'portfolio.json'
    ASSETS_FILE = 'varliklar.json'
    PAPER_FILE = 'paper_trading.json'

#telegram
class TelegramConfig:
    DEFAULT_TOKEN = " "
    DEFAULT_CHAT_ID = " "

#UI
class UIConfig:
    ZOOM_LEVELS = {
        '1wk': {'default': 30, 'full': 50},  #Zoom out for weekly data
        '1d': {'default': 60, 'full': 80},   #Default zoom for daily data
        '4h': {'default': 80, 'full': 100}    #Zoom in for 4-hour data

    }

# DEFAULT ASSETS
DEFAULT_COIN_MAP: dict[str, str] = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Ripple (XRP)": "XRP-USD",
    "Avax (AVAX)": "AVAX-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Pepe": "PEPE-USD",
    "ONS ALTIN ($)": "XAU_GOLD",
    "GRAM ALTIN (TL)": "GRAM_TRY",
    "EUR/USD": "EURUSD=X",
    "Türk Hava Yolları (THYAO)": "THYAO.IS",
    "Pegasus (PGSUS)": "PGSUS.IS"
}

# PATTERNS
PATTERN_INFO: dict[str, str] = {
    "İkili Dip (W)": "📉 **W Formasyonu:** Yükseliş sinyali.",
    "İkili Tepe (M)": "📈 **M Formasyonu:** Düşüş sinyali.",
    "Doji": "⚠️ **Doji:** Kararsızlık.",
    "Hammer": "🔨 **Çekiç:** Dipten dönüş.",
    "Yutan Boğa": "🚀 **Yutan Boğa:** Güçlü alım.",
    "RSI Pozitif Uyumsuzluk": "Boğa Uyumsuzluğu: Fiyat düşerken RSI yükseliyor. Yükseliş sinyali.",
    "RSI Negatif Uyumsuzluk": "Ayı Uyumsuzluğu: Fiyat yükselirken RSI düşüyor. Düşüş sinyali.",
    "Yüksek Hacimli Kırılım": "Hacimli Kırılım: Önemli bir seviye güçlü hacimle aşıldı.",
    "Trend Dönüşümü": "Trend Dönüşü: EMA50 kesişimi ve hacim onayı.",
    "Büyük Alım (Hacim)": "Büyük Alım: Ortalamanın çok üzerinde hacim. Yön arayışı.",
    "Büyük Satış (Hacim)": "Büyük Satış: Ortalamanın çok üzerinde satış hacmi.",
    "Üçgen Kırılımı": "Üçgen Kırılımı Formasyonu",
    "OBO": "Omuz Baş Omuz: Düşüş Formasyonu",
    "TOBO": "Ters Omuz Baş Omuz: Yükseliş Formasyonu",
    "Bayrak/Flama": "Bayrak/Flama Formasyonu",
    "Yükselen Takoz": "Yükselen Takoz: Düşüş Formasyonu",
    "Düşen Takoz": "Düşen Takoz: Yükseliş Formasyonu",
    "Harmonik ABCD": "Harmonik ABCD Formasyonu",
    "Harmonik Butterfly": "Harmonik Butterfly Formasyonu"
}
