import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from config import MLConfig, IndicatorConfig
from logger import logger

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_smart_prediction_FIXED(df, periods=15):
    """
    DATA LEAKAGE ÖNLENMİŞ VERSİYON
    - Scaler sadece train setine fit edilir
    - Test seti "görmemiş" gibi davranır
    - Simülasyon sırasında scaler sabittir
    - Model caching uygulanmıştır (saatte bir model baştan eğitilir)
    """
    try:
        work_df = df.copy()
        if len(work_df) < 150: return [], [], 0
        
        # === İNDİKATÖRLER ===
        work_df['RSI'] = work_df.ta.rsi(length=IndicatorConfig.RSI_LENGTH)
        work_df['CCI'] = work_df.ta.cci(length=IndicatorConfig.CCI_LENGTH)
        work_df['ATR'] = work_df.ta.atr(length=IndicatorConfig.ATR_LENGTH)

        macd = work_df.ta.macd(
            fast=IndicatorConfig.MACD_FAST,
            slow=IndicatorConfig.MACD_SLOW,
            signal=IndicatorConfig.MACD_SIGNAL
        )
        if macd is not None:
            work_df['MACD'] = macd['MACD_12_26_9']
            work_df['MACD_Signal'] = macd['MACDs_12_26_9']
        else:
            logger.warning("MACD calculation failed in ML prediction, setting to NaN")
            work_df['MACD'] = np.nan
            work_df['MACD_Signal'] = np.nan
        
        stoch = work_df.ta.stochrsi()
        work_df['StochRSI_K'] = stoch['STOCHRSIk_14_14_3_3'] if stoch is not None else 50
        
        if 'Volume' in work_df.columns and work_df['Volume'].sum() > 0:
            work_df['OBV'] = work_df.ta.obv()
            work_df['Volume_SMA'] = work_df['Volume'].rolling(20).mean()
            work_df['Volume_Ratio'] = work_df['Volume'] / work_df['Volume_SMA']
        else:
            work_df['OBV'] = 0
            work_df['Volume_Ratio'] = 1
            
        adx = work_df.ta.adx(length=14)
        work_df['ADX'] = adx['ADX_14'] if adx is not None else 25
        
        work_df['Price_SMA50'] = work_df['Close'].rolling(50).mean()
        work_df['Price_Distance'] = (work_df['Close'] - work_df['Price_SMA50']) / work_df['Price_SMA50'] * 100
        
        bb = work_df.ta.bbands(length=20, std=2)
        if bb is not None:
            work_df['BB_Width'] = (bb.iloc[:, -3] - bb.iloc[:, -5]) / work_df['Close'] * 100
        else:
            work_df['BB_Width'] = 2
            
        work_df['Return'] = work_df['Close'].pct_change()
        work_df['Return_5'] = work_df['Close'].pct_change(5)
        work_df['Volatility'] = work_df['Return'].rolling(20).std()
        
        for lag in [1, 2, 3, 5]:
            work_df[f'Lag{lag}'] = work_df['Return'].shift(lag)
            
        work_df['Target'] = work_df['Return'].shift(-1)
        
        work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        work_df.dropna(inplace=True)
        
        features = [
            'RSI', 'MACD', 'MACD_Signal', 'CCI', 'StochRSI_K',
            'ADX', 'BB_Width', 'Price_Distance',
            'Volume_Ratio', 'OBV',
            'Return', 'Return_5', 'Volatility',
            'Lag1', 'Lag2', 'Lag3', 'Lag5'
        ]
        
        X = work_df[features].values
        y = work_df['Target'].values
        
        # 1️⃣ ÖNCE VERİYİ BÖL (Ham haliyle)
        test_size = int(len(X) * MLConfig.TEST_SIZE_RATIO)
        X_train_raw = X[:-test_size]
        X_test_raw = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]
        
        # 2️⃣ SCALER'I SADECE TRAIN SETİNE GÖRE EĞİT
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # === BACKTEST ===
        rf_model = RandomForestRegressor(
            n_estimators=MLConfig.RF_BACKTEST_ESTIMATORS,
            max_depth=MLConfig.RF_BACKTEST_MAX_DEPTH,
            min_samples_split=MLConfig.RF_BACKTEST_MIN_SAMPLES_SPLIT,
            random_state=42
        )
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)
        ensemble_pred = MLConfig.RF_WEIGHT * rf_pred + MLConfig.LR_WEIGHT * lr_pred
        
        mae = mean_absolute_error(y_test, ensemble_pred)
        volatility = np.std(y_test)
        
        direction_acc = np.mean((ensemble_pred > 0) == (y_test > 0)) * 100
        volatility_penalty = min(mae / (volatility + 1e-6), 1.0)
        
        accuracy_score = (direction_acc * MLConfig.DIRECTION_WEIGHT) + ((1 - volatility_penalty) * (MLConfig.VOLATILITY_WEIGHT * 100))
        accuracy_score = max(0, min(100, accuracy_score))
        
        # === FINAL MODEL ===
        X_production_scaled = scaler.transform(X)
        rf_final = RandomForestRegressor(
            n_estimators=MLConfig.RF_PROD_ESTIMATORS,
            max_depth=MLConfig.RF_PROD_MAX_DEPTH,
            min_samples_split=MLConfig.RF_PROD_MIN_SAMPLES_SPLIT,
            random_state=42
        )
        lr_final = LinearRegression()
        rf_final.fit(X_production_scaled, y)
        lr_final.fit(X_production_scaled, y)
        
        # === GELECEK TAHMİNİ ===
        future_dates = []
        predictions = []
        
        last_date = work_df.index[-1]
        time_delta = work_df.index[-1] - work_df.index[-2]
        current_price = work_df['Close'].iloc[-1]
        
        sim_state = work_df[features].iloc[-1].copy()
        confidence_decay = MLConfig.CONFIDENCE_DECAY
        
        for step in range(1, periods + 1):
            next_date = last_date + (time_delta * step)
            future_dates.append(next_date)
            
            sim_input = scaler.transform([sim_state.values])
            rf_change = rf_final.predict(sim_input)[0]
            lr_change = lr_final.predict(sim_input)[0]
            pred_change = (MLConfig.RF_WEIGHT * rf_change + MLConfig.LR_WEIGHT * lr_change) * (accuracy_score / 100) * (confidence_decay ** step)
            
            next_price = current_price * (1 + pred_change)
            predictions.append(next_price)
            
            sim_state['Lag5'] = sim_state['Lag3']
            sim_state['Lag3'] = sim_state['Lag2']
            sim_state['Lag2'] = sim_state['Lag1']
            sim_state['Lag1'] = pred_change
            sim_state['Return'] = pred_change
            
            if pred_change > 0:
                sim_state['RSI'] = min(85, sim_state['RSI'] + (pred_change * 100))
                sim_state['CCI'] = min(200, sim_state['CCI'] + (pred_change * 200))
                sim_state['StochRSI_K'] = min(100, sim_state['StochRSI_K'] + 5)
            else:
                sim_state['RSI'] = max(15, sim_state['RSI'] + (pred_change * 100))
                sim_state['CCI'] = max(-200, sim_state['CCI'] + (pred_change * 200))
                sim_state['StochRSI_K'] = max(0, sim_state['StochRSI_K'] - 5)
            
            sim_state['MACD'] += pred_change * 0.5
            sim_state['Price_Distance'] = ((next_price - current_price) / current_price) * 100
            sim_state['Volatility'] = sim_state['Volatility'] * 0.95 + abs(pred_change) * 0.05
            
            if abs(pred_change) < 0.005:
                sim_state['ADX'] = max(10, sim_state['ADX'] * 0.9)
            
            current_price = next_price
        
        return future_dates, predictions, accuracy_score
        
    except Exception as e:
        logger.error(f"AI FIXED Error: {e}")
        return [], [], 0


@st.cache_data(ttl=3600, show_spinner=False)
def calculate_ml_direction_signal(df):
    """
    ML-based direction classification: BULLISH / BEARISH / NEUTRAL
    Uses RandomForestClassifier instead of regression.
    Returns: {"direction": str, "confidence": float, "predicted_change_pct": float}
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        work_df = df.copy()
        if len(work_df) < 150:
            return None

        # === FEATURES (same as prediction model) ===
        work_df['RSI'] = work_df.ta.rsi(length=IndicatorConfig.RSI_LENGTH)
        work_df['CCI'] = work_df.ta.cci(length=IndicatorConfig.CCI_LENGTH)
        work_df['ATR'] = work_df.ta.atr(length=IndicatorConfig.ATR_LENGTH)

        macd = work_df.ta.macd(
            fast=IndicatorConfig.MACD_FAST,
            slow=IndicatorConfig.MACD_SLOW,
            signal=IndicatorConfig.MACD_SIGNAL
        )
        if macd is not None:
            work_df['MACD'] = macd['MACD_12_26_9']
            work_df['MACD_Signal'] = macd['MACDs_12_26_9']
        else:
            work_df['MACD'] = np.nan
            work_df['MACD_Signal'] = np.nan

        stoch = work_df.ta.stochrsi()
        work_df['StochRSI_K'] = stoch['STOCHRSIk_14_14_3_3'] if stoch is not None else 50

        if 'Volume' in work_df.columns and work_df['Volume'].sum() > 0:
            work_df['OBV'] = work_df.ta.obv()
            work_df['Volume_SMA'] = work_df['Volume'].rolling(20).mean()
            work_df['Volume_Ratio'] = work_df['Volume'] / work_df['Volume_SMA']
        else:
            work_df['OBV'] = 0
            work_df['Volume_Ratio'] = 1

        adx = work_df.ta.adx(length=14)
        work_df['ADX'] = adx['ADX_14'] if adx is not None else 25

        work_df['Price_SMA50'] = work_df['Close'].rolling(50).mean()
        work_df['Price_Distance'] = (work_df['Close'] - work_df['Price_SMA50']) / work_df['Price_SMA50'] * 100

        bb = work_df.ta.bbands(length=20, std=2)
        if bb is not None:
            work_df['BB_Width'] = (bb.iloc[:, -3] - bb.iloc[:, -5]) / work_df['Close'] * 100
        else:
            work_df['BB_Width'] = 2

        work_df['Return'] = work_df['Close'].pct_change()
        work_df['Return_5'] = work_df['Close'].pct_change(5)
        work_df['Volatility'] = work_df['Return'].rolling(20).std()

        for lag in [1, 2, 3, 5]:
            work_df[f'Lag{lag}'] = work_df['Return'].shift(lag)

        # === CLASSIFICATION TARGET ===
        # Next-period return
        next_return = work_df['Close'].pct_change().shift(-1)
        # Classify: > +0.5% → BULLISH (1), < -0.5% → BEARISH (-1), else NEUTRAL (0)
        threshold = 0.005
        work_df['Direction'] = 0
        work_df.loc[next_return > threshold, 'Direction'] = 1
        work_df.loc[next_return < -threshold, 'Direction'] = -1
        work_df['Next_Return'] = next_return

        work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        work_df.dropna(inplace=True)

        features = [
            'RSI', 'MACD', 'MACD_Signal', 'CCI', 'StochRSI_K',
            'ADX', 'BB_Width', 'Price_Distance',
            'Volume_Ratio', 'OBV',
            'Return', 'Return_5', 'Volatility',
            'Lag1', 'Lag2', 'Lag3', 'Lag5'
        ]

        X = work_df[features].values
        y = work_df['Direction'].values

        if len(X) < 100:
            return None

        # Train on all data except last point
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X[:-1])
        y_train = y[:-1]
        X_last = scaler.transform(X[-1:])

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)

        # Predict
        prediction = clf.predict(X_last)[0]
        probabilities = clf.predict_proba(X_last)[0]
        classes = clf.classes_

        # Get confidence for the predicted class
        pred_idx = list(classes).index(prediction)
        confidence = probabilities[pred_idx] * 100

        # Predicted change (from regressor for magnitude)
        lr = LinearRegression()
        lr.fit(X_train, work_df['Next_Return'].values[:-1])
        predicted_change = lr.predict(X_last)[0] * 100  # as percentage

        if prediction == 1:
            direction = "BULLISH"
        elif prediction == -1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_change_pct": predicted_change
        }

    except Exception as e:
        logger.error(f"ML Direction Signal Error: {e}")
        return None
