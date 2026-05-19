#!/usr/bin/env python3
"""
LightGBM ML model for forex signal confidence
Predicts probability that next bar closes higher (for BUY bias)
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModel:
    """LightGBM classifier for trade confidence scoring"""
    
    def __init__(self, model_path='config/ml_model.pkl'):
        self.model = None
        self.model_path = model_path
        self.feature_names = None
    
    def create_features(self, df, lookback=20):
        """
        Create 34 features from OHLCV + optional cross-asset data.
        df: DataFrame with columns [open, high, low, close, volume]
            Optionally also: [dxy_close, gold_close, tnx_close, vix_close]
        """
        
        features = pd.DataFrame(index=df.index)
        
        # --- Price momentum (5 features) ---
        features['return_1bar'] = df['close'].pct_change(1)
        features['return_5bar'] = df['close'].pct_change(5)
        features['return_10bar'] = df['close'].pct_change(10)
        features['return_20bar'] = df['close'].pct_change(20)
        features['close_vs_ema20'] = (df['close'] / df['close'].ewm(span=20).mean()) - 1
        
        # --- Volatility (5 features) ---
        features['atr'] = self._atr(df['high'], df['low'], df['close'], 14)
        features['atr_pct'] = features['atr'] / df['close']
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['range_pct'] = (df['high'] - df['low']) / df['open']
        
        # --- Indicators (10 features) ---
        features['rsi_14'] = self._rsi(df['close'], 14)
        features['adx_14'] = self._adx(df['high'], df['low'], df['close'], 14)[0]
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(df['close'], 20, 2.0)
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)  # 0=lower, 1=upper
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid
        
        macd, macd_signal, macd_hist = self._macd(df['close'], 12, 26, 9)
        features['macd'] = macd
        features['macd_histogram'] = macd_hist
        
        stoch_k, stoch_d = self._stochastic(df['high'], df['low'], df['close'], 14, 3, 3)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_position'] = stoch_k / 100  # 0=oversold, 1=overbought
        
        # --- Volume (3 features) ---
        vol = df['volume'].replace(0, np.nan).fillna(1)  # forex volume is often 0 from Yahoo
        features['volume_sma_ratio'] = vol / vol.rolling(20).mean()
        features['volume_trend'] = vol.pct_change(5)
        features['price_volume_trend'] = (df['close'].pct_change() * vol).rolling(10).mean()
        
        # --- Time features (4 features) ---
        if 'date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            if isinstance(df.index, pd.DatetimeIndex):
                dates = df.index
            else:
                dates = pd.DatetimeIndex(df['date'])

            features['hour_sin'] = np.sin(2 * np.pi * dates.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * dates.hour / 24)
            features['day_of_week'] = dates.dayofweek / 7
            features['day_of_month'] = dates.day / 31
        else:
            features['hour_sin'] = 0
            features['hour_cos'] = 0
            features['day_of_week'] = 0
            features['day_of_month'] = 0
        
        # --- Cross-asset features (8 features) ---
        # Real values when columns are present; 0 fallback keeps feature set consistent.
        if 'dxy_close' in df.columns:
            features['dxy_return_1d'] = df['dxy_close'].pct_change(1)
            features['dxy_return_5d'] = df['dxy_close'].pct_change(5)
        else:
            features['dxy_return_1d'] = 0.0
            features['dxy_return_5d'] = 0.0

        if 'gold_close' in df.columns:
            features['gold_return_1d'] = df['gold_close'].pct_change(1)
            features['gold_return_5d'] = df['gold_close'].pct_change(5)
        else:
            features['gold_return_1d'] = 0.0
            features['gold_return_5d'] = 0.0

        if 'tnx_close' in df.columns:
            features['tnx_return_1d'] = df['tnx_close'].pct_change(1)
            features['tnx_return_5d'] = df['tnx_close'].pct_change(5)
        else:
            features['tnx_return_1d'] = 0.0
            features['tnx_return_5d'] = 0.0

        if 'vix_close' in df.columns:
            features['vix_level'] = df['vix_close'] / 20.0  # normalise around typical VIX
            features['vix_change_1d'] = df['vix_close'].pct_change(1)
        else:
            features['vix_level'] = 0.0
            features['vix_change_1d'] = 0.0

        # --- Target: did price go up next bar? (0/1) ---
        features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop rows with NaN
        features = features.dropna()
        
        logger.info(f"Created {len(features.columns)-1} features, {len(features)} samples")
        return features
    
    def train(self, df, test_size=0.2):
        """Train LightGBM model"""
        
        logger.info("Creating features...")
        features = self.create_features(df)
        
        # Separate features and target
        X = features.drop('target', axis=1)
        y = features['target']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)}")
        
        # Train LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)]
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model trained | Accuracy: {accuracy:.2%} | AUC: {auc:.2%}")
        
        # Save model and feature names together
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return accuracy, auc
    
    def predict_confidence(self, df):
        """Predict confidence (0-1) for next bar"""
        if self.model is None:
            logger.error("Model not trained")
            return 0.5
        
        try:
            features = self.create_features(df)
            if len(features) == 0:
                return 0.5
            
            X = features[self.feature_names].iloc[-1:]
            confidence = self.model.predict_proba(X)[0][1]
            return confidence
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return 0.5
    
    def load(self):
        """Load pre-trained model"""
        try:
            payload = joblib.load(self.model_path)
            self.model = payload['model']
            self.feature_names = payload['feature_names']
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except:
            logger.warning(f"Could not load model from {self.model_path}")
            return False
    
    # Helper indicator functions
    @staticmethod
    def _rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    @staticmethod
    def _adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        di_plus = 100 * (plus_dm.rolling(period).mean() / atr)
        di_minus = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        return adx, di_plus, di_minus
    
    @staticmethod
    def _bollinger_bands(data, period=20, std_dev=2.0):
        sma = data.rolling(period).mean()
        std = data.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def _macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _stochastic(high, low, close, k_period=14, k_smooth=3, d_smooth=3):
        low_min = low.rolling(k_period).min()
        high_max = high.rolling(k_period).max()
        k_raw = 100 * (close - low_min) / (high_max - low_min)
        k = k_raw.rolling(k_smooth).mean()
        d = k.rolling(d_smooth).mean()
        return k, d

print("MLModel module loaded")
