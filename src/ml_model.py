#!/usr/bin/env python3
"""
LightGBM ML model for forex signal confidence.
Feature engineering must stay in sync with retrain_model.py.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)

CROSS_ASSET_NAMES = ('dxy', 'gold', 'tnx', 'vix')


def make_features(df, spread_pips=1.0):
    """
    Build feature matrix from an OHLCV DataFrame (DatetimeIndex).
    Cross-asset columns expected as {name}_close (e.g. dxy_close).
    Returns DataFrame without NaN rows; includes 'target' column.
    """
    f = pd.DataFrame(index=df.index)

    # Price momentum
    for n in [1, 3, 5, 10, 20]:
        f[f'ret_{n}'] = df['close'].pct_change(n)
    f['close_vs_ema20'] = df['close'] / df['close'].ewm(span=20).mean() - 1
    f['close_vs_ema50'] = df['close'] / df['close'].ewm(span=50).mean() - 1

    # Volatility / ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    f['atr14']     = tr.rolling(14).mean()
    f['atr14_pct'] = f['atr14'] / df['close']
    f['vol20']     = df['close'].pct_change().rolling(20).std()
    f['hl_ratio']  = (df['high'] - df['low']) / df['close']
    f['atr_ratio'] = f['atr14'] / f['atr14'].rolling(20).mean()

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Bollinger Bands
    bb_mid   = df['close'].rolling(20).mean()
    bb_std   = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    f['bb_pos']   = (df['close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    f['bb_width'] = (bb_upper - bb_lower) / bb_mid

    # MACD
    ema12  = df['close'].ewm(span=12).mean()
    ema26  = df['close'].ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    f['macd_hist'] = macd - signal
    f['macd_norm'] = (macd - signal) / df['close']

    # Stochastic
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    stoch_k = 100 * (df['close'] - lo14) / (hi14 - lo14).replace(0, np.nan)
    f['stoch_k'] = stoch_k.rolling(3).mean()
    f['stoch_d'] = f['stoch_k'].rolling(3).mean()

    # ADX
    plus_dm  = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    atr14    = tr.rolling(14).mean()
    di_plus  = 100 * plus_dm.rolling(14).mean()  / atr14.replace(0, np.nan)
    di_minus = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    f['adx14'] = dx.rolling(14).mean()

    # Volume
    vol = df['volume'].replace(0, np.nan).ffill().fillna(1)
    f['vol_ratio'] = vol / vol.rolling(20).mean()

    # Time features
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
    f['hour_sin']    = np.sin(2 * np.pi * idx.hour / 24)
    f['hour_cos']    = np.cos(2 * np.pi * idx.hour / 24)
    f['dow']         = idx.dayofweek / 6
    f['london_open'] = ((idx.hour >= 8)  & (idx.hour < 12)).astype(int)
    f['ny_open']     = ((idx.hour >= 13) & (idx.hour < 17)).astype(int)

    # Cross-asset features
    for name in CROSS_ASSET_NAMES:
        col = f'{name}_close'
        if col in df.columns:
            f[f'{name}_ret1d'] = df[col].pct_change(1)
            f[f'{name}_ret5d'] = df[col].pct_change(5)
            if name == 'vix':
                f['vix_level'] = df[col] / 20.0
        else:
            f[f'{name}_ret1d'] = np.nan
            f[f'{name}_ret5d'] = np.nan
            if name == 'vix':
                f['vix_level'] = np.nan

    # Target (only used during training, harmless during inference)
    pip_size = 0.0001
    spread   = spread_pips * pip_size
    next_ret = df['close'].shift(-1) / df['close'] - 1
    f['target'] = (next_ret > spread / df['close']).astype(int)

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


class MLModel:
    """LightGBM classifier for trade confidence scoring."""

    def __init__(self, model_path='config/ml_model.pkl'):
        self.model         = None
        self.model_path    = model_path
        self.feature_names = None

    def predict_confidence(self, df):
        """Return probability (0-1) that the next bar is a tradeable BUY."""
        if self.model is None:
            return 0.5
        try:
            feat = make_features(df)
            if feat.empty:
                return 0.5
            X = feat[self.feature_names].iloc[-1:]
            return float(self.model.predict_proba(X)[0][1])
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 0.5

    def load(self):
        try:
            payload = joblib.load(self.model_path)
            self.model         = payload['model']
            self.feature_names = payload['feature_names']
            trained_at = payload.get('trained_at', 'unknown')
            n_samples  = payload.get('n_samples', '?')
            wf         = payload.get('walk_forward', {})
            logger.info(
                f"Model loaded from {self.model_path} | "
                f"trained {trained_at[:10]} on {n_samples:,} samples | "
                f"walk-forward acc={wf.get('accuracy', 0):.2%} AUC={wf.get('auc', 0):.3f}"
            )
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
