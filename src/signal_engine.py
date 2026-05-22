import numpy as np
import pandas as pd
from src.indicators.indicators import Indicators

class SignalEngine:
    """Mean-reversion dominant signal engine"""

    def __init__(self, ml_confidence_threshold=0.62):
        self.ind = Indicators()
        self.ml_threshold = ml_confidence_threshold

    def calculate_indicators(self, df):
        """Calculate all indicators"""
        df['ema20'] = self.ind.ema(df['close'], 20)
        df['ema50'] = self.ind.ema(df['close'], 50)
        df['rsi'] = self.ind.rsi(df['close'], 14)
        df['atr'] = self.ind.atr(df['high'], df['low'], df['close'], 14)
        df['atr_ma10'] = df['atr'].rolling(10).mean()  # ATR trend — rising = expanding volatility
        df['adx'], df['di_plus'], df['di_minus'] = self.ind.adx(df['high'], df['low'], df['close'], 14)
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = self.ind.bollinger_bands(df['close'], 20, 2.0)
        return df

    def detect_regime(self, adx_value):
        """Determine if market is trending or ranging"""
        return 'TREND' if adx_value >= 25 else 'RANGE'

    def generate_signal(self, df, ml_confidence=0.65, d1_trend=None):
        """
        Signal logic with multi-timeframe trend filter.
        d1_trend: 'UP', 'DOWN', or None (no filter).
        """
        if len(df) < 50:
            return {'direction': 'NONE', 'confidence': 0, 'regime': 'UNKNOWN'}

        close    = df['close'].iloc[-1]
        ema20    = df['ema20'].iloc[-1]
        ema50    = df['ema50'].iloc[-1]
        rsi      = df['rsi'].iloc[-1]
        adx      = df['adx'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        atr      = df['atr'].iloc[-1]
        atr_ma10 = df['atr_ma10'].iloc[-1]

        regime = self.detect_regime(adx)

        # ATR expanding >20% above 10-day avg → trending/volatile → skip mean-reversion
        atr_expanding = pd.notna(atr_ma10) and atr > atr_ma10 * 1.2

        candidate = None

        # --- PRIMARY: MEAN REVERSION ---
        if not atr_expanding:
            if close <= bb_lower * 1.02 and rsi < 42:
                candidate = 'BUY'
            elif close >= bb_upper * 0.98 and rsi > 65:
                candidate = 'SELL'

        # --- SECONDARY: TREND FOLLOWING (strong trend + expanding ATR) ---
        if candidate is None and adx > 30 and atr_expanding:
            if ema20 > ema50 and 40 < rsi < 65:
                candidate = 'BUY'
            elif ema20 < ema50 and 35 < rsi < 60:
                candidate = 'SELL'

        if candidate is None:
            return {'direction': 'NONE', 'confidence': 0, 'regime': regime, 'reason': 'No signal'}

        # Multi-timeframe filter: align with daily trend
        if d1_trend == 'UP' and candidate == 'SELL':
            return {'direction': 'NONE', 'confidence': 0, 'regime': regime, 'reason': 'Against D1 trend'}
        if d1_trend == 'DOWN' and candidate == 'BUY':
            return {'direction': 'NONE', 'confidence': 0, 'regime': regime, 'reason': 'Against D1 trend'}

        if ml_confidence >= self.ml_threshold:
            return {
                'direction': candidate,
                'confidence': ml_confidence,
                'regime': regime,
                'reason': 'Signal confirmed'
            }

        return {'direction': 'NONE', 'confidence': 0, 'regime': regime, 'reason': 'Low ML confidence'}

print("SignalEngine (mean-reversion dominant) loaded")
