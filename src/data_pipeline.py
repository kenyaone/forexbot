import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataPipeline:
    """Fetch and normalize OHLCV data — cTrader or Deriv when connected, mock otherwise."""

    def __init__(self, ctrader_client=None, deriv_client=None, mt5_client=None, db_connection=None):
        self.ct = ctrader_client
        self.deriv = deriv_client
        self.mt5 = mt5_client
        self.db_connection = db_connection
        self.price_cache = {}

    def fetch_historical_data(self, pair, timeframe='H1', bars=500):
        if self.ct is not None:
            try:
                df = self.ct.get_candles(pair, timeframe, bars)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"{pair}: cTrader candle fetch failed ({e}), trying Deriv")

        if self.deriv is not None:
            try:
                df = self.deriv.get_candles(pair, timeframe, bars)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"{pair}: Deriv candle fetch failed ({e}), using mock")

        return self._mock_candles(bars)

    def normalise_data(self, df):
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def get_live_tick(self, pair):
        if self.mt5 is not None:
            try:
                tick = self.mt5.get_tick(pair)
                if tick is not None:
                    return tick
            except Exception as e:
                logger.warning(f"{pair}: MT5 tick failed ({e})")

        if self.ct is not None:
            try:
                return self.ct.get_tick(pair)
            except Exception as e:
                logger.warning(f"{pair}: cTrader tick failed ({e})")

        if self.deriv is not None:
            try:
                return self.deriv.get_tick(pair)
            except Exception as e:
                logger.warning(f"{pair}: Deriv tick failed ({e}), using mock")

        return self._mock_tick(pair)

    # ------------------------------------------------------------------
    def _mock_candles(self, bars):
        dates = pd.date_range(end=datetime.utcnow(), periods=bars, freq='1h')
        close = pd.Series(np.cumsum(np.random.randn(bars) * 0.001) + 1.085)
        return pd.DataFrame({
            'time':   dates,
            'open':   close.shift(1).fillna(close.iloc[0]),
            'high':   close + np.abs(np.random.randn(bars)) * 0.0005,
            'low':    close - np.abs(np.random.randn(bars)) * 0.0005,
            'close':  close,
            'volume': np.random.randint(1000, 10000, bars),
        })

    def _mock_tick(self, pair):
        return {'pair': pair, 'bid': 1.0850, 'ask': 1.0852, 'time': datetime.utcnow()}

    def store_candles(self, pair, df, timeframe='H1'):
        self.price_cache[f"{pair}_{timeframe}"] = df

    def get_candles(self, pair, timeframe='H1', bars=100):
        key = f"{pair}_{timeframe}"
        if key in self.price_cache:
            return self.price_cache[key].tail(bars)
        df = self.fetch_historical_data(pair, timeframe, bars)
        self.store_candles(pair, df, timeframe)
        return df

print("DataPipeline (cTrader/Deriv) loaded")
