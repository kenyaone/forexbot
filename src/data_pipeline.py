import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

_YFINANCE_TICKERS = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CHF': 'USDCHF=X',
    'USD/CAD': 'USDCAD=X',
    'EUR/GBP': 'EURGBP=X',
    'NZD/USD': 'NZDUSD=X',
}

_YFINANCE_INTERVALS = {
    'H1': '1h', 'H4': '1h', 'H2': '1h', 'D1': '1d', '1h': '1h', '1d': '1d',
}

_CROSS_ASSET_TICKERS = {
    'dxy': 'DX-Y.NYB',
    'gold': 'GC=F',
    'tnx': '^TNX',
    'vix': '^VIX',
}


class DataPipeline:
    """Fetch and normalize OHLCV data — cTrader, Deriv, yfinance, or mock fallback."""

    def __init__(self, ctrader_client=None, deriv_client=None, mt5_client=None, db_connection=None):
        self.ct = ctrader_client
        self.deriv = deriv_client
        self.mt5 = mt5_client
        self.db_connection = db_connection
        self.price_cache = {}
        self._cross_asset_cache = None
        self._cross_asset_fetched_at = None

    def fetch_historical_data(self, pair, timeframe='H1', bars=500):
        if self.ct is not None:
            try:
                df = self.ct.get_candles(pair, timeframe, bars)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"{pair}: cTrader candle fetch failed ({e}), trying yfinance")

        if self.deriv is not None:
            try:
                df = self.deriv.get_candles(pair, timeframe, bars)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"{pair}: Deriv candle fetch failed ({e}), trying yfinance")

        df = self._yfinance_candles(pair, timeframe, bars)
        if df is not None and not df.empty:
            return df

        logger.warning(f"{pair}: all data sources failed — skipping pair this cycle")
        return None

    def _yfinance_candles(self, pair, timeframe='H1', bars=100):
        try:
            import yfinance as yf
            ticker = _YFINANCE_TICKERS.get(pair)
            if not ticker:
                return None
            interval = _YFINANCE_INTERVALS.get(timeframe, '1h')
            days = max(15, bars // 5 + 5) if interval == '1h' else (bars + 30)
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=days)
            df = yf.download(ticker, start=start_dt, end=end_dt, interval=interval,
                             progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume',
            })
            df.reset_index(inplace=True)
            for col in ('Datetime', 'Date', 'index'):
                if col in df.columns:
                    df.rename(columns={col: 'time'}, inplace=True)
                    break
            ts = pd.to_datetime(df['time'])
            df['time'] = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']].dropna()
            # Drop the current incomplete bar — its OHLCV changes every minute
            # and causes alternating ML confidence values across cycles
            if interval == '1h' and len(df) > 1:
                now_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                if pd.to_datetime(df['time'].iloc[-1]) >= pd.Timestamp(now_hour):
                    df = df.iloc[:-1]
            df = df.tail(bars).reset_index(drop=True)
            logger.info(f"{pair}: yfinance fetched {len(df)} bars ({interval})")
            return df
        except Exception as e:
            logger.warning(f"{pair}: yfinance fetch failed ({e})")
            return None

    def get_cross_assets(self, force_refresh=False):
        """Fetch daily DXY, Gold, TNX, VIX data. Cached for 4 hours."""
        now = datetime.utcnow()
        if (not force_refresh and self._cross_asset_cache is not None
                and self._cross_asset_fetched_at is not None
                and (now - self._cross_asset_fetched_at).total_seconds() < 4 * 3600):
            return self._cross_asset_cache

        try:
            import yfinance as yf
            end_dt   = now
            start_dt = end_dt - timedelta(days=15)
            frames = {}
            for name, ticker in _CROSS_ASSET_TICKERS.items():
                try:
                    df = yf.download(ticker, start=start_dt, end=end_dt, interval='1d',
                                     progress=False, auto_adjust=True)
                    if df is None or df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    close_col = 'Close' if 'Close' in df.columns else df.columns[3]
                    s = df[close_col].copy()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    frames[f'{name}_close'] = s
                except Exception:
                    continue
            result = pd.DataFrame(frames) if frames else pd.DataFrame()
            self._cross_asset_cache = result
            self._cross_asset_fetched_at = now
            if not result.empty:
                logger.info(f"Cross-assets fetched: {list(result.columns)}")
            return result
        except Exception as e:
            logger.warning(f"Cross-asset fetch failed: {e}")
            return pd.DataFrame()

    def merge_cross_assets(self, df_h1):
        """Forward-fill daily cross-asset data into an H1 DataFrame (DatetimeIndex required)."""
        cross = self.get_cross_assets()
        if cross.empty:
            return df_h1
        if 'time' in df_h1.columns:
            # Reset-index based df — set time as index temporarily
            df_h1 = df_h1.set_index('time')
            aligned = cross.reindex(df_h1.index, method='ffill')
            merged = df_h1.join(aligned, how='left')
            return merged.reset_index().rename(columns={'index': 'time'})
        else:
            aligned = cross.reindex(df_h1.index, method='ffill')
            return df_h1.join(aligned, how='left')

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
