#!/usr/bin/env python3
"""
Fetch real historical forex data from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexDataFetcher:
    """Download real forex OHLCV data"""
    
    # Yahoo Finance forex ticker format: EURUSD=X, GBPUSD=X, etc.
    FOREX_TICKERS = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CHF': 'USDCHF=X'
    }
    
    @staticmethod
    def fetch_pair(pair, start_date, end_date, interval='1d'):
        """
        Fetch OHLCV data for a forex pair
        pair: 'EUR/USD' etc
        interval: '1d' (daily), '1h' (hourly), '5m' (5-minute)
        """
        
        if pair not in ForexDataFetcher.FOREX_TICKERS:
            logger.error(f"Pair {pair} not supported")
            return None
        
        ticker = ForexDataFetcher.FOREX_TICKERS[pair]
        logger.info(f"Fetching {pair} ({ticker}) from {start_date.date()} to {end_date.date()}")
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            
            if df.empty:
                logger.error(f"No data returned for {pair}")
                return None
            
            # Flatten MultiIndex columns returned by newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Reset index to make date a column
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'date', 'Datetime': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            # Keep only OHLCV
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Fetched {len(df)} bars for {pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching {pair}: {str(e)}")
            return None
    
    CROSS_ASSET_TICKERS = {
        'dxy':  'DX-Y.NYB',
        'gold': 'GC=F',
        'tnx':  '^TNX',
        'vix':  '^VIX',
    }

    @staticmethod
    def fetch_cross_assets(start_date, end_date, interval='1d'):
        """Fetch DXY, Gold, US 10Y yield (TNX), VIX — returns dict of {name: df}"""
        result = {}
        for name, ticker in ForexDataFetcher.CROSS_ASSET_TICKERS.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date,
                                 interval=interval, progress=False)
                if df.empty:
                    logger.warning(f"No data for {name} ({ticker})")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Close']].rename(columns={'Close': 'close'})
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'date', 'Datetime': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                result[name] = df[['date', 'close']]
                logger.info(f"Fetched {name} ({ticker}): {len(df)} bars")
            except Exception as e:
                logger.warning(f"Could not fetch {name} ({ticker}): {e}")
        return result

    @staticmethod
    def fetch_all_pairs(start_date, end_date, interval='1d'):
        """Fetch all 5 pairs at once"""
        pairs_data = {}
        
        for pair in ForexDataFetcher.FOREX_TICKERS.keys():
            df = ForexDataFetcher.fetch_pair(pair, start_date, end_date, interval)
            if df is not None:
                pairs_data[pair] = df
        
        logger.info(f"Fetched {len(pairs_data)} pairs")
        return pairs_data

if __name__ == '__main__':
    # Example: fetch EUR/USD for last 4 years
    end = datetime.now()
    start = end - timedelta(days=365*4)
    
    fetcher = ForexDataFetcher()
    data = fetcher.fetch_pair('EUR/USD', start, end, interval='1d')
    
    if data is not None:
        print(data.head())
        print(f"\nShape: {data.shape}")

print("ForexDataFetcher module loaded")
