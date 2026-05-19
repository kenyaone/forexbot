#!/usr/bin/env python3
"""
Backtrader integration with ML model confidence scoring
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.signal_engine import SignalEngine
from src.risk_manager import RiskManager
from src.ml_model import MLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexStrategy(bt.Strategy):
    """Backtrader strategy with ML confidence gating"""

    params = (
        ('risk_per_trade', 0.02),
        ('max_concurrent_trades', 3),
        ('ml_confidence_threshold', 0.62),
        ('cross_data', None),  # dict: {'dxy': df, 'gold': df, 'tnx': df, 'vix': df}
    )

    def __init__(self):
        self.signal_engine = SignalEngine(ml_confidence_threshold=self.params.ml_confidence_threshold)
        self.risk_manager = RiskManager(
            account_equity=self.broker.getvalue(),
            risk_per_trade=self.params.risk_per_trade,
            max_concurrent_trades=self.params.max_concurrent_trades
        )
        self.ml_model = MLModel()
        self.ml_model.load()  # Load pre-trained model

        # Pre-index cross-asset series by date for fast lookup in next()
        self._cross_indexed = {}
        if self.params.cross_data:
            for name, cdf in self.params.cross_data.items():
                self._cross_indexed[name] = cdf.set_index('date')['close']

        self.order_list = []
        self.trade_log = []
        self.wins = 0
        self.losses = 0
        self.signals_fired = 0
        self.signals_blocked_by_ml = 0

    def next(self):
        """Called at every bar"""

        # Build dataframe from recent bars (open+volume needed by ml_model.create_features)
        closes = np.array([self.data.close[i] for i in range(-50, 0)])
        highs = np.array([self.data.high[i] for i in range(-50, 0)])
        lows = np.array([self.data.low[i] for i in range(-50, 0)])
        opens = np.array([self.data.open[i] for i in range(-50, 0)])
        volumes = np.array([self.data.volume[i] for i in range(-50, 0)])

        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        # Append cross-asset close columns aligned to current bar
        if self._cross_indexed:
            current_date = pd.Timestamp(self.data.datetime.date(0))
            n = len(closes)
            for name, series in self._cross_indexed.items():
                recent = series[series.index <= current_date].tail(n)
                col = f'{name}_close'
                if len(recent) >= n:
                    df[col] = recent.values[-n:]
                elif len(recent) > 0:
                    pad = np.full(n - len(recent), recent.values[0])
                    df[col] = np.concatenate([pad, recent.values])

        # Calculate indicators then generate signal
        df = self.signal_engine.calculate_indicators(df)

        # Get ML confidence from model
        try:
            ml_confidence = self.ml_model.predict_confidence(df)
        except Exception:
            ml_confidence = 0.55  # Fallback

        # Get indicator-only signal (ml_confidence=1.0 bypasses internal ML gate
        # so we can count and filter it ourselves below)
        signal = self.signal_engine.generate_signal(df, ml_confidence=1.0)

        if signal['direction'] != 'NONE':
            self.signals_fired += 1

        # ML gate: only trade if confidence >= threshold
        if signal['direction'] == 'NONE' or ml_confidence < self.params.ml_confidence_threshold:
            if signal['direction'] != 'NONE':
                self.signals_blocked_by_ml += 1
            return

        # Check if already in trade
        if self.position.size != 0:
            return

        # Get latest price
        price = self.data.close[0]
        atr = self.signal_engine.ind.atr(df['high'], df['low'], df['close'], 14).iloc[-1]

        # Regime-based SL/TP:
        #   Mean-reversion (RANGE): tight TP at 2× ATR (price only needs to reach midline)
        #   Trend-following (TREND): wider SL at 1.5× ATR + bigger target at 4.5× ATR
        if signal.get('regime') == 'TREND':
            sl_mult, tp_mult = 1.5, 4.5   # RR = 3:1, break-even = 25%
        else:
            sl_mult, tp_mult = 1.0, 2.0   # RR = 2:1, break-even = 33%

        if signal['direction'] == 'BUY':
            sl_price = price - (atr * sl_mult)
            tp_price = price + (atr * tp_mult)
            risk_pips = (price - sl_price) * 10000
        else:
            sl_price = price + (atr * sl_mult)
            tp_price = price - (atr * tp_mult)
            risk_pips = (sl_price - price) * 10000

        # Calculate position size — backtrader units (1 lot = 100,000 units)
        account_value = self.broker.getvalue()
        risk_dollars = account_value * self.params.risk_per_trade
        lot_size = risk_dollars / (risk_pips * 10) if risk_pips > 0 else 0.01
        lot_size = max(min(lot_size, 1.0), 0.01)
        units = lot_size * 100000  # convert lots to backtrader units

        # Place bracket order
        if signal['direction'] == 'BUY':
            self.buy_bracket(
                size=units,
                exectype=bt.Order.Market,
                stopprice=sl_price,
                limitprice=tp_price
            )
        else:
            self.sell_bracket(
                size=units,
                exectype=bt.Order.Market,
                stopprice=sl_price,
                limitprice=tp_price
            )

        logger.debug(f"Trade entered | {signal['direction']} | ML conf: {ml_confidence:.2%}")

    def notify_trade(self, trade):
        """Called when a trade closes"""
        if trade.isclosed:
            pnl = trade.pnl
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1

class BacktestRunner:
    """Run walk-forward backtests with ML filtering"""

    def __init__(self, data_df, start_date, end_date, cross_data=None):
        self.data_df = data_df
        self.start_date = start_date
        self.end_date = end_date
        self.cross_data = cross_data  # dict: {'dxy': df, ...}
        self.results = []

    def run_single_backtest(self, test_start, test_end, params):
        """Run one backtest window"""

        logger.info(f"Backtest: {test_start.date()} → {test_end.date()}")

        # Filter data
        test_data = self.data_df[(self.data_df['date'] >= test_start) & (self.data_df['date'] <= test_end)].copy()

        if len(test_data) == 0:
            logger.warning("No data for test period")
            return None

        test_data = test_data.set_index('date')

        # Create cerebro — large cash simulates leveraged margin account
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=test_data)
        cerebro.adddata(data)
        cerebro.addstrategy(ForexStrategy, cross_data=self.cross_data, **params)
        real_capital = 10000
        cerebro.broker.setcash(1000000)  # leverage simulation; returns reported vs real_capital
        cerebro.broker.setcommission(commission=0.0007)

        # Run
        initial_portfolio = cerebro.broker.getvalue()
        strats = cerebro.run()
        final_portfolio = cerebro.broker.getvalue()

        pnl_dollars = final_portfolio - initial_portfolio
        return_pct = (pnl_dollars / real_capital) * 100

        strat = strats[0]
        total_trades = strat.wins + strat.losses
        win_rate = (strat.wins / total_trades * 100) if total_trades > 0 else 0

        result = {
            'test_period': f"{test_start.date()} → {test_end.date()}",
            'initial_equity': real_capital,
            'final_equity': real_capital + pnl_dollars,
            'pnl_dollars': pnl_dollars,
            'return_pct': return_pct,
            'trades_executed': total_trades,
            'signals_fired': strat.signals_fired,
            'signals_blocked_by_ml': strat.signals_blocked_by_ml,
            'wins': strat.wins,
            'losses': strat.losses,
            'win_rate': win_rate,
        }

        logger.info(f"Result: {return_pct:+.2f}% (${pnl_dollars:+.0f}) | Trades: {total_trades} | WR: {win_rate:.1f}% | ML blocked: {strat.signals_blocked_by_ml}")
        return result

    def run_walk_forward(self, in_sample_months=12, out_sample_months=3, step_months=3):
        """Run walk-forward test"""

        logger.info("Starting walk-forward test WITH ML CONFIDENCE GATING (threshold: 62%)")

        current_date = self.start_date

        while current_date < self.end_date:
            test_start = current_date
            test_end = test_start + timedelta(days=out_sample_months * 30)

            if test_end > self.end_date:
                break

            params = {
                'risk_per_trade': 0.02,
                'max_concurrent_trades': 3,
                'ml_confidence_threshold': 0.62
            }

            result = self.run_single_backtest(test_start, test_end, params)
            if result:
                self.results.append(result)

            current_date += timedelta(days=step_months * 30)

        return self.results

    def print_summary(self):
        """Print results with ML filtering stats"""
        if not self.results:
            logger.warning("No results")
            return

        returns = [r['return_pct'] for r in self.results]
        total_trades = sum(r['trades_executed'] for r in self.results)
        total_signals = sum(r['signals_fired'] for r in self.results)
        total_blocked = sum(r['signals_blocked_by_ml'] for r in self.results)
        total_wins = sum(r['wins'] for r in self.results)
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD TEST WITH ML CONFIDENCE GATING")
        logger.info("="*80)
        logger.info(f"Windows: {len(self.results)}")
        logger.info(f"Signals fired: {total_signals}")
        logger.info(f"Signals blocked by ML (conf < 62%): {total_blocked} ({total_blocked/total_signals*100:.1f}% filtered)" if total_signals > 0 else "Signals fired: 0")
        logger.info(f"Trades executed: {total_trades}")
        logger.info(f"Overall win rate: {overall_win_rate:.1f}%")
        logger.info(f"Avg return: {np.mean(returns):+.2f}%")
        logger.info(f"Std dev: {np.std(returns):.2f}%")
        logger.info(f"Best window: {np.max(returns):+.2f}%")
        logger.info(f"Worst window: {np.min(returns):+.2f}%")
        logger.info(f"Profitable windows: {sum(1 for r in returns if r > 0)} / {len(self.results)}")
        logger.info("="*80 + "\n")

        for i, result in enumerate(self.results):
            logger.info(f"Window {i+1}: {result['test_period']} → {result['return_pct']:+.2f}% (${result['pnl_dollars']:+.0f}) | Trades: {result['trades_executed']} | WR: {result['win_rate']:.1f}% | ML blocked: {result['signals_blocked_by_ml']}")

if __name__ == '__main__':
    from src.data_fetcher import ForexDataFetcher

    end = datetime.now()
    start = end - timedelta(days=365*4)

    data = ForexDataFetcher.fetch_pair('EUR/USD', start, end, interval='1d')
    cross_data = ForexDataFetcher.fetch_cross_assets(start, end, interval='1d')

    if data is not None:
        runner = BacktestRunner(data, start, end, cross_data=cross_data)
        results = runner.run_walk_forward()
        runner.print_summary()

print("Backtest module loaded")
