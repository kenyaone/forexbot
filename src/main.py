#!/usr/bin/env python3
"""
Forex Trading Bot — Main Entry Point
Orchestrates data pipeline, signal generation, risk management, and execution
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

from src.data_pipeline import DataPipeline
from src.signal_engine import SignalEngine
from src.risk_manager import RiskManager, RiskState
from src.order_executor import OrderExecutor
from src.ml_model import MLModel
from src.data_fetcher import ForexDataFetcher
from src.ctrader_client import CTraderClient
from src.deriv_client import DerivClient
from src.mt5_client import MT5FileClient
from src.mt5_direct_client import MT5DirectClient
from src.mt5_bridge_file_client import MT5BridgeFileClient
from src.alerting import alert_trade_opened, alert_trade_closed, alert_daily_summary, alert_error, _send_telegram

# Load environment variables
load_dotenv('config/.env')

def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def _init_ctrader():
    """Build a CTraderClient from env vars. Returns client or None."""
    client_id     = os.getenv('CTRADER_CLIENT_ID', '')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET', '')
    access_token  = os.getenv('CTRADER_ACCESS_TOKEN', '')
    account_id    = os.getenv('CTRADER_ACCOUNT_ID', '')
    demo          = os.getenv('CTRADER_ENVIRONMENT', 'demo').lower() == 'demo'

    if not all([client_id, client_secret, access_token, account_id]):
        return None
    if 'your_' in client_id:
        return None

    try:
        ct = CTraderClient(client_id, client_secret, access_token, account_id, demo=demo)
        ct.connect(timeout=20)
        return ct
    except Exception as e:
        logging.getLogger(__name__).warning(f"cTrader init failed: {e}")
        return None

def _init_mt5():
    """Try MT5BridgeFileClient (ForexBotEA via rpyc) then local Wine fallback."""
    host = os.getenv('MT5_HOST', '')
    if host:
        port = int(os.getenv('MT5_PORT', '18812'))
        # File bridge via rpyc — works regardless of MT5 Python package version
        try:
            client = MT5BridgeFileClient(host=host, port=port)
            if client.ping(timeout=6.0):
                logger.info(f"MT5 file bridge connected at {host}:{port}")
                return client
            logger.warning(f"MT5BridgeFileClient: EA not responding — is ForexBotEA running on a chart?")
        except Exception as e:
            logging.getLogger(__name__).warning(f"MT5BridgeFileClient init failed: {e}")
        # Fallback: direct Python API (requires matching package/terminal versions)
        try:
            client = MT5DirectClient(host=host, port=port)
            if client.ping():
                logger.info(f"MT5 direct API connected at {host}:{port}")
                return client
        except Exception as e:
            logging.getLogger(__name__).warning(f"MT5DirectClient init failed: {e}")

    # Local Wine file bridge fallback
    files_path = os.getenv('MT5_FILES_PATH', '')
    if not files_path or not os.path.isdir(files_path):
        return None
    try:
        client = MT5FileClient(files_path)
        if client.ping(timeout=3.0):
            return client
        logger.warning("MT5 EA not responding — make sure ForexBotEA is running on a chart")
        return None
    except Exception as e:
        logging.getLogger(__name__).warning(f"MT5FileClient init failed: {e}")
        return None

def _init_deriv():
    """Build a DerivClient from env vars. Returns client or None."""
    app_id    = os.getenv('DERIV_APP_ID', '')
    api_token = os.getenv('DERIV_API_TOKEN', '')

    if not all([app_id, api_token]):
        return None
    if 'your_' in app_id or 'your_' in api_token:
        return None

    try:
        d = DerivClient(app_id, api_token)
        d.connect()
        return d
    except Exception as e:
        logging.getLogger(__name__).warning(f"Deriv init failed: {e}")
        return None

# Setup logging
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ForexTradingBot:
    def __init__(self, account_equity=10000, risk_per_trade=0.02):
        logger.info(f"Initializing bot with equity: ${account_equity}")

        # Initialize broker — MT5 first, then cTrader, then Deriv, then mock
        mt5   = _init_mt5()
        ct    = _init_ctrader()  if mt5  is None else None
        deriv = _init_deriv()    if ct   is None and mt5 is None else None

        if mt5:
            logger.info("Broker: MT5 (IC Markets) connected via file bridge")
            info = mt5.get_account_info()
            if info and info.get('BALANCE'):
                account_equity = float(info['BALANCE'])
                logger.info(f"Real account equity from MT5: ${account_equity:.2f}")
        elif ct:
            logger.info("Broker: cTrader connected")
        elif deriv:
            logger.info("Broker: Deriv connected")
        else:
            logger.warning("No broker configured — running in mock mode")

        self.mt5 = mt5
        self.account_equity = account_equity
        self.current_equity = account_equity

        # Initialize modules
        self.data_pipeline = DataPipeline(ctrader_client=ct, deriv_client=deriv, mt5_client=mt5)
        self.signal_engine = SignalEngine(ml_confidence_threshold=0.52)
        self.risk_manager = RiskManager(
            account_equity=account_equity,
            risk_per_trade=risk_per_trade,
            max_daily_loss=0.05,
            max_concurrent_trades=3
        )
        self.order_executor = OrderExecutor(
            ctrader_client=ct,
            deriv_client=deriv,
            mt5_client=mt5,
            risk_manager=self.risk_manager
        )
        
        # Initialize ML model
        self.ml_model = MLModel(model_path='config/ml_model.pkl')
        if not self.ml_model.load():
            logger.warning("ML model not found — using fallback confidence=0.55")
        
        self.trading_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF']
        self.running = False
        self.daily_pnl = 0
        self._cross_data = {}
        self._cycle_count = 0
        self._scan_results = {}  # pair → (direction, confidence)
    
    def is_trading_hours(self):
        """Check if we're in trading session (07:00-17:00 UTC)"""
        now = _utcnow()
        hour = now.hour
        return 7 <= hour < 17
    
    def process_pair(self, pair):
        """Generate signal and execute trade for a pair"""
        
        # Fetch latest data
        df = self.data_pipeline.fetch_historical_data(pair, timeframe='H1', bars=100)
        df = self.data_pipeline.normalise_data(df)
        
        # Calculate indicators
        df = self.signal_engine.calculate_indicators(df)
        
        # Merge cross-asset columns so ML model can use them
        for name, cdf in self._cross_data.items():
            col = f'{name}_close'
            if not cdf.empty:
                recent_val = cdf['close'].iloc[-1]
                df[col] = recent_val  # scalar broadcast — latest value on all rows

        # Get ML confidence from model
        try:
            ml_confidence = self.ml_model.predict_confidence(df)
        except Exception:
            logger.warning(f"{pair}: ML prediction failed, using fallback 0.55")
            ml_confidence = 0.55
        
        # Generate signal with ML confidence
        signal = self.signal_engine.generate_signal(df, ml_confidence=ml_confidence)
        
        logger.info(f"{pair}: {signal['direction']} (ML conf: {ml_confidence:.2%}, regime: {signal.get('regime')})")
        self._scan_results[pair] = (signal['direction'], ml_confidence)

        if signal['direction'] == 'NONE':
            return
        
        # Get latest price
        tick = self.data_pipeline.get_live_tick(pair)
        entry_price = tick['ask'] if signal['direction'] == 'BUY' else tick['bid']

        # Pip size per pair (JPY pairs use 0.01, all others 0.0001)
        pip = 0.01 if 'JPY' in pair else 0.0001
        sl_pips = 30   # fixed 30-pip stop
        tp_pips = 60   # fixed 60-pip target (1:2 R:R)
        atr_pips = sl_pips  # for position sizing

        if signal['direction'] == 'BUY':
            sl_price = round(entry_price - sl_pips * pip, 5)
            tp_price = round(entry_price + tp_pips * pip, 5)
        else:
            sl_price = round(entry_price + sl_pips * pip, 5)
            tp_price = round(entry_price - tp_pips * pip, 5)
        
        # Calculate position size
        lot_size = self.risk_manager.calculate_position_size(entry_price, sl_price, atr_pips)
        
        # Place order
        result = self.order_executor.place_order(
            pair=pair,
            direction=signal['direction'],
            lot_size=lot_size,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            signal_confidence=signal['confidence']
        )
        
        if result['success']:
            logger.info(f"Order placed: {result['order_id']} | {pair} {signal['direction']} {lot_size} lots | ML conf: {ml_confidence:.2%}")
            alert_trade_opened(
                pair=pair, direction=signal['direction'], volume=lot_size,
                entry=entry_price, sl=sl_price, tp=tp_price,
                confidence=ml_confidence * 100, ticket=result.get('order_id', 0)
            )
        else:
            logger.warning(f"Order rejected: {result['reason']}")
    
    def check_exits(self):
        """Monitor open trades and close if TP/SL/time exit"""
        now = _utcnow()
        
        for order_id, order in list(self.order_executor.open_orders.items()):
            tick = self.data_pipeline.get_live_tick(order['pair'])
            current_price = (tick['bid'] + tick['ask']) / 2
            
            exit_check = self.order_executor.check_exit_conditions(order_id, current_price, now)
            
            if exit_check['should_close']:
                result = self.order_executor.close_order(
                    order_id,
                    close_price=exit_check['close_price'],
                    reason=exit_check['reason']
                )
                if result['success']:
                    logger.info(f"Order closed: {order_id} | Reason: {exit_check['reason']} | P&L: ${result['pnl_usd']:.2f}")
                    self.daily_pnl += result['pnl_usd']
                    alert_trade_closed(
                        pair=order['pair'], direction=order['direction'],
                        volume=order.get('lot_size', 0), entry=order.get('entry_price', 0),
                        close_price=exit_check['close_price'], profit=result['pnl_usd'],
                        ticket=order_id
                    )
    
    def update_risk_state(self):
        """Check daily loss and update risk state"""
        state = self.risk_manager.check_risk_state(self.current_equity, self.daily_pnl)
        
        if state != RiskState.NORMAL:
            logger.warning(f"Risk state changed to: {state.value} (Daily P&L: ${self.daily_pnl:.2f})")
        
        if state == RiskState.EMERGENCY_STOP:
            logger.critical("EMERGENCY STOP TRIGGERED — closing all positions")
            self.running = False
    
    def run_one_cycle(self):
        """Execute one trading cycle"""
        
        if not self.is_trading_hours():
            logger.info("Outside trading hours — skipping cycle")
            return
        
        try:
            # Refresh equity from MT5
            if self.mt5:
                info = self.mt5.get_account_info()
                if info and info.get('EQUITY'):
                    self.current_equity = float(info['EQUITY'])
                    self.risk_manager.account_equity = self.current_equity

            # Sync OANDA open trades (removes stale local orders)
            self.order_executor.sync_open_trades()

            # Check exits first
            self.check_exits()

            # Refresh cross-asset data once per cycle (shared across all pairs)
            end = _utcnow()
            start = end - timedelta(days=60)
            self._cross_data = ForexDataFetcher.fetch_cross_assets(start, end, interval='1d')

            # Generate signals for each pair
            for pair in self.trading_pairs:
                self.process_pair(pair)
            
            # Update risk state
            self.update_risk_state()

            self._cycle_count += 1
            logger.info(f"Cycle complete | Equity: ${self.current_equity:.2f} | Daily P&L: ${self.daily_pnl:.2f} | State: {self.risk_manager.state.value}")

            # Send scan heartbeat every 4 cycles (4 hours)
            if self._cycle_count % 4 == 0 and self._scan_results:
                lines = []
                for p, (direction, conf) in self._scan_results.items():
                    bar = '🟢' if direction != 'NONE' else ('🟡' if conf >= 0.48 else '⚪')
                    lines.append(f"{bar} {p}: {conf:.0%} {direction if direction != 'NONE' else ''}")
                _send_telegram(
                    f"<b>📊 ForexBot Scan — {_utcnow().strftime('%H:%M UTC')}</b>\n"
                    + "\n".join(lines)
                    + f"\nEquity: ${self.current_equity:,.2f} | P&L: ${self.daily_pnl:+.2f}"
                )
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
    
    def run(self, cycle_interval=3600):
        """Main bot loop"""
        self.running = True
        logger.info("Bot started with ML model")

        _send_telegram(
            f"<b>🤖 ForexBot Started</b>\n"
            f"Balance: <b>${self.current_equity:,.2f}</b>\n"
            f"Pairs: {', '.join(self.trading_pairs)}\n"
            f"Risk/trade: {int(self.risk_manager.risk_per_trade * 100)}% | "
            f"Threshold: 55% confidence\n"
            f"{_utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        last_summary_day = None

        try:
            while self.running:
                self.run_one_cycle()

                # Send daily summary once per day at end of trading session
                now = _utcnow()
                if now.hour >= 17 and last_summary_day != now.date():
                    last_summary_day = now.date()
                    trades_today = len([o for o in self.order_executor.open_orders.values()])
                    alert_daily_summary(
                        equity=self.current_equity,
                        balance=self.current_equity,
                        daily_pnl=self.daily_pnl,
                        trades=trades_today,
                    )

                time.sleep(cycle_interval)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            alert_error(f"Fatal error — bot stopped: {str(e)}")
        finally:
            logger.info("Bot shutdown complete")

if __name__ == '__main__':
    bot = ForexTradingBot(account_equity=10000, risk_per_trade=0.02)
    bot.run(cycle_interval=3600)  # Run every hour for H1 bars
