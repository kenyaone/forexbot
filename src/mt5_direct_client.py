"""
Remote MT5 client via mt5linux/rpyc bridge.
Connects to the rpyc server running on a Windows machine that has MT5 open.

Windows setup:
  pip install MetaTrader5 rpyc plumbum
  python -m mt5linux --host 0.0.0.0 -p 18812

Same interface as MT5FileClient.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def _load_mt5_config():
    """Load login credentials from config/mt5_config.json."""
    cfg_path = Path(__file__).parent.parent / 'config' / 'mt5_config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}

_SYMBOL_MAP = {
    'EUR/USD': 'EURUSD',
    'GBP/USD': 'GBPUSD',
    'USD/JPY': 'USDJPY',
    'AUD/USD': 'AUDUSD',
    'USD/CHF': 'USDCHF',
}

# MT5 order type constants (same as MetaTrader5 package)
ORDER_TYPE_BUY  = 0
ORDER_TYPE_SELL = 1
ORDER_FILLING_FOK = 0


class MT5DirectClient:
    """Connect to MT5 on a remote Windows machine via mt5linux rpyc bridge."""

    def __init__(self, host: str, port: int = 18812):
        self.host = host
        self.port = port
        self._mt5 = None
        cfg = _load_mt5_config()
        self._login    = int(cfg.get('login', 0)) or None
        self._password = cfg.get('password', '') or None
        self._server   = cfg.get('server', '') or None

    def _connect(self):
        from mt5linux import MetaTrader5
        self._mt5 = MetaTrader5(host=self.host, port=self.port)
        kwargs = {}
        if self._login:    kwargs['login']    = self._login
        if self._password: kwargs['password'] = self._password
        if self._server:   kwargs['server']   = self._server
        ok = self._mt5.initialize(**kwargs)
        if not ok:
            raise ConnectionError(f"MT5 initialize failed: {self._mt5.last_error()}")

    def _ensure_connected(self):
        """Reconnect if the rpyc stream has dropped."""
        try:
            if self._mt5 is not None:
                self._mt5.account_info()
                return
        except Exception:
            pass
        for attempt in range(3):
            try:
                self._connect()
                if self._mt5.account_info() is not None:
                    logger.info("MT5 reconnected")
                    return
            except Exception as e:
                logger.warning(f"MT5 reconnect attempt {attempt+1} failed: {e}")
                time.sleep(2)
        raise ConnectionError("MT5 bridge unreachable after 3 attempts")

    def ping(self, timeout=5.0):
        try:
            self._connect()
            info = self._mt5.account_info()
            return info is not None
        except Exception as e:
            logger.warning(f"MT5DirectClient ping failed: {e}")
            return False

    def get_account_info(self):
        try:
            self._ensure_connected()
            info = self._mt5.account_info()
            if info is None:
                return None
            return {
                'LOGIN':   str(info.login),
                'BALANCE': str(round(info.balance, 2)),
                'EQUITY':  str(round(info.equity, 2)),
                'SERVER':  info.server,
            }
        except Exception as e:
            logger.warning(f"MT5 get_account_info failed: {e}")
            return None

    def get_tick(self, pair, timeout=5.0):
        try:
            self._ensure_connected()
            sym = _sym(pair)
            tick = self._mt5.symbol_info_tick(sym)
            if tick is None:
                return None
            return {
                'pair': pair,
                'bid':  tick.bid,
                'ask':  tick.ask,
                'time': datetime.utcnow(),
            }
        except Exception as e:
            logger.warning(f"MT5 get_tick failed: {e}")
            return None

    def place_order(self, pair, direction, volume, sl_price, tp_price, timeout=10.0):
        try:
            self._ensure_connected()
            sym = _sym(pair)
            tick = self._mt5.symbol_info_tick(sym)
            if tick is None:
                return {'success': False, 'reason': 'No tick data'}

            order_type = ORDER_TYPE_BUY if direction == 'BUY' else ORDER_TYPE_SELL
            price = tick.ask if direction == 'BUY' else tick.bid

            request = {
                'action':    self._mt5.TRADE_ACTION_DEAL,
                'symbol':    sym,
                'volume':    float(round(volume, 2)),
                'type':      order_type,
                'price':     price,
                'sl':        float(sl_price),
                'tp':        float(tp_price),
                'deviation': 20,
                'magic':     20250518,
                'comment':   'ForexBot',
                'type_time': self._mt5.ORDER_TIME_GTC,
                'type_filling': ORDER_FILLING_FOK,
            }

            result = self._mt5.order_send(request)
            if result is None:
                return {'success': False, 'reason': str(self._mt5.last_error())}

            if result.retcode == self._mt5.TRADE_RETCODE_DONE:
                return {
                    'success': True,
                    'ticket': result.order,
                    'price':  result.price,
                    'reason': 'OK',
                }
            return {'success': False, 'reason': f'{result.retcode}: {result.comment}'}

        except Exception as e:
            logger.error(f"MT5 place_order failed: {e}")
            return {'success': False, 'reason': str(e)}

    def close_order(self, ticket, timeout=10.0):
        try:
            self._ensure_connected()
            positions = self._mt5.positions_get(ticket=int(ticket))
            if not positions:
                return {'success': False, 'reason': 'Position not found'}

            pos = positions[0]
            sym = pos.symbol
            tick = self._mt5.symbol_info_tick(sym)
            if tick is None:
                return {'success': False, 'reason': 'No tick data'}

            close_type = ORDER_TYPE_SELL if pos.type == 0 else ORDER_TYPE_BUY
            close_price = tick.bid if pos.type == 0 else tick.ask

            request = {
                'action':    self._mt5.TRADE_ACTION_DEAL,
                'symbol':    sym,
                'volume':    pos.volume,
                'type':      close_type,
                'position':  ticket,
                'price':     close_price,
                'deviation': 20,
                'magic':     20250518,
                'comment':   'ForexBot close',
                'type_time': self._mt5.ORDER_TIME_GTC,
                'type_filling': ORDER_FILLING_FOK,
            }

            result = self._mt5.order_send(request)
            if result is None:
                return {'success': False, 'reason': str(self._mt5.last_error())}

            if result.retcode == self._mt5.TRADE_RETCODE_DONE:
                return {'success': True}
            return {'success': False, 'reason': f'{result.retcode}: {result.comment}'}

        except Exception as e:
            logger.error(f"MT5 close_order failed: {e}")
            return {'success': False, 'reason': str(e)}

    def get_positions(self, timeout=5.0):
        try:
            self._ensure_connected()
            positions = self._mt5.positions_get()
            if positions is None:
                return []
            result = []
            for p in positions:
                result.append({
                    'ticket':     p.ticket,
                    'symbol':     p.symbol,
                    'direction':  'BUY' if p.type == 0 else 'SELL',
                    'volume':     p.volume,
                    'open_price': p.price_open,
                    'sl':         p.sl,
                    'tp':         p.tp,
                    'profit':     p.profit,
                })
            return result
        except Exception as e:
            logger.warning(f"MT5 get_positions failed: {e}")
            return []


def _sym(pair):
    s = _SYMBOL_MAP.get(pair)
    if s is None:
        raise ValueError(f"Pair not supported: {pair}")
    return s


print("MT5DirectClient loaded")
