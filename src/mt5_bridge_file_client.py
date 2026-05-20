"""
File-bridge MT5 client that works via the rpyc bridge on a remote Windows machine.
Uses ForexBotEA.mq5 running inside MT5 for order execution, bypassing the
MetaTrader5 Python package version dependency entirely.
"""

import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_SYMBOL_MAP = {
    'EUR/USD': 'EURUSD',
    'GBP/USD': 'GBPUSD',
    'USD/JPY': 'USDJPY',
    'AUD/USD': 'AUDUSD',
    'USD/CHF': 'USDCHF',
}

FILES_DIR = r'C:\Users\Admin\AppData\Roaming\MetaQuotes\Terminal\Common\Files'


class MT5BridgeFileClient:
    """Send commands to ForexBotEA via shared files, using rpyc for file I/O."""

    def __init__(self, host: str, port: int = 18812):
        self.host = host
        self.port = port
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            import rpyc
            self._conn = rpyc.classic.connect(self.host, self.port)
        return self._conn

    def ping(self, timeout=5.0):
        try:
            resp = self._send('INFO', timeout=timeout)
            return resp is not None and resp.startswith('OK')
        except Exception as e:
            logger.warning(f"MT5BridgeFileClient ping failed: {e}")
            return False

    def get_account_info(self, timeout=5.0):
        resp = self._send('INFO', timeout=timeout)
        if resp and resp.startswith('OK'):
            return _parse_kv(resp[3:])
        return None

    def get_tick(self, pair, timeout=5.0):
        sym = _sym(pair)
        resp = self._send(f'TICK|{sym}', timeout=timeout)
        if resp and resp.startswith('OK'):
            d = _parse_kv(resp[3:])
            return {
                'pair': pair,
                'bid':  float(d.get('BID', 0)),
                'ask':  float(d.get('ASK', 0)),
                'time': datetime.utcnow(),
            }
        return None

    def place_order(self, pair, direction, volume, sl_price, tp_price, timeout=10.0):
        sym = _sym(pair)
        cmd = f'{direction}|{sym}|{volume:.2f}|{sl_price:.5f}|{tp_price:.5f}'
        resp = self._send(cmd, timeout=timeout)
        if resp and resp.startswith('OK'):
            d = _parse_kv(resp[3:])
            return {
                'success': True,
                'ticket':  int(d.get('TICKET', 0)),
                'price':   float(d.get('PRICE', 0)),
                'reason':  'OK',
            }
        return {'success': False, 'reason': resp or 'No response from EA'}

    def close_order(self, ticket, timeout=10.0):
        resp = self._send(f'CLOSE|{ticket}', timeout=timeout)
        if resp and resp.startswith('OK'):
            return {'success': True}
        return {'success': False, 'reason': resp or 'No response'}

    def modify_order(self, ticket, sl_price, tp_price, timeout=10.0):
        cmd = f'MODIFY|{ticket}|{sl_price:.5f}|{tp_price:.5f}'
        resp = self._send(cmd, timeout=timeout)
        if resp and resp.startswith('OK'):
            return {'success': True}
        return {'success': False, 'reason': resp or 'No response'}

    def get_positions(self, timeout=5.0):
        resp = self._send('POSITIONS', timeout=timeout)
        if not resp or not resp.startswith('OK'):
            return []
        parts = resp[3:].split('|')
        positions = []
        for p in parts:
            if not p or p == 'NONE':
                continue
            fields = p.split(',')
            if len(fields) >= 8:
                positions.append({
                    'ticket':     int(fields[0]),
                    'symbol':     fields[1],
                    'direction':  fields[2],
                    'volume':     float(fields[3]),
                    'open_price': float(fields[4]),
                    'sl':         float(fields[5]),
                    'tp':         float(fields[6]),
                    'profit':     float(fields[7]),
                })
        return positions

    # ------------------------------------------------------------------
    def _send(self, command: str, timeout=5.0):
        cmd_file = FILES_DIR + r'\forexbot_cmd.txt'
        res_file = FILES_DIR + r'\forexbot_res.txt'
        try:
            conn = self._get_conn()
            # MQL5 FILE_TXT reads/writes UTF-16 LE with BOM
            conn.execute(f"""
import os
if os.path.exists(r'{res_file}'):
    os.remove(r'{res_file}')
with open(r'{cmd_file}', 'wb') as _f:
    _f.write({repr(command.encode('utf-16'))})
""")
            deadline = time.time() + timeout
            while time.time() < deadline:
                exists = conn.eval(f"os.path.exists(r'{res_file}')")
                if exists:
                    raw = bytes(conn.eval(f"open(r'{res_file}','rb').read()"))
                    result = raw.decode('utf-16', errors='replace').strip()
                    conn.eval(f"os.remove(r'{res_file}')")
                    return result
                time.sleep(0.1)
            logger.warning(f"MT5BridgeFileClient: timeout waiting for response to '{command}'")
            return None
        except Exception as e:
            logger.error(f"MT5BridgeFileClient send error: {e}")
            self._conn = None
            return None


def _sym(pair):
    s = _SYMBOL_MAP.get(pair)
    if s is None:
        raise ValueError(f"Pair not supported: {pair}")
    return s


def _parse_kv(s):
    result = {}
    for part in s.split('|'):
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result
