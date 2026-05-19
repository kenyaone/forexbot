"""
File-based bridge to MT5 terminal running in Wine.
Python writes a command to forexbot_cmd.txt inside MT5's Files folder.
The ForexBotEA.mq5 Expert Advisor reads it, executes, and writes the result
to forexbot_res.txt. Python polls for the result.

Setup:
  1. ForexBotEA.mq5 must be compiled and running on any chart in MT5
  2. MT5_FILES_PATH in config/.env must point to MT5's MQL5/Files folder
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Forex pair → MT5 symbol name
_SYMBOL_MAP = {
    'EUR/USD': 'EURUSD',
    'GBP/USD': 'GBPUSD',
    'USD/JPY': 'USDJPY',
    'AUD/USD': 'AUDUSD',
    'USD/CHF': 'USDCHF',
}


class MT5FileClient:
    """Send commands to MT5 via shared files in MT5's MQL5/Files folder."""

    def __init__(self, files_path: str):
        self.files_path = Path(files_path)
        self.cmd_file = self.files_path / 'forexbot_cmd.txt'
        self.res_file = self.files_path / 'forexbot_res.txt'

    def ping(self, timeout=5.0):
        """Check if the EA is running by sending INFO command."""
        resp = self._send('INFO', timeout=timeout)
        return resp is not None and resp.startswith('OK')

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_tick(self, pair, timeout=5.0):
        """Return {'bid', 'ask'} or None."""
        sym = _sym(pair)
        resp = self._send(f'TICK|{sym}', timeout=timeout)
        if resp and resp.startswith('OK'):
            d = _parse_kv(resp[3:])
            return {
                'pair': pair,
                'bid': float(d.get('BID', 0)),
                'ask': float(d.get('ASK', 0)),
                'time': datetime.utcnow(),
            }
        return None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    def place_order(self, pair, direction, volume, sl_price, tp_price, timeout=10.0):
        """
        Place a market order.
        Returns {'success', 'ticket', 'price', 'reason'}
        """
        sym = _sym(pair)
        cmd = f'{direction}|{sym}|{volume:.2f}|{sl_price:.5f}|{tp_price:.5f}'
        resp = self._send(cmd, timeout=timeout)
        if resp and resp.startswith('OK'):
            d = _parse_kv(resp[3:])
            return {
                'success': True,
                'ticket': int(d.get('TICKET', 0)),
                'price': float(d.get('PRICE', 0)),
                'reason': 'OK',
            }
        reason = resp or 'No response from EA'
        return {'success': False, 'reason': reason}

    def close_order(self, ticket, timeout=10.0):
        """Close a position by ticket number."""
        resp = self._send(f'CLOSE|{ticket}', timeout=timeout)
        if resp and resp.startswith('OK'):
            return {'success': True}
        return {'success': False, 'reason': resp or 'No response'}

    def get_positions(self, timeout=5.0):
        """Return list of open position dicts."""
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

    def get_account_info(self, timeout=5.0):
        """Return account dict or None."""
        resp = self._send('INFO', timeout=timeout)
        if resp and resp.startswith('OK'):
            return _parse_kv(resp[3:])
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _send(self, command: str, timeout=5.0):
        """Write command file and wait for result file."""
        # Clear any stale result
        if self.res_file.exists():
            self.res_file.unlink()

        # Write command
        try:
            self.cmd_file.write_text(command, encoding='utf-8')
        except Exception as e:
            logger.error(f"MT5FileClient: failed to write command: {e}")
            return None

        # Poll for result
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.res_file.exists():
                try:
                    result = self.res_file.read_text(encoding='utf-8').strip()
                    self.res_file.unlink(missing_ok=True)
                    return result
                except Exception:
                    pass
            time.sleep(0.05)

        logger.warning(f"MT5FileClient: timeout waiting for response to '{command}'")
        self.cmd_file.unlink(missing_ok=True)
        return None


def _sym(pair):
    s = _SYMBOL_MAP.get(pair)
    if s is None:
        raise ValueError(f"Pair not supported: {pair}")
    return s


def _parse_kv(s):
    """Parse 'KEY=VAL|KEY=VAL' into dict."""
    result = {}
    for part in s.split('|'):
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result


print("MT5FileClient loaded")
