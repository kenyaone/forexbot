"""
Synchronous Deriv WebSocket API client.

Uses websocket-client (not asyncio) so it integrates cleanly with our
blocking bot loop. Each public method opens a short-lived connection,
sends one request, waits for the response, and returns.

Authentication:
  - app_id  : register at app.deriv.com → API Token page (or use 1089 for testing)
  - api_token: generated on the same page, enable "Trading" scope for live orders

Instruments (Deriv symbol names):
  EUR/USD → frxEURUSD,  GBP/USD → frxGBPUSD,  USD/JPY → frxUSDJPY
  AUD/USD → frxAUDUSD,  USD/CHF → frxUSDCHF

Order model (Deriv multiplier contracts):
  - BUY  → contract_type = MULTUP
  - SELL → contract_type = MULTDOWN
  - stake      = amount risked in USD (replaces lot_size)
  - stop_loss  = stake (lose the stake at SL price)
  - take_profit= stake × RR  (profit target at TP price)
"""

import json
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime

import websocket

logger = logging.getLogger(__name__)

_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id={app_id}"

_SYMBOL_MAP = {
    'EUR/USD': 'frxEURUSD',
    'GBP/USD': 'frxGBPUSD',
    'USD/JPY': 'frxUSDJPY',
    'AUD/USD': 'frxAUDUSD',
    'USD/CHF': 'frxUSDCHF',
}

_GRANULARITY = {
    'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
    'H1': 3600, 'H4': 14400, 'D1': 86400, '1d': 86400,
}


class DerivClient:
    """Synchronous Deriv API client."""

    def __init__(self, app_id, api_token, demo=True):
        self.app_id = str(app_id)
        self.api_token = api_token
        self._url = _WS_URL.format(app_id=self.app_id)
        self._account_id = None

    # ------------------------------------------------------------------
    # Public: verify credentials on startup
    # ------------------------------------------------------------------
    def connect(self):
        """Authorize and return account info. Raises on failure."""
        resp = self._call({'authorize': self.api_token})
        if 'error' in resp:
            raise ConnectionError(f"Deriv auth failed: {resp['error']['message']}")
        self._account_id = resp['authorize']['loginid']
        balance = resp['authorize']['balance']
        currency = resp['authorize']['currency']
        logger.info(f"Deriv authorized — account {self._account_id}, balance {currency} {balance}")
        return resp['authorize']

    # ------------------------------------------------------------------
    # Public: market data
    # ------------------------------------------------------------------
    def get_candles(self, pair, timeframe='H1', bars=200):
        """Return DataFrame with open/high/low/close/volume columns."""
        symbol = _sym(pair)
        granularity = _GRANULARITY.get(timeframe, 3600)
        end_ts = int(time.time())
        start_ts = end_ts - bars * granularity - 3600  # small buffer

        resp = self._call({
            'ticks_history': symbol,
            'style': 'candles',
            'granularity': granularity,
            'start': start_ts,
            'end': end_ts,
            'count': bars,
            'adjust_start_time': 1,
        })

        if 'error' in resp:
            raise RuntimeError(f"ticks_history error: {resp['error']['message']}")

        candles = resp.get('candles', [])
        if not candles:
            raise RuntimeError(f"No candles returned for {pair}")

        rows = [{
            'time':   pd.Timestamp(c['epoch'], unit='s'),
            'open':   float(c['open']),
            'high':   float(c['high']),
            'low':    float(c['low']),
            'close':  float(c['close']),
            'volume': 0,   # Deriv doesn't provide volume for forex
        } for c in candles]

        return pd.DataFrame(rows)

    def get_tick(self, pair):
        """Return latest bid/ask as {'bid', 'ask', 'pair', 'time'}."""
        symbol = _sym(pair)
        resp = self._call({'ticks': symbol})

        if 'error' in resp:
            raise RuntimeError(f"tick error: {resp['error']['message']}")

        tick = resp.get('tick', {})
        mid = float(tick.get('quote', 0))
        # Deriv ticks give mid price; approximate spread ~0.2 pips for majors
        spread = 0.00010
        return {
            'pair': pair,
            'bid': mid - spread / 2,
            'ask': mid + spread / 2,
            'time': datetime.utcnow(),
        }

    # ------------------------------------------------------------------
    # Public: trading
    # ------------------------------------------------------------------
    def place_order(self, pair, direction, stake_usd, sl_price, tp_price,
                    entry_price, multiplier=100):
        """
        Place a multiplier contract.
        stake_usd  : USD amount at risk (risk_dollars from risk manager)
        sl_price   : stop-loss price level
        tp_price   : take-profit price level
        multiplier : leverage (10, 20, 50, 100, 200, 500)
        Returns {'success', 'contract_id', 'reason'}
        """
        symbol = _sym(pair)
        contract_type = 'MULTUP' if direction == 'BUY' else 'MULTDOWN'

        # Convert price-based SL/TP to USD P&L amounts
        if direction == 'BUY':
            sl_pips = abs(entry_price - sl_price) * 10000
            tp_pips = abs(tp_price - entry_price) * 10000
        else:
            sl_pips = abs(sl_price - entry_price) * 10000
            tp_pips = abs(entry_price - tp_price) * 10000

        rr = tp_pips / sl_pips if sl_pips > 0 else 2.0
        sl_amount = round(stake_usd, 2)
        tp_amount = round(stake_usd * rr, 2)

        # First get a proposal to confirm pricing
        proposal_resp = self._call({
            'proposal': 1,
            'amount': stake_usd,
            'basis': 'stake',
            'contract_type': contract_type,
            'currency': 'USD',
            'symbol': symbol,
            'multiplier': multiplier,
            'limit_order': {
                'stop_loss':   {'order_type': 'stop',  'order_amount': sl_amount},
                'take_profit': {'order_type': 'limit', 'order_amount': tp_amount},
            },
        })

        if 'error' in proposal_resp:
            return {'success': False, 'reason': proposal_resp['error']['message']}

        proposal_id = proposal_resp['proposal']['id']

        # Buy the proposal
        buy_resp = self._call({'buy': proposal_id, 'price': stake_usd})

        if 'error' in buy_resp:
            return {'success': False, 'reason': buy_resp['error']['message']}

        contract_id = buy_resp['buy']['contract_id']
        logger.info(f"Deriv order placed: {contract_id} | {pair} {direction} stake=${stake_usd}")
        return {'success': True, 'contract_id': contract_id, 'reason': 'OK'}

    def close_position(self, contract_id):
        """Sell (close) an open contract."""
        resp = self._call({'sell': contract_id, 'price': 0})
        if 'error' in resp:
            return {'success': False, 'reason': resp['error']['message']}
        return {'success': True, 'sold_for': resp['sell'].get('sold_for')}

    def get_open_contracts(self):
        """Return list of open contract dicts."""
        resp = self._call({'portfolio': 1})
        if 'error' in resp:
            return []
        return resp.get('portfolio', {}).get('contracts', [])

    # ------------------------------------------------------------------
    # Internal: single-shot WebSocket call
    # ------------------------------------------------------------------
    def _call(self, payload, timeout=15):
        """Open a WS connection, send payload, return the first matching response."""
        result = {}
        done = threading.Event()

        def on_message(ws, message):
            result.update(json.loads(message))
            done.set()
            ws.close()

        def on_error(ws, error):
            result['error'] = {'message': str(error)}
            done.set()

        ws = websocket.WebSocketApp(
            self._url,
            on_message=on_message,
            on_error=on_error,
            on_open=lambda ws: ws.send(json.dumps(payload)),
        )

        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()

        if not done.wait(timeout):
            ws.close()
            return {'error': {'message': f'Timeout after {timeout}s'}}

        return result


# ------------------------------------------------------------------
def _sym(pair):
    s = _SYMBOL_MAP.get(pair)
    if s is None:
        raise ValueError(f"Pair not supported by Deriv: {pair}")
    return s


print("DerivClient loaded")
