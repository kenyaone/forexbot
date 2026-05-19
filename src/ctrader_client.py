"""
Synchronous wrapper around the Twisted-based cTrader Open API.
Runs the Twisted reactor in a background daemon thread; exposes blocking
methods that the main bot loop can call normally.

Auth flow:
  1. App auth  (clientId + clientSecret)
  2. Account auth (accessToken from OAuth)
  3. Symbol-list fetch (name → symbolId cache)
"""

import threading
import time
import logging
import pandas as pd
from datetime import datetime, timezone

from twisted.internet import reactor

from ctrader_open_api import Client, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq, ProtoOAApplicationAuthRes,
    ProtoOAAccountAuthReq, ProtoOAAccountAuthRes,
    ProtoOASymbolsListReq, ProtoOASymbolsListRes,
    ProtoOAGetTrendbarsReq, ProtoOAGetTrendbarsRes,
    ProtoOASubscribeSpotsReq, ProtoOASubscribeSpotsRes,
    ProtoOAUnsubscribeSpotsReq,
    ProtoOANewOrderReq, ProtoOAClosePositionReq,
    ProtoOAReconcileReq, ProtoOAReconcileRes,
    ProtoOAErrorRes,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOATrendbarPeriod, ProtoOAOrderType, ProtoOATradeSide,
)

logger = logging.getLogger(__name__)

_PERIOD_MAP = {
    'M1': ProtoOATrendbarPeriod.Value('M1'),
    'M5': ProtoOATrendbarPeriod.Value('M5'),
    'M15': ProtoOATrendbarPeriod.Value('M15'),
    'M30': ProtoOATrendbarPeriod.Value('M30'),
    'H1': ProtoOATrendbarPeriod.Value('H1'),
    'H4': ProtoOATrendbarPeriod.Value('H4'),
    'D1': ProtoOATrendbarPeriod.Value('D1'),
    '1d': ProtoOATrendbarPeriod.Value('D1'),
}

# cTrader volume units: 1 standard lot = 100 (cTrader's internal unit where 1 = 0.01 lots)
_VOLUME_PER_LOT = 100


class CTraderClient:
    """Thread-safe synchronous cTrader Open API client."""

    def __init__(self, client_id, client_secret, access_token, account_id, demo=True):
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token = access_token
        self._account_id = int(account_id)
        self._host = EndPoints.PROTOBUF_DEMO_HOST if demo else EndPoints.PROTOBUF_LIVE_HOST

        self._client = None
        self._lock = threading.Lock()
        self._msg_id = 0

        # Pending synchronous calls: clientMsgId → [Event, result]
        self._pending = {}

        # Events for auth stages
        self._app_authed = threading.Event()
        self._acct_authed = threading.Event()
        self._symbols_ready = threading.Event()

        # Cache: 'EUR/USD' → symbolId (int)
        self._symbol_ids = {}

        # Latest spot prices: symbolId → {'bid': float, 'ask': float}
        self._spot_cache = {}

    # ------------------------------------------------------------------
    # Public: connect (call once before using the client)
    # ------------------------------------------------------------------
    def connect(self, timeout=20):
        """Start Twisted reactor in a daemon thread and authenticate."""
        self._client = Client(self._host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._client.startService()

        t = threading.Thread(
            target=reactor.run,
            kwargs={'installSignalHandlers': False},
            daemon=True,
            name='ctrader-reactor',
        )
        t.start()

        if not self._app_authed.wait(timeout):
            raise TimeoutError("cTrader app auth timed out")
        if not self._acct_authed.wait(timeout):
            raise TimeoutError("cTrader account auth timed out")
        if not self._symbols_ready.wait(timeout):
            raise TimeoutError("cTrader symbol list timed out")

        logger.info(f"cTrader connected — {len(self._symbol_ids)} symbols cached")

    # ------------------------------------------------------------------
    # Public: data methods
    # ------------------------------------------------------------------
    def get_candles(self, pair, timeframe='H1', bars=200):
        """Return DataFrame with open/high/low/close/volume columns."""
        symbol_id = self._resolve_symbol(pair)
        period = _PERIOD_MAP.get(timeframe, _PERIOD_MAP['H1'])

        # Request up to 'bars' candles ending now
        to_ts = int(time.time() * 1000)
        from_ts = to_ts - bars * _bar_ms(timeframe) - 1

        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = from_ts
        req.toTimestamp = to_ts
        req.count = bars

        resp = self._send_sync(req, timeout=15)
        if resp is None:
            raise RuntimeError(f"get_candles timed out for {pair}")

        rows = []
        for b in resp.trendbar:
            ts = pd.Timestamp(b.utcTimestampInMinutes * 60, unit='s', tz='UTC').tz_localize(None)
            # Prices in cTrader are in 1e-5 (pips * 10 for 5-digit brokers)
            divisor = 10 ** resp.digits
            close = b.close / divisor
            low   = b.low / divisor
            # open and high are stored as deltas from close in some versions
            delta_open  = getattr(b, 'deltaOpen', 0)
            delta_high  = getattr(b, 'deltaHigh', 0)
            open_p  = (b.close + delta_open) / divisor if delta_open else close
            high_p  = (b.close + delta_high) / divisor if delta_high else close
            rows.append({
                'time': ts, 'open': open_p, 'high': high_p,
                'low': low, 'close': close, 'volume': b.volume,
            })

        return pd.DataFrame(rows)

    def get_tick(self, pair):
        """Return {'bid': float, 'ask': float} from the spot cache."""
        symbol_id = self._resolve_symbol(pair)

        # Subscribe if not cached
        if symbol_id not in self._spot_cache:
            self._subscribe_spots([symbol_id])
            time.sleep(1.0)  # allow spot event to arrive

        spot = self._spot_cache.get(symbol_id, {})
        return {
            'pair': pair,
            'bid': spot.get('bid', 0.0),
            'ask': spot.get('ask', 0.0),
            'time': datetime.utcnow(),
        }

    def place_order(self, pair, direction, lot_size, sl_price, tp_price):
        """
        Place a market order with SL and TP.
        Returns {'success': bool, 'position_id': int or None, 'reason': str}
        """
        symbol_id = self._resolve_symbol(pair)
        side = ProtoOATradeSide.Value('BUY' if direction == 'BUY' else 'SELL')
        volume = int(lot_size * _VOLUME_PER_LOT)

        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId = symbol_id
        req.orderType = ProtoOAOrderType.Value('MARKET')
        req.tradeSide = side
        req.volume = volume
        req.stopLoss = sl_price
        req.takeProfit = tp_price

        resp = self._send_sync(req, timeout=15)
        if resp is None:
            return {'success': False, 'reason': 'Order request timed out'}

        # On success, resp is ProtoOAExecutionEvent or similar — check for error
        if hasattr(resp, 'errorCode'):
            return {'success': False, 'reason': resp.description or resp.errorCode}

        pos_id = None
        if hasattr(resp, 'position'):
            pos_id = resp.position.positionId

        return {'success': True, 'position_id': pos_id, 'reason': 'OK'}

    def close_position(self, position_id, volume=None):
        """Close a position (partially or fully)."""
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self._account_id
        req.positionId = int(position_id)
        # volume omitted → full close

        resp = self._send_sync(req, timeout=15)
        if resp is None:
            return {'success': False, 'reason': 'Close request timed out'}
        if hasattr(resp, 'errorCode'):
            return {'success': False, 'reason': resp.description or resp.errorCode}
        return {'success': True}

    def get_open_positions(self):
        """Return list of open position dicts from the broker."""
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self._account_id
        resp = self._send_sync(req, timeout=15)
        if resp is None:
            return []
        return list(resp.position) if hasattr(resp, 'position') else []

    # ------------------------------------------------------------------
    # Internal: Twisted callbacks (run on reactor thread)
    # ------------------------------------------------------------------
    def _on_connected(self, client):
        req = ProtoOAApplicationAuthReq()
        req.clientId = self._client_id
        req.clientSecret = self._client_secret
        d = client.send(req)
        d.addErrback(self._on_error)

    def _on_disconnected(self, client, reason):
        logger.warning(f"cTrader disconnected: {reason}")

    def _on_message(self, client, message):
        msg_type = message.__class__.__name__

        if msg_type == 'ProtoOAApplicationAuthRes':
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = self._account_id
            req.accessToken = self._access_token
            d = client.send(req)
            d.addErrback(self._on_error)
            self._app_authed.set()

        elif msg_type == 'ProtoOAAccountAuthRes':
            self._acct_authed.set()
            # Fetch symbol list
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = self._account_id
            d = client.send(req)
            d.addErrback(self._on_error)

        elif msg_type == 'ProtoOASymbolsListRes':
            for sym in message.symbol:
                name = sym.symbolName.replace('/', '_')   # store as 'EUR_USD'
                self._symbol_ids[name] = sym.symbolId
            self._symbols_ready.set()

        elif msg_type == 'ProtoOASpotEvent':
            sid = message.symbolId
            spot = self._spot_cache.setdefault(sid, {})
            if message.HasField('bid'):
                spot['bid'] = message.bid / 100000
            if message.HasField('ask'):
                spot['ask'] = message.ask / 100000

        else:
            # Try to resolve a pending sync call
            client_msg_id = getattr(message, 'clientMsgId', None)
            if client_msg_id and client_msg_id in self._pending:
                ev, holder = self._pending.pop(client_msg_id)
                holder.append(message)
                ev.set()

    def _on_error(self, failure):
        logger.error(f"cTrader Twisted error: {failure.getErrorMessage()}")

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------
    def _send_sync(self, req, timeout=10):
        """Send a request and block until the response arrives."""
        with self._lock:
            self._msg_id += 1
            mid = str(self._msg_id)

        ev = threading.Event()
        holder = []
        self._pending[mid] = (ev, holder)

        req.clientMsgId = mid
        reactor.callFromThread(self._client.send, req)

        if not ev.wait(timeout):
            self._pending.pop(mid, None)
            return None
        return holder[0] if holder else None

    def _subscribe_spots(self, symbol_ids):
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId.extend(symbol_ids)
        reactor.callFromThread(self._client.send, req)

    def _resolve_symbol(self, pair):
        key = pair.replace('/', '_')
        sid = self._symbol_ids.get(key)
        if sid is None:
            raise ValueError(f"Symbol not found in cTrader symbol list: {pair}")
        return sid


def _bar_ms(timeframe):
    """Milliseconds per bar for a given timeframe."""
    mapping = {
        'M1': 60_000, 'M5': 300_000, 'M15': 900_000, 'M30': 1_800_000,
        'H1': 3_600_000, 'H4': 14_400_000, 'D1': 86_400_000, '1d': 86_400_000,
    }
    return mapping.get(timeframe, 3_600_000)


print("CTraderClient loaded")
