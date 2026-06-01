"""
Direct MT5 client via rpyc classic bridge.
Uses rpyc.classic.connect() to call the MetaTrader5 Python API on the remote
Windows machine. Results are serialised to plain dicts/primitives on the remote
side before being returned, which avoids the rpyc netref expiry ("result expired")
problem with NamedTuple objects.

Windows bridge: python -m mt5linux --host 0.0.0.0 -p 18812
"""

import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_SYMBOL_MAP = {
    'EUR/USD': 'EURUSD',
    'GBP/USD': 'GBPUSD',
    'USD/JPY': 'USDJPY',
    'AUD/USD': 'AUDUSD',
    'USD/CHF': 'USDCHF',
}


class MT5DirectClient:
    """Connect to MT5 on a remote Windows machine via rpyc classic + MetaTrader5 API."""

    def __init__(self, host: str, port: int = 18812):
        self.host = host
        self.port = port
        self._conn = None

    def _get_conn(self):
        if self._conn is not None:
            try:
                self._conn.ping()
                return self._conn
            except Exception:
                self._conn = None
        import rpyc
        self._conn = rpyc.classic.connect(self.host, self.port)
        self._conn.execute('import MetaTrader5 as mt5, subprocess, time')
        ok = self._conn.eval('mt5.initialize(timeout=5000)')
        if not ok:
            # MT5 terminal may be gone — restart it and retry once
            logger.warning('mt5.initialize failed, restarting terminal64.exe...')
            self._conn.execute(
                r"subprocess.Popen([r'C:\Program Files\MetaTrader 5\terminal64.exe'], "
                r"creationflags=0x08000000)"
            )
            self._conn.eval('time.sleep(20)')
            ok = self._conn.eval('mt5.initialize(timeout=10000)')
            if not ok:
                err = self._conn.eval('str(mt5.last_error())')
                self._conn = None
                raise RuntimeError(f'mt5.initialize failed after terminal restart: {err}')
            logger.info('MT5 terminal restarted and re-initialized successfully')
        return self._conn

    def _exec_and_get(self, setup_code, result_expr):
        """Execute setup_code on remote, then return eval(result_expr) as a local value.
        result_expr must evaluate to a primitive type (dict/list/str/int/float/bool/None).
        """
        conn = self._get_conn()
        conn.execute(setup_code)
        return conn.eval(result_expr)

    def ping(self, timeout=5.0):
        try:
            result = self._exec_and_get(
                '_ai = mt5.account_info()',
                '_ai.login if _ai else None'
            )
            return result is not None
        except Exception as e:
            logger.warning(f"MT5DirectClient ping failed: {e}")
            self._conn = None
            return False

    def get_account_info(self, timeout=5.0):
        try:
            result = self._exec_and_get(
                """
_ai = mt5.account_info()
_ai_dict = {"LOGIN": str(_ai.login), "BALANCE": str(round(_ai.balance, 2)),
             "EQUITY": str(round(_ai.equity, 2)), "SERVER": str(_ai.server),
             "ACCT_TRADE": "1" if _ai.trade_allowed else "0"} if _ai else None
""",
                '_ai_dict'
            )
            return result
        except Exception as e:
            logger.warning(f"MT5 get_account_info failed: {e}")
            return None

    def get_tick(self, pair, timeout=5.0):
        try:
            sym = _sym(pair)
            result = self._exec_and_get(
                f'_tick = mt5.symbol_info_tick("{sym}")',
                f'{{"bid": float(_tick.bid), "ask": float(_tick.ask)}} if _tick else None'
            )
            if result is None:
                return None
            return {'pair': pair, 'bid': result['bid'], 'ask': result['ask'],
                    'time': datetime.utcnow()}
        except Exception as e:
            logger.warning(f"MT5 get_tick({pair}) failed: {e}")
            return None

    def place_order(self, pair, direction, volume, sl_price, tp_price, timeout=10.0):
        try:
            sym = _sym(pair)

            # Get fresh tick
            tick = self._exec_and_get(
                f'_tick = mt5.symbol_info_tick("{sym}")',
                f'{{"bid": float(_tick.bid), "ask": float(_tick.ask)}} if _tick else None'
            )
            if tick is None:
                return {'success': False, 'reason': 'No tick data'}

            price = tick['ask'] if direction == 'BUY' else tick['bid']

            # Recalculate SL/TP from fresh price — prevents "Invalid stops" when
            # price has moved more than SL_PIPS since signal generation.
            sl_pips = int(os.getenv('SL_PIPS', '30'))
            tp_pips = int(os.getenv('TP_PIPS', '60'))
            pip = 0.01 if 'JPY' in sym else 0.0001
            if direction == 'BUY':
                sl = round(price - sl_pips * pip, 5)
                tp = round(price + tp_pips * pip, 5)
            else:
                sl = round(price + sl_pips * pip, 5)
                tp = round(price - tp_pips * pip, 5)

            # Detect filling mode supported by symbol (FOK=1 flag, IOC=2 flag)
            fill_mode = self._exec_and_get(
                f'_si = mt5.symbol_info("{sym}")',
                'int(_si.filling_mode) if _si else 1'
            )
            fill = 0   # ORDER_FILLING_FOK
            if fill_mode & 1:
                fill = 0
            elif fill_mode & 2:
                fill = 1   # ORDER_FILLING_IOC
            else:
                fill = 2   # ORDER_FILLING_RETURN

            order_type = 0 if direction == 'BUY' else 1
            vol = float(round(volume, 2))

            result = self._exec_and_get(
                f"""
_req = {{
    "action":       mt5.TRADE_ACTION_DEAL,
    "symbol":       "{sym}",
    "volume":       {vol},
    "type":         {order_type},
    "price":        {price},
    "sl":           {sl},
    "tp":           {tp},
    "deviation":    20,
    "magic":        20250518,
    "comment":      "ForexBot",
    "type_time":    mt5.ORDER_TIME_GTC,
    "type_filling": {fill},
}}
_r = mt5.order_send(_req)
_r_dict = {{"retcode": int(_r.retcode), "order": int(_r.order), "price": float(_r.price), "comment": str(_r.comment)}} if _r else None
""",
                '_r_dict'
            )

            if result is None:
                err = self._exec_and_get('', 'str(mt5.last_error())')
                return {'success': False, 'reason': f'order_send None: {err}'}

            if result['retcode'] == 10009:  # TRADE_RETCODE_DONE
                return {
                    'success': True,
                    'ticket': result['order'],
                    'price':  result['price'],
                    'reason': 'OK',
                }
            return {'success': False, 'reason': f"{result['retcode']}: {result['comment']}"}

        except Exception as e:
            logger.error(f"MT5 place_order failed: {e}")
            self._conn = None
            return {'success': False, 'reason': str(e)}

    def close_order(self, ticket, timeout=10.0):
        try:
            # Get position details
            pos = self._exec_and_get(
                f'_pos = mt5.positions_get(ticket={int(ticket)})',
                '{"symbol": str(_pos[0].symbol), "volume": float(_pos[0].volume), "type": int(_pos[0].type)} if _pos else None'
            )
            if pos is None:
                return {'success': False, 'reason': 'Position not found'}

            sym   = pos['symbol']
            vol   = pos['volume']
            ptype = pos['type']   # 0=BUY, 1=SELL

            tick = self._exec_and_get(
                f'_tick = mt5.symbol_info_tick("{sym}")',
                f'{{"bid": float(_tick.bid), "ask": float(_tick.ask)}} if _tick else None'
            )
            if tick is None:
                return {'success': False, 'reason': 'No tick data'}

            close_price = tick['bid'] if ptype == 0 else tick['ask']
            close_type  = 1 if ptype == 0 else 0

            fill_mode = self._exec_and_get(
                f'_si = mt5.symbol_info("{sym}")',
                'int(_si.filling_mode) if _si else 1'
            )
            fill = 0
            if fill_mode & 1:
                fill = 0
            elif fill_mode & 2:
                fill = 1
            else:
                fill = 2

            result = self._exec_and_get(
                f"""
_req = {{
    "action":       mt5.TRADE_ACTION_DEAL,
    "symbol":       "{sym}",
    "volume":       {vol},
    "type":         {close_type},
    "position":     {int(ticket)},
    "price":        {close_price},
    "deviation":    20,
    "magic":        20250518,
    "comment":      "ForexBot close",
    "type_time":    mt5.ORDER_TIME_GTC,
    "type_filling": {fill},
}}
_r = mt5.order_send(_req)
_r_dict = {{"retcode": int(_r.retcode), "comment": str(_r.comment)}} if _r else None
""",
                '_r_dict'
            )

            if result is None:
                return {'success': False, 'reason': 'close order_send returned None'}
            if result['retcode'] == 10009:
                return {'success': True}
            return {'success': False, 'reason': f"{result['retcode']}: {result['comment']}"}

        except Exception as e:
            logger.error(f"MT5 close_order failed: {e}")
            self._conn = None
            return {'success': False, 'reason': str(e)}

    def modify_order(self, ticket, sl_price, tp_price, timeout=10.0):
        try:
            pos = self._exec_and_get(
                f'_pos = mt5.positions_get(ticket={int(ticket)})',
                '{"symbol": str(_pos[0].symbol)} if _pos else None'
            )
            if pos is None:
                return {'success': False, 'reason': 'Position not found'}
            sym = pos['symbol']

            result = self._exec_and_get(
                f"""
_req = {{
    "action":   mt5.TRADE_ACTION_SLTP,
    "position": {int(ticket)},
    "symbol":   "{sym}",
    "sl":       {float(sl_price)},
    "tp":       {float(tp_price)},
}}
_r = mt5.order_send(_req)
_r_dict = {{"retcode": int(_r.retcode), "comment": str(_r.comment)}} if _r else None
""",
                '_r_dict'
            )

            if result is None:
                return {'success': False, 'reason': 'modify returned None'}
            if result['retcode'] == 10009:
                return {'success': True}
            return {'success': False, 'reason': f"{result['retcode']}: {result['comment']}"}

        except Exception as e:
            logger.error(f"MT5 modify_order failed: {e}")
            self._conn = None
            return {'success': False, 'reason': str(e)}

    def get_positions(self, timeout=5.0):
        try:
            result = self._exec_and_get(
                """
_positions = mt5.positions_get()
_pos_list = [{"ticket": int(p.ticket), "symbol": str(p.symbol),
               "direction": "BUY" if int(p.type) == 0 else "SELL",
               "volume": float(p.volume), "open_price": float(p.price_open),
               "sl": float(p.sl), "tp": float(p.tp), "profit": float(p.profit)}
              for p in (_positions or [])]
""",
                '_pos_list'
            )
            return result or []
        except Exception as e:
            logger.warning(f"MT5 get_positions failed: {e}")
            return []

    def get_deal_history(self, from_date=None, to_date=None):
        """Return completed trades (matched IN+OUT deals) from MT5 history."""
        try:
            from datetime import timedelta
            if from_date is None:
                from_date = datetime.utcnow() - timedelta(days=90)
            if to_date is None:
                to_date = datetime.utcnow() + timedelta(hours=1)

            from_ts = int(from_date.timestamp())
            to_ts   = int(to_date.timestamp())

            deals = self._exec_and_get(
                f"""
import datetime as _dt
_deals = mt5.history_deals_get(
    _dt.datetime.utcfromtimestamp({from_ts}),
    _dt.datetime.utcfromtimestamp({to_ts})
)
_dl = []
if _deals:
    for _d in _deals:
        _dl.append({{
            'ticket':      int(_d.ticket),
            'position_id': int(_d.position_id),
            'time':        int(_d.time),
            'symbol':      str(_d.symbol),
            'type':        int(_d.type),   # 0=BUY deal, 1=SELL deal
            'entry':       int(_d.entry),  # 0=IN, 1=OUT
            'volume':      float(_d.volume),
            'price':       float(_d.price),
            'profit':      float(_d.profit),
            'commission':  float(_d.commission),
            'swap':        float(_d.swap),
            'magic':       int(_d.magic),
        }})
""",
                '_dl'
            )
            if not deals:
                return []

            # Group by position_id, match IN deal to OUT deal
            by_pos = {}
            for d in deals:
                pid = d['position_id']
                if pid not in by_pos:
                    by_pos[pid] = {'in': None, 'out': None}
                if d['entry'] == 0:   # DEAL_ENTRY_IN
                    by_pos[pid]['in'] = d
                elif d['entry'] == 1: # DEAL_ENTRY_OUT
                    by_pos[pid]['out'] = d

            trades = []
            for pid, pair_deals in by_pos.items():
                d_in  = pair_deals['in']
                d_out = pair_deals['out']
                if d_in is None or d_out is None:
                    continue  # position still open or partial data
                # Opening deal type tells us position direction
                direction = 'BUY' if d_in['type'] == 0 else 'SELL'
                pnl = d_out['profit'] + d_out['commission'] + d_out['swap']
                trades.append({
                    'position_id':  pid,
                    'close_ticket': d_out['ticket'],
                    'symbol':       d_out['symbol'],
                    'direction':    direction,
                    'volume':       d_in['volume'],
                    'entry_price':  d_in['price'],
                    'close_price':  d_out['price'],
                    'open_time':    d_in['time'],
                    'close_time':   d_out['time'],
                    'profit':       d_out['profit'],
                    'commission':   d_out['commission'],
                    'swap':         d_out['swap'],
                    'pnl':          pnl,
                    'magic':        d_out['magic'],
                })
            return trades

        except Exception as e:
            logger.warning(f"MT5 get_deal_history failed: {e}")
            return []


def _sym(pair):
    s = _SYMBOL_MAP.get(pair)
    if s is None:
        raise ValueError(f"Pair not supported: {pair}")
    return s


print("MT5DirectClient loaded")
