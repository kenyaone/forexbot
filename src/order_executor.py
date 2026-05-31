from datetime import datetime
import logging

logger = logging.getLogger(__name__)

_MT5_SYM_TO_PAIR = {
    'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'USDJPY': 'USD/JPY',
    'AUDUSD': 'AUD/USD', 'USDCHF': 'USD/CHF',
}


class OrderExecutor:
    """Place and manage orders via cTrader or Deriv API (mock fallback)."""

    def __init__(self, ctrader_client=None, deriv_client=None, mt5_client=None, risk_manager=None):
        self.ct = ctrader_client
        self.deriv = deriv_client
        self.mt5 = mt5_client
        self.risk_manager = risk_manager
        self.open_orders = {}
        self.trade_log = []
        self.stale_closes = []  # drained by main loop to send alerts + update daily_pnl

    # ------------------------------------------------------------------
    def place_order(self, pair, direction, lot_size, entry_price, sl_price, tp_price,
                    signal_confidence=0.65):

        if not self.risk_manager.can_trade():
            return {'success': False, 'reason': f'Trading halted: {self.risk_manager.state}'}

        if len(self.risk_manager.open_trades) >= self.risk_manager.max_concurrent_trades:
            return {'success': False, 'reason': 'Max concurrent trades reached'}

        lot_size *= self.risk_manager.get_lot_multiplier()
        if lot_size < 0.01:
            return {'success': False, 'reason': 'Lot size too small after scaling'}

        if self.mt5 is not None:
            return self._mt5_place(pair, direction, lot_size, entry_price,
                                   sl_price, tp_price, signal_confidence)

        if self.ct is not None:
            return self._ctrader_place(pair, direction, lot_size, entry_price,
                                       sl_price, tp_price, signal_confidence)

        if self.deriv is not None:
            return self._deriv_place(pair, direction, lot_size, entry_price,
                                     sl_price, tp_price, signal_confidence)

        return self._mock_place(pair, direction, lot_size, entry_price,
                                sl_price, tp_price, signal_confidence)

    def _mt5_place(self, pair, direction, lot_size, entry_price,
                   sl_price, tp_price, signal_confidence):
        result = self.mt5.place_order(pair, direction, lot_size, sl_price, tp_price)
        if not result['success']:
            return {'success': False, 'reason': result['reason']}
        ticket     = result.get('ticket', 0)
        fill_price = result.get('fill_price') or result.get('price') or entry_price
        pip        = 0.01 if 'JPY' in pair else 0.0001
        slippage   = ((fill_price - entry_price) / pip if direction == 'BUY'
                      else (entry_price - fill_price) / pip)
        return self._register_order(str(ticket), pair, direction, lot_size,
                                    fill_price, sl_price, tp_price, signal_confidence,
                                    slippage_pips=slippage)

    def _ctrader_place(self, pair, direction, lot_size, entry_price,
                       sl_price, tp_price, signal_confidence):
        result = self.ct.place_order(pair, direction, lot_size, sl_price, tp_price)
        if not result['success']:
            return {'success': False, 'reason': result['reason']}

        pos_id = result.get('position_id') or f"CT_{pair}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        return self._register_order(str(pos_id), pair, direction, lot_size,
                                    entry_price, sl_price, tp_price, signal_confidence)

    def _deriv_place(self, pair, direction, lot_size, entry_price,
                     sl_price, tp_price, signal_confidence):
        # Convert lot_size to stake: 1 lot risk at 2% of $10k = $200 stake
        account_value = self.risk_manager.account_equity
        risk_pips = abs(entry_price - sl_price) * 10000
        stake_usd = max(round(account_value * 0.02, 2), 1.0)

        result = self.deriv.place_order(
            pair, direction, stake_usd, sl_price, tp_price, entry_price
        )
        if not result['success']:
            return {'success': False, 'reason': result['reason']}

        contract_id = str(result['contract_id'])
        return self._register_order(contract_id, pair, direction, lot_size,
                                    entry_price, sl_price, tp_price, signal_confidence)

    def _mock_place(self, pair, direction, lot_size, entry_price,
                    sl_price, tp_price, signal_confidence):
        order_id = f"MOCK_{pair}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        return self._register_order(order_id, pair, direction, lot_size,
                                    entry_price, sl_price, tp_price, signal_confidence,
                                    slippage_pips=0.0)

    def _register_order(self, order_id, pair, direction, lot_size,
                        entry_price, sl_price, tp_price, signal_confidence,
                        slippage_pips=0.0):
        order = {
            'order_id': order_id, 'pair': pair, 'direction': direction,
            'lot_size': lot_size, 'entry_price': entry_price,
            'sl_price': sl_price, 'tp_price': tp_price,
            'open_time': datetime.utcnow(), 'signal_confidence': signal_confidence,
            'slippage_pips': slippage_pips,
            'status': 'OPEN',
        }
        self.open_orders[order_id] = order
        self.risk_manager.add_trade(pair, direction, entry_price, sl_price, tp_price, lot_size)
        self.trade_log.append({'timestamp': datetime.utcnow(), 'action': 'OPEN', 'order': order})
        return {'success': True, 'order_id': order_id, 'order': order}

    # ------------------------------------------------------------------
    def close_order(self, order_id, close_price, reason='MANUAL'):
        if order_id not in self.open_orders:
            return {'success': False, 'reason': 'Order not found'}

        order = self.open_orders[order_id]

        if self.mt5 is not None and not order_id.startswith('MOCK_'):
            try:
                self.mt5.close_order(int(order_id))
            except Exception as e:
                logger.warning(f"MT5 close failed: {e}")

        elif self.ct is not None and not order_id.startswith('MOCK_'):
            try:
                self.ct.close_position(int(order_id))
            except Exception as e:
                logger.warning(f"cTrader close failed: {e}")

        elif self.deriv is not None and not order_id.startswith('MOCK_'):
            try:
                self.deriv.close_position(int(order_id))
            except Exception as e:
                logger.warning(f"Deriv close failed: {e}")

        return self._finalize_close(order_id, order, close_price, reason)

    def _finalize_close(self, order_id, order, close_price, reason):
        pip = 0.01 if 'JPY' in order['pair'] else 0.0001
        pip_val = (pip / close_price * 100000) if 'JPY' in order['pair'] and close_price > 0 else 10.0
        if order['direction'] == 'BUY':
            pnl_pips = (close_price - order['entry_price']) / pip
        else:
            pnl_pips = (order['entry_price'] - close_price) / pip

        pnl_usd = pnl_pips * order['lot_size'] * pip_val
        order.update({
            'close_price': close_price, 'close_time': datetime.utcnow(),
            'pnl_pips': pnl_pips, 'pnl_usd': pnl_usd,
            'close_reason': reason, 'status': 'CLOSED',
        })
        del self.open_orders[order_id]
        self.risk_manager.close_trade(order['pair'], close_price, reason)
        self.trade_log.append({'timestamp': datetime.utcnow(), 'action': 'CLOSE', 'order': order})
        return {'success': True, 'order_id': order_id, 'pnl_usd': pnl_usd, 'pnl_pips': pnl_pips}

    # ------------------------------------------------------------------
    def sync_open_trades(self):
        """Reconcile local state with broker open positions."""
        if self.mt5 is not None:
            self._sync_mt5()
        elif self.ct is not None:
            self._sync_ctrader()
        elif self.deriv is not None:
            self._sync_deriv()

    def _sync_mt5(self):
        try:
            positions = self.mt5.get_positions()
            live_tickets = {str(p['ticket']) for p in positions}

            # Remove stale local orders (closed by MT5 via SL/TP)
            for oid in [o for o in list(self.open_orders) if not o.startswith('MOCK_') and o not in live_tickets]:
                order = self.open_orders[oid]
                # Estimate close price: TP if price moved in profit direction, else SL
                try:
                    tick = self.mt5.get_tick(order['pair'])
                    mid = (tick['bid'] + tick['ask']) / 2 if tick else order['entry_price']
                except Exception:
                    mid = order['entry_price']
                pip = 0.01 if 'JPY' in order['pair'] else 0.0001
                if order['direction'] == 'BUY':
                    close_price = order['tp_price'] if mid >= order['tp_price'] else order['sl_price']
                else:
                    close_price = order['tp_price'] if mid <= order['tp_price'] else order['sl_price']
                result = self._finalize_close(oid, order, close_price, 'MT5_CLOSED')
                logger.info(f"Reconcile: MT5 closed {oid} ({order['pair']} {order['direction']}) | est. P&L: ${result['pnl_usd']:.2f}")
                self.stale_closes.append({'order': order, 'pnl_usd': result['pnl_usd'],
                                          'close_price': close_price, 'reason': 'MT5_CLOSED'})

            # Add MT5 positions not yet tracked locally (e.g. after bot restart)
            for pos in positions:
                tid = str(pos['ticket'])
                if tid not in self.open_orders:
                    pair = _MT5_SYM_TO_PAIR.get(pos['symbol'], pos['symbol'])
                    order = {
                        'order_id': tid, 'pair': pair, 'direction': pos['direction'],
                        'lot_size': pos['volume'], 'entry_price': pos['open_price'],
                        'sl_price': pos['sl'], 'tp_price': pos['tp'],
                        'open_time': datetime.utcnow(), 'signal_confidence': 0.0,
                        'status': 'OPEN',
                    }
                    self.open_orders[tid] = order
                    self.risk_manager.add_trade(pair, pos['direction'], pos['open_price'],
                                                pos['sl'], pos['tp'], pos['volume'])
                    logger.info(f"Reconcile: loaded MT5 position {tid} ({pair} {pos['direction']})")
                    # Alert so user knows about positions loaded on startup
                    try:
                        from src.alerting import alert_trade_opened
                        alert_trade_opened(
                            pair=pair, direction=pos['direction'], volume=pos['volume'],
                            entry=pos['open_price'], sl=pos['sl'], tp=pos['tp'],
                            confidence=0.0, ticket=int(tid),
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"MT5 sync failed: {e}")

    def _sync_ctrader(self):
        try:
            live = {str(p.positionId) for p in self.ct.get_open_positions()}
            for oid in [o for o in list(self.open_orders) if not o.startswith('MOCK_') and o not in live]:
                logger.info(f"Reconcile: removing stale order {oid}")
                order = self.open_orders.pop(oid)
                self.risk_manager.close_trade(order['pair'], order['entry_price'], 'SYNC')
        except Exception as e:
            logger.warning(f"cTrader sync failed: {e}")

    def _sync_deriv(self):
        try:
            live = {str(c['contract_id']) for c in self.deriv.get_open_contracts()}
            for oid in [o for o in list(self.open_orders) if not o.startswith('MOCK_') and o not in live]:
                logger.info(f"Reconcile: removing stale order {oid}")
                order = self.open_orders.pop(oid)
                self.risk_manager.close_trade(order['pair'], order['entry_price'], 'SYNC')
        except Exception as e:
            logger.warning(f"Deriv sync failed: {e}")

    def check_exit_conditions(self, order_id, current_price, current_time):
        if order_id not in self.open_orders:
            return {'should_close': False}

        order = self.open_orders[order_id]

        if order['direction'] == 'BUY':
            if current_price <= order['sl_price']:
                return {'should_close': True, 'reason': 'SL', 'close_price': order['sl_price']}
            if current_price >= order['tp_price']:
                return {'should_close': True, 'reason': 'TP', 'close_price': order['tp_price']}
        else:
            if current_price >= order['sl_price']:
                return {'should_close': True, 'reason': 'SL', 'close_price': order['sl_price']}
            if current_price <= order['tp_price']:
                return {'should_close': True, 'reason': 'TP', 'close_price': order['tp_price']}

        hours_open = (current_time - order['open_time']).total_seconds() / 3600
        if hours_open >= 48:
            return {'should_close': True, 'reason': 'TIME', 'close_price': current_price}

        return {'should_close': False}

    def apply_trailing_stop(self, order_id, current_price):
        """Trail SL once 20+ pips in profit, keeping it 15 pips behind price."""
        if order_id not in self.open_orders:
            return False
        order = self.open_orders[order_id]
        pip = 0.01 if 'JPY' in order['pair'] else 0.0001
        trail_trigger = 20 * pip
        trail_dist    = 15 * pip

        if order['direction'] == 'BUY':
            profit = current_price - order['entry_price']
            if profit >= trail_trigger:
                new_sl = round(current_price - trail_dist, 5)
                if new_sl > order['sl_price']:
                    self._update_sl(order_id, order, new_sl, order['tp_price'])
                    return True
        else:
            profit = order['entry_price'] - current_price
            if profit >= trail_trigger:
                new_sl = round(current_price + trail_dist, 5)
                if new_sl < order['sl_price']:
                    self._update_sl(order_id, order, new_sl, order['tp_price'])
                    return True
        return False

    def _update_sl(self, order_id, order, new_sl, tp):
        if self.mt5 is not None:
            try:
                result = self.mt5.modify_order(int(order_id), new_sl, tp)
                if result.get('success'):
                    old_sl = order['sl_price']
                    order['sl_price'] = new_sl
                    logger.info(f"Trail stop {order['pair']} SL {old_sl:.5f} → {new_sl:.5f}")
            except Exception as e:
                logger.warning(f"Trailing stop modify failed for {order_id}: {e}")

    def get_trade_log(self):
        return self.trade_log
