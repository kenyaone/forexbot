from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Place and manage orders via cTrader or Deriv API (mock fallback)."""

    def __init__(self, ctrader_client=None, deriv_client=None, mt5_client=None, risk_manager=None):
        self.ct = ctrader_client
        self.deriv = deriv_client
        self.mt5 = mt5_client
        self.risk_manager = risk_manager
        self.open_orders = {}
        self.trade_log = []

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
        ticket = result.get('ticket', 0)
        return self._register_order(str(ticket), pair, direction, lot_size,
                                    entry_price, sl_price, tp_price, signal_confidence)

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
                                    entry_price, sl_price, tp_price, signal_confidence)

    def _register_order(self, order_id, pair, direction, lot_size,
                        entry_price, sl_price, tp_price, signal_confidence):
        order = {
            'order_id': order_id, 'pair': pair, 'direction': direction,
            'lot_size': lot_size, 'entry_price': entry_price,
            'sl_price': sl_price, 'tp_price': tp_price,
            'open_time': datetime.utcnow(), 'signal_confidence': signal_confidence,
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
        if order['direction'] == 'BUY':
            pnl_pips = (close_price - order['entry_price']) * 10000
        else:
            pnl_pips = (order['entry_price'] - close_price) * 10000

        pnl_usd = pnl_pips * order['lot_size'] * 10
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
            live = {str(p['ticket']) for p in self.mt5.get_positions()}
            for oid in [o for o in list(self.open_orders) if not o.startswith('MOCK_') and o not in live]:
                logger.info(f"Reconcile: removing stale MT5 order {oid}")
                order = self.open_orders.pop(oid)
                self.risk_manager.close_trade(order['pair'], order['entry_price'], 'SYNC')
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

    def get_trade_log(self):
        return self.trade_log

print("OrderExecutor (cTrader/Deriv) loaded")
