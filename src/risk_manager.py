import pandas as pd
from enum import Enum

class RiskState(Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    REDUCED = "REDUCED"
    HALTED = "HALTED"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class RiskManager:
    """Manage position sizing, risk limits, and account protection"""
    
    def __init__(self, account_equity, risk_per_trade=0.02, max_daily_loss=0.05, 
                 max_drawdown=0.25, max_concurrent_trades=3):
        self.account_equity = account_equity
        self.peak_equity = account_equity
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_concurrent_trades = max_concurrent_trades
        self.state = RiskState.NORMAL
        self.daily_loss = 0
        self.open_trades = []
    
    def calculate_position_size(self, pair, entry_price, sl_price):
        """Position sizing: risk_per_trade % of equity over SL distance."""
        risk_amount = self.account_equity * self.risk_per_trade
        pip = 0.01 if 'JPY' in str(pair) else 0.0001
        sl_distance_pips = round(abs(entry_price - sl_price) / pip)
        if sl_distance_pips == 0:
            return 0.01
        # Pip value per standard lot in USD
        if 'JPY' in str(pair) and entry_price > 0:
            pip_value_per_lot = (pip / entry_price) * 100000
        else:
            pip_value_per_lot = 10.0
        lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)
        lot_size = max(round(lot_size, 2), 0.01)
        return lot_size
    
    def check_risk_state(self, current_equity, daily_pnl):
        """Update risk state based on equity and daily loss"""
        self.daily_loss = daily_pnl
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        daily_loss_pct = abs(daily_pnl) / self.account_equity if self.account_equity > 0 else 0
        
        # State machine
        if drawdown >= self.max_drawdown:
            self.state = RiskState.EMERGENCY_STOP
        elif daily_loss_pct >= self.max_daily_loss:
            self.state = RiskState.HALTED
        elif drawdown >= 0.15:
            self.state = RiskState.REDUCED
        elif daily_loss_pct >= 0.04:
            self.state = RiskState.CAUTION
        else:
            self.state = RiskState.NORMAL
        
        return self.state
    
    def can_trade(self):
        """Check if trading is allowed in current state"""
        return self.state in [RiskState.NORMAL, RiskState.CAUTION, RiskState.REDUCED]
    
    def get_lot_multiplier(self):
        """Reduce position size based on risk state"""
        multipliers = {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTION: 0.75,
            RiskState.REDUCED: 0.50,
            RiskState.HALTED: 0.0,
            RiskState.EMERGENCY_STOP: 0.0,
        }
        return multipliers.get(self.state, 0)
    
    def check_correlation(self, open_pairs, new_pair, correlation_matrix, threshold=0.7):
        """
        Prevent trading correlated pairs simultaneously
        """
        for pair in open_pairs:
            if pair in correlation_matrix.index and new_pair in correlation_matrix.columns:
                corr = correlation_matrix.loc[pair, new_pair]
                if abs(corr) > threshold:
                    return False  # Too correlated
        return True  # Safe to trade
    
    def add_trade(self, pair, direction, entry_price, sl_price, tp_price, lot_size):
        """Record an open trade"""
        trade = {
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'lot_size': lot_size,
            'pnl': 0
        }
        self.open_trades.append(trade)
        return trade
    
    def close_trade(self, pair, close_price, reason='TP'):
        """Close a trade and record P&L"""
        for trade in self.open_trades:
            if trade['pair'] == pair:
                pip = 0.01 if 'JPY' in pair else 0.0001
                pip_val = (pip / close_price * 100000) if 'JPY' in pair and close_price > 0 else 10.0
                if trade.get('direction') == 'BUY':
                    pnl_pips = (close_price - trade['entry_price']) / pip
                else:
                    pnl_pips = (trade['entry_price'] - close_price) / pip
                pnl_usd = pnl_pips * trade['lot_size'] * pip_val
                trade['close_price'] = close_price
                trade['pnl'] = pnl_usd
                trade['close_reason'] = reason
                self.open_trades.remove(trade)
                self.daily_loss += pnl_usd
                return trade
        return None
