#!/usr/bin/env python3
"""
Live performance metrics tracker — persists across restarts.
Tracks all closed trades and computes: win rate, profit factor, Sharpe, max drawdown, slippage.
"""

import json
import math
import os
from datetime import datetime


class MetricsTracker:
    def __init__(self, persist_path='data/metrics.json'):
        self.persist_path = persist_path
        self.trades = []
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self._load()

    def record_trade(self, pair, direction, entry, close_price, pnl_usd, pnl_pips,
                     outcome, lot, slippage_pips=0.0, confidence=0.0):
        self.trades.append({
            'timestamp': datetime.utcnow().isoformat(),
            'pair': pair,
            'direction': direction,
            'entry': entry,
            'close': close_price,
            'pnl_usd': pnl_usd,
            'pnl_pips': pnl_pips,
            'outcome': outcome,
            'lot': lot,
            'slippage_pips': slippage_pips,
            'confidence': confidence,
        })
        self._save()

    def update_equity(self, equity):
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd > self.max_drawdown:
                self.max_drawdown = dd
                self._save()

    def compute(self, window=None):
        trades = self.trades[-window:] if window and window < len(self.trades) else self.trades
        if not trades:
            return {}

        n = len(trades)
        wins   = [t for t in trades if t['pnl_usd'] > 0]
        losses = [t for t in trades if t['pnl_usd'] <= 0]

        win_rate     = len(wins) / n
        gross_wins   = sum(t['pnl_usd'] for t in wins)
        gross_losses = abs(sum(t['pnl_usd'] for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        expectancy   = sum(t['pnl_usd'] for t in trades) / n

        # Annualised Sharpe from per-trade P&L (~208 trades/year at ~4/week)
        pnls = [t['pnl_usd'] for t in trades]
        avg  = expectancy
        if n > 1:
            variance = sum((p - avg) ** 2 for p in pnls) / (n - 1)
            std = math.sqrt(variance) if variance > 0 else 1e-9
            sharpe = (avg / std) * math.sqrt(208)
        else:
            sharpe = 0.0

        avg_slip = sum(t.get('slippage_pips', 0.0) for t in trades) / n

        # Per-pair breakdown
        pairs = {}
        for t in trades:
            p = t['pair']
            if p not in pairs:
                pairs[p] = {'n': 0, 'wins': 0, 'pnl': 0.0}
            pairs[p]['n']    += 1
            pairs[p]['wins'] += 1 if t['pnl_usd'] > 0 else 0
            pairs[p]['pnl']  += t['pnl_usd']

        return {
            'total_trades':      n,
            'win_rate':          win_rate,
            'profit_factor':     profit_factor,
            'expectancy':        expectancy,
            'sharpe':            sharpe,
            'max_drawdown':      self.max_drawdown,
            'avg_slippage_pips': avg_slip,
            'gross_wins':        gross_wins,
            'gross_losses':      gross_losses,
            'pairs':             pairs,
        }

    def format_telegram(self, window=20):
        all_m    = self.compute()
        recent_m = self.compute(window=window)
        if not all_m:
            return "No closed trades recorded yet."

        def pf_str(pf):
            return f"{pf:.2f}" if pf != float('inf') else "∞"

        lines = [
            "<b>📊 Live Performance Metrics</b>",
            f"<i>All time — {all_m['total_trades']} trades</i>",
            f"Win rate:       <b>{all_m['win_rate']:.1%}</b>",
            f"Profit factor:  <b>{pf_str(all_m['profit_factor'])}</b>  "
            f"(gross +${all_m['gross_wins']:,.0f} / -${all_m['gross_losses']:,.0f})",
            f"Expectancy:     <b>${all_m['expectancy']:+.2f}</b> / trade",
            f"Sharpe (ann.):  <b>{all_m['sharpe']:.2f}</b>",
            f"Max drawdown:   <b>{all_m['max_drawdown']:.1%}</b>",
            f"Avg slippage:   {all_m['avg_slippage_pips']:.1f} pips",
        ]

        if recent_m and recent_m['total_trades'] >= 5:
            lines += [
                "",
                f"<i>Last {recent_m['total_trades']} trades</i>",
                f"WR: <b>{recent_m['win_rate']:.1%}</b>  "
                f"PF: <b>{pf_str(recent_m['profit_factor'])}</b>  "
                f"E: <b>${recent_m['expectancy']:+.2f}</b>  "
                f"Sharpe: <b>{recent_m['sharpe']:.2f}</b>",
            ]

        if all_m.get('pairs'):
            lines.append("")
            lines.append("<i>Per pair</i>")
            for pair, s in sorted(all_m['pairs'].items()):
                wr = s['wins'] / s['n'] * 100 if s['n'] else 0
                lines.append(f"  {pair}: {s['n']}t  WR {wr:.0f}%  P&L ${s['pnl']:+,.0f}")

        return "\n".join(lines)

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.persist_path) or '.', exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump({
                    'trades':       self.trades,
                    'peak_equity':  self.peak_equity,
                    'max_drawdown': self.max_drawdown,
                }, f, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            self.trades       = data.get('trades', [])
            self.peak_equity  = data.get('peak_equity', 0.0)
            self.max_drawdown = data.get('max_drawdown', 0.0)
        except Exception:
            self.trades = []
            self.peak_equity = 0.0
            self.max_drawdown = 0.0
