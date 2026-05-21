#!/usr/bin/env python3
"""
Standalone backtest — simulates the live bot strategy on 2 years of H1 data.
Precomputes all indicators and ML features on the full dataset (fast).

Usage:
    python run_backtest.py
    python run_backtest.py --sl 25 --tp 50 --threshold 0.38
"""

import argparse, sys, threading
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, '/home/tele/forex-bot')
from src.ml_model import make_features
import joblib
import lightgbm as lgb

PAIRS = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CHF': 'USDCHF=X',
}
SPREADS = {'EUR/USD': 1.0, 'GBP/USD': 1.5, 'USD/JPY': 1.2, 'AUD/USD': 1.5, 'USD/CHF': 1.8}
CROSS_ASSETS = {'dxy': 'DX-Y.NYB', 'gold': 'GC=F', 'tnx': '^TNX', 'vix': '^VIX'}


def _fetch_yf(ticker, start, end, interval, timeout_sec=40):
    result = [None]
    def _dl():
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             progress=False, auto_adjust=True)
            result[0] = df
        except Exception:
            pass
    t = threading.Thread(target=_dl, daemon=True)
    t.start(); t.join(timeout_sec)
    return result[0]


def fetch_h1(ticker, start, end, timeout_sec=40):
    df = _fetch_yf(ticker, start, end, '1h', timeout_sec)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low',
                             'Close':'close','Volume':'volume'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def fetch_cross_assets(start, end, timeout_sec=30):
    """Fetch daily DXY, Gold, TNX, VIX and return combined DataFrame."""
    frames = {}
    for name, ticker in CROSS_ASSETS.items():
        df = _fetch_yf(ticker, start - timedelta(days=10), end, '1d', timeout_sec)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close_col = 'Close' if 'Close' in df.columns else df.columns[3]
        s = df[close_col].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        frames[f'{name}_close'] = s
    return pd.DataFrame(frames) if frames else pd.DataFrame()


def merge_cross_assets(df_h1, cross_df):
    """Forward-fill daily cross-asset data into hourly index."""
    if cross_df.empty:
        return df_h1
    aligned = cross_df.reindex(df_h1.index, method='ffill')
    return df_h1.join(aligned, how='left')


def precompute_signals(df, model, feature_names, threshold):
    """
    Compute indicators and ML confidence on the FULL dataset at once.
    Returns a boolean Series: True where a signal fires and ML passes.
    Also returns direction for each bar.
    """
    from src.signal_engine import SignalEngine
    se = SignalEngine(ml_confidence_threshold=0.0)  # no threshold here

    # --- Indicators (vectorized over full dataset) ---
    from src.indicators.indicators import Indicators
    ind = Indicators()
    df = df.copy()
    df['ema20'] = ind.ema(df['close'], 20)
    df['ema50'] = ind.ema(df['close'], 50)
    df['rsi']   = ind.rsi(df['close'], 14)
    df['atr']   = ind.atr(df['high'], df['low'], df['close'], 14)
    df['atr_ma10'] = df['atr'].rolling(10).mean()
    df['adx'], df['di_plus'], df['di_minus'] = ind.adx(
        df['high'], df['low'], df['close'], 14)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = ind.bollinger_bands(df['close'], 20, 2.0)

    # --- ML features (vectorized) ---
    feat = make_features(df, compute_target=False)

    # Align feat to df index
    ml_conf_series = pd.Series(np.nan, index=df.index)
    if not feat.empty:
        valid_feat = feat[feature_names]
        proba = model.predict_proba(valid_feat)[:, 1]
        ml_conf_series.loc[feat.index] = proba

    # --- Signal logic (vectorized) ---
    close    = df['close']
    rsi      = df['rsi']
    adx      = df['adx']
    bb_upper = df['bb_upper']
    bb_lower = df['bb_lower']
    atr      = df['atr']
    atr_ma10 = df['atr_ma10']
    ema20    = df['ema20']
    ema50    = df['ema50']

    atr_expanding = atr > atr_ma10 * 1.2

    # Mean reversion
    mr_buy  = (~atr_expanding) & (close <= bb_lower * 1.02) & (rsi < 38)
    mr_sell = (~atr_expanding) & (close >= bb_upper * 0.98) & (rsi > 65)

    # Trend following
    tf_buy  = atr_expanding & (adx > 30) & (ema20 > ema50) & (rsi > 40) & (rsi < 65)
    tf_sell = atr_expanding & (adx > 30) & (ema20 < ema50) & (rsi > 35) & (rsi < 60)

    direction = pd.Series('NONE', index=df.index)
    direction[mr_buy | tf_buy]   = 'BUY'
    direction[mr_sell | tf_sell] = 'SELL'
    # SELL overrides BUY if both fire (shouldn't happen, but safe)

    ml_pass = ml_conf_series >= threshold
    signal_fires = (direction != 'NONE') & ml_pass

    return signal_fires, direction, ml_conf_series


def simulate_pair(pair, df, model, feature_names, sl_pips, tp_pips,
                  threshold, risk_pct=0.02, starting_equity=10000):
    pip        = 0.01 if 'JPY' in pair else 0.0001
    spread_pip = SPREADS.get(pair, 1.5)
    equity     = starting_equity
    trades     = []

    signal_fires, direction, ml_conf = precompute_signals(df, model, feature_names, threshold)

    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    times  = df.index.to_numpy()

    # Minimum bars warmup
    warmup = 60
    last_exit = -1  # avoid overlapping trades on same pair

    for i in range(warmup, len(df) - 49):
        if i <= last_exit:
            continue

        # Session filter: London open or NY open
        hour = pd.Timestamp(times[i]).hour
        if not ((7 <= hour < 10) or (13 <= hour < 16)):
            continue

        if not signal_fires.iloc[i]:
            continue

        d      = direction.iloc[i]
        entry  = closes[i] + spread_pip * pip * (1 if d == 'BUY' else -1)
        tp_lvl = entry + tp_pips * pip if d == 'BUY' else entry - tp_pips * pip
        sl_lvl = entry - sl_pips * pip if d == 'BUY' else entry + sl_pips * pip

        pip_val = (pip / entry * 100000) if 'JPY' in pair else 10.0
        lot     = max(round((equity * risk_pct) / (sl_pips * pip_val), 2), 0.01)

        outcome = 'TIMEOUT'
        close_price = closes[min(i + 48, len(df) - 1)]
        for j in range(i + 1, min(i + 49, len(df))):
            if d == 'BUY':
                if highs[j] >= tp_lvl: outcome = 'TP'; close_price = tp_lvl; break
                if lows[j]  <= sl_lvl: outcome = 'SL'; close_price = sl_lvl; break
            else:
                if lows[j]  <= tp_lvl: outcome = 'TP'; close_price = tp_lvl; break
                if highs[j] >= sl_lvl: outcome = 'SL'; close_price = sl_lvl; break

        pips_gained = (close_price - entry)/pip if d == 'BUY' else (entry - close_price)/pip
        pnl    = pips_gained * lot * pip_val
        equity += pnl
        last_exit = i + 48

        trades.append({
            'time': pd.Timestamp(times[i]), 'pair': pair, 'direction': d,
            'lot': lot, 'pips': pips_gained, 'pnl': pnl, 'outcome': outcome,
            'ml_conf': ml_conf.iloc[i], 'equity': equity,
        })

    return trades


def print_report(all_trades, starting_equity):
    if not all_trades:
        print("No trades. Try lower --threshold or check signal conditions.")
        return

    df = pd.DataFrame(all_trades).sort_values('time')
    wins  = df[df['outcome'] == 'TP']
    total = len(df)
    win_rate = len(wins) / total * 100

    running = starting_equity; peak = starting_equity; max_dd = 0
    for _, row in df.iterrows():
        running += row['pnl']
        peak = max(peak, running)
        max_dd = max(max_dd, (peak - running) / peak * 100)

    final_equity = running
    total_return = (final_equity - starting_equity) / starting_equity * 100
    avg_win  = wins['pnl'].mean() if len(wins) else 0
    losses   = df[df['outcome'] != 'TP']
    avg_loss = losses['pnl'].mean() if len(losses) else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    print("\n" + "=" * 60)
    print("  BACKTEST REPORT — 2-Year H1 Simulation")
    print("=" * 60)
    print(f"  Period       : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"  Pairs        : {', '.join(df['pair'].unique())}")
    print(f"  Total trades : {total}")
    print(f"  Win rate     : {win_rate:.1f}%  ({len(wins)} TP / {len(losses)} SL+timeout)")
    print(f"  Avg win      : ${avg_win:+.2f}")
    print(f"  Avg loss     : ${avg_loss:+.2f}")
    print(f"  Expectancy   : ${expectancy:+.2f} per trade")
    print(f"  Total return : {total_return:+.1f}%  (${final_equity - starting_equity:+,.0f})")
    print(f"  Max drawdown : {max_dd:.1f}%")
    print(f"  Final equity : ${final_equity:,.2f}")
    print("=" * 60)
    print("\nPer-pair:")
    for pair in df['pair'].unique():
        p  = df[df['pair'] == pair]
        wr = len(p[p['outcome']=='TP']) / len(p) * 100
        print(f"  {pair:10s}  {len(p):3d} trades  WR {wr:.0f}%  P&L ${p['pnl'].sum():+,.0f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sl',        type=int,   default=30)
    parser.add_argument('--tp',        type=int,   default=60)
    parser.add_argument('--threshold', type=float, default=0.38)
    parser.add_argument('--risk',      type=float, default=0.02)
    parser.add_argument('--equity',    type=float, default=10000)
    args = parser.parse_args()

    print(f"SL={args.sl}p  TP={args.tp}p  ML≥{args.threshold}  risk={args.risk:.0%}")
    payload       = joblib.load('config/ml_model.pkl')
    model         = payload['model']
    feature_names = payload['feature_names']
    wf = payload.get('walk_forward', {})
    print(f"Model AUC={wf.get('auc',0):.3f}  trained {payload.get('trained_at','?')[:10]}")

    end = datetime.now(); start = end - timedelta(days=729)
    print(f"Fetching H1 data {start.date()} → {end.date()}...")
    print("  Fetching cross-assets (DXY/Gold/TNX/VIX)...", end='', flush=True)
    cross_df = fetch_cross_assets(start, end)
    print(f" {len(cross_df.columns)} series loaded" if not cross_df.empty else " none")

    all_trades = []
    for pair, ticker in PAIRS.items():
        print(f"  {pair}...", end='', flush=True)
        df = fetch_h1(ticker, start, end, timeout_sec=40)
        if df is None or len(df) < 100:
            print(" skipped"); continue
        df = merge_cross_assets(df, cross_df)
        trades = simulate_pair(pair, df, model, feature_names,
                               args.sl, args.tp, args.threshold,
                               args.risk, args.equity)
        all_trades.extend(trades)
        wr = sum(1 for t in trades if t['outcome']=='TP') / max(len(trades), 1) * 100
        print(f" {len(trades)} trades  WR {wr:.0f}%")

    print_report(all_trades, args.equity)


if __name__ == '__main__':
    main()
