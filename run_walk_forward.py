#!/usr/bin/env python3
"""
Walk-forward validation with full trade simulation per fold.
Splits history into rolling train/test windows, trains a fresh model on each,
then simulates actual trades on the out-of-sample test window.

Gives real evidence of out-of-sample edge — not just AUC.

Usage:
    python run_walk_forward.py
    python run_walk_forward.py --train_days 270 --test_days 90 --threshold 0.54
    python run_walk_forward.py --sl 25 --tp 50 --threshold 0.52
"""

import argparse, sys, threading
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/tele/forex-bot')
from src.ml_model import make_features

PAIRS = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CHF': 'USDCHF=X',
}
SPREADS = {'EUR/USD': 1.0, 'GBP/USD': 1.5, 'USD/JPY': 1.2, 'AUD/USD': 1.5, 'USD/CHF': 1.8}
CROSS_ASSETS = {'dxy': 'DX-Y.NYB', 'gold': 'GC=F', 'tnx': '^TNX', 'vix': '^VIX'}
WARMUP_BARS = 80  # bars needed before test period to warm up indicators


def _fetch_yf(ticker, start, end, interval, timeout_sec=45):
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


def fetch_h1(ticker, start, end):
    df = _fetch_yf(ticker, start, end, '1h')
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                             'Close': 'close', 'Volume': 'volume'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']].dropna()


def fetch_cross_assets(start, end):
    frames = {}
    for name, ticker in CROSS_ASSETS.items():
        df = _fetch_yf(ticker, start - timedelta(days=10), end, '1d')
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = 'Close' if 'Close' in df.columns else df.columns[3]
        s = df[col].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        frames[f'{name}_close'] = s
    return pd.DataFrame(frames) if frames else pd.DataFrame()


def merge_cross(df_h1, cross_df):
    if cross_df.empty:
        return df_h1
    return df_h1.join(cross_df.reindex(df_h1.index, method='ffill'), how='left')


def train_model(X, y):
    pos_weight = (1 - y.mean()) / y.mean() if y.mean() > 0 else 1.0
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.02, max_depth=5, num_leaves=25,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        reg_alpha=0.2, reg_lambda=0.2, scale_pos_weight=pos_weight,
        random_state=42, verbose=-1,
    )
    m.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])
    return m


def simulate_trades(pair, df, model, feature_names, sl_pips, tp_pips,
                    threshold, risk_pct, starting_equity, test_start):
    """Simulate trades on df; only count entries at or after test_start."""
    pip        = 0.01 if 'JPY' in pair else 0.0001
    spread_pip = SPREADS.get(pair, 1.5)
    equity     = starting_equity
    trades     = []

    # Compute features on full df (with warmup), filter signals to test window
    feat = make_features(df, compute_target=False)
    if feat.empty:
        return trades

    valid_feat = feat[feature_names] if all(c in feat.columns for c in feature_names) else feat
    proba = model.predict_proba(valid_feat)[:, 1]
    ml_conf = pd.Series(proba, index=feat.index)

    # Signal logic — must stay in sync with run_backtest.py
    from src.indicators.indicators import Indicators
    ind = Indicators()
    d = df.copy()
    d['ema20'] = ind.ema(d['close'], 20)
    d['ema50'] = ind.ema(d['close'], 50)
    d['rsi']   = ind.rsi(d['close'], 14)
    d['atr']   = ind.atr(d['high'], d['low'], d['close'], 14)
    d['atr_ma10'] = d['atr'].rolling(10).mean()
    d['adx'], d['di_plus'], d['di_minus'] = ind.adx(d['high'], d['low'], d['close'], 14)
    d['bb_upper'], d['bb_mid'], d['bb_lower'] = ind.bollinger_bands(d['close'], 20, 2.0)

    atr_expanding = d['atr'] > d['atr_ma10'] * 1.2
    mr_buy  = (~atr_expanding) & (d['close'] <= d['bb_lower'] * 1.02) & (d['rsi'] < 42)
    mr_sell = (~atr_expanding) & (d['close'] >= d['bb_upper'] * 0.98) & (d['rsi'] > 65)
    tf_buy  = atr_expanding & (d['adx'] > 30) & (d['ema20'] > d['ema50']) & (d['rsi'] > 40) & (d['rsi'] < 65)
    tf_sell = atr_expanding & (d['adx'] > 30) & (d['ema20'] < d['ema50']) & (d['rsi'] > 35) & (d['rsi'] < 60)

    direction = pd.Series('NONE', index=d.index)
    direction[mr_buy | tf_buy]   = 'BUY'
    direction[mr_sell | tf_sell] = 'SELL'

    signal_fires = (direction != 'NONE') & (ml_conf.reindex(d.index) >= threshold)

    closes = d['close'].values
    highs  = d['high'].values
    lows   = d['low'].values
    times  = d.index.to_numpy()
    last_exit = -1

    for i in range(WARMUP_BARS, len(d) - 49):
        if pd.Timestamp(times[i]) < test_start:
            continue
        if i <= last_exit:
            continue
        hour = pd.Timestamp(times[i]).hour
        if not ((7 <= hour < 10) or (13 <= hour < 16)):
            continue
        if not signal_fires.iloc[i]:
            continue

        dir_i  = direction.iloc[i]
        entry  = closes[i] + spread_pip * pip * (1 if dir_i == 'BUY' else -1)
        tp_lvl = entry + tp_pips * pip if dir_i == 'BUY' else entry - tp_pips * pip
        sl_lvl = entry - sl_pips * pip if dir_i == 'BUY' else entry + sl_pips * pip

        pip_val = (pip / entry * 100000) if 'JPY' in pair else 10.0
        lot     = max(round((equity * risk_pct) / (sl_pips * pip_val), 2), 0.01)

        outcome     = 'TIMEOUT'
        close_price = closes[min(i + 48, len(d) - 1)]
        for j in range(i + 1, min(i + 49, len(d))):
            if dir_i == 'BUY':
                if highs[j] >= tp_lvl: outcome = 'TP'; close_price = tp_lvl; break
                if lows[j]  <= sl_lvl: outcome = 'SL'; close_price = sl_lvl; break
            else:
                if lows[j]  <= tp_lvl: outcome = 'TP'; close_price = tp_lvl; break
                if highs[j] >= sl_lvl: outcome = 'SL'; close_price = sl_lvl; break

        pips_gained = (close_price - entry) / pip if dir_i == 'BUY' else (entry - close_price) / pip
        pnl    = pips_gained * lot * pip_val
        equity += pnl
        last_exit = i + 48

        trades.append({
            'time': pd.Timestamp(times[i]), 'pair': pair, 'direction': dir_i,
            'pips': pips_gained, 'pnl': pnl, 'outcome': outcome, 'equity': equity,
        })

    return trades


def fold_metrics(trades, start_equity):
    if not trades:
        return None
    df = pd.DataFrame(trades).sort_values('time')
    wins   = df[df['outcome'] == 'TP']
    total  = len(df)
    wr     = len(wins) / total * 100
    eq     = start_equity
    peak   = eq; max_dd = 0
    for _, r in df.iterrows():
        eq += r['pnl']
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100)
    ret = (eq - start_equity) / start_equity * 100
    avg_win  = wins['pnl'].mean() if len(wins) else 0
    losses   = df[df['outcome'] != 'TP']
    avg_loss = losses['pnl'].mean() if len(losses) else 0
    exp      = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    gw = wins['pnl'].sum(); gl = abs(losses['pnl'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    return {'trades': total, 'wr': wr, 'expectancy': exp, 'return': ret, 'max_dd': max_dd, 'pf': pf}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_days', type=int,   default=270)
    parser.add_argument('--test_days',  type=int,   default=90)
    parser.add_argument('--step_days',  type=int,   default=90)
    parser.add_argument('--sl',         type=int,   default=30)
    parser.add_argument('--tp',         type=int,   default=60)
    parser.add_argument('--threshold',  type=float, default=0.54)
    parser.add_argument('--risk',       type=float, default=0.02)
    parser.add_argument('--equity',     type=float, default=10000)
    args = parser.parse_args()

    total_days = args.train_days + args.test_days
    n_folds    = (729 - total_days) // args.step_days + 1
    print(f"\nWalk-Forward Validation")
    print(f"Train: {args.train_days}d  Test: {args.test_days}d  Step: {args.step_days}d  → {n_folds} folds")
    print(f"SL={args.sl}p  TP={args.tp}p  ML≥{args.threshold}  risk={args.risk:.0%}\n")

    # Load saved feature names from production model
    try:
        payload       = joblib.load('config/ml_model.pkl')
        feature_names = payload['feature_names']
    except Exception:
        print("Warning: could not load config/ml_model.pkl feature names — will infer from data")
        feature_names = None

    end_date   = datetime.utcnow()
    full_start = end_date - timedelta(days=729)

    print("Fetching H1 data and cross-assets...", flush=True)
    cross_df = fetch_cross_assets(full_start, end_date)
    print(f"  Cross-assets: {list(cross_df.columns) if not cross_df.empty else 'none'}")

    pair_data = {}
    for pair, ticker in PAIRS.items():
        df = fetch_h1(ticker, full_start, end_date)
        if df is None or len(df) < total_days * 6:
            print(f"  {pair}: insufficient data — skipped")
            continue
        df = merge_cross(df, cross_df)
        pair_data[pair] = df
        print(f"  {pair}: {len(df)} bars")

    if not pair_data:
        print("No data available."); return

    fold_results = []

    for fold in range(n_folds):
        fold_start  = fold * args.step_days
        train_start = end_date - timedelta(days=729 - fold_start)
        train_end   = train_start + timedelta(days=args.train_days)
        test_start  = train_end
        test_end    = test_start + timedelta(days=args.test_days)

        if test_end > end_date:
            break

        print(f"Fold {fold+1}/{n_folds}: train {train_start.date()}→{train_end.date()} "
              f"| test {test_start.date()}→{test_end.date()}", flush=True)

        # Build training features across all pairs
        train_frames = []
        for pair, df in pair_data.items():
            df_train = df[(df.index >= train_start) & (df.index < train_end)]
            if len(df_train) < 200:
                continue
            feat = make_features(df_train, spread_pips=SPREADS.get(pair, 1.5))
            if feat.empty:
                continue
            feat['pair'] = pair
            train_frames.append(feat)

        if not train_frames:
            print(f"  Fold {fold+1}: no training data — skipped")
            continue

        combined = pd.concat(train_frames).sort_index()
        feat_cols = [c for c in combined.columns if c not in ('target', 'pair')]
        X_train   = combined[feat_cols]
        y_train   = combined['target']

        if feature_names is None:
            feature_names = feat_cols
        else:
            # Align to production feature set
            for c in feature_names:
                if c not in X_train.columns:
                    X_train[c] = 0.0
            X_train = X_train[feature_names]

        # Compute out-of-sample AUC on test features (no trade sim needed for AUC)
        test_frames = []
        for pair, df in pair_data.items():
            df_test_feats = df[(df.index >= test_start) & (df.index < test_end)]
            if len(df_test_feats) < 50:
                continue
            feat = make_features(df_test_feats, spread_pips=SPREADS.get(pair, 1.5))
            if feat.empty or 'target' not in feat.columns:
                continue
            feat['pair'] = pair
            test_frames.append(feat)

        fold_model = train_model(X_train, y_train)
        print(f"  Trained on {len(X_train):,} samples")

        # Out-of-sample AUC
        oos_auc = None
        if test_frames:
            combined_test = pd.concat(test_frames).sort_index()
            X_te = combined_test[feature_names] if all(c in combined_test.columns for c in feature_names) else combined_test[feat_cols]
            y_te = combined_test['target']
            if len(y_te.unique()) == 2:
                proba = fold_model.predict_proba(X_te)[:, 1]
                oos_auc = roc_auc_score(y_te, proba)

        # Trade simulation on test window (include WARMUP_BARS from before test_start)
        all_trades = []
        for pair, df in pair_data.items():
            warmup_start = test_start - timedelta(hours=WARMUP_BARS)
            df_sim = df[(df.index >= warmup_start) & (df.index < test_end)]
            if len(df_sim) < WARMUP_BARS + 10:
                continue
            trades = simulate_trades(pair, df_sim, fold_model, feature_names,
                                     args.sl, args.tp, args.threshold,
                                     args.risk, args.equity, test_start)
            all_trades.extend(trades)

        m = fold_metrics(all_trades, args.equity * len(pair_data))
        if m:
            auc_str = f"  AUC={oos_auc:.3f}" if oos_auc else ""
            pf_str  = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "∞"
            print(f"  {m['trades']} trades  WR {m['wr']:.0f}%  E ${m['expectancy']:+.2f}  "
                  f"Return {m['return']:+.1f}%  MaxDD {m['max_dd']:.1f}%  PF {pf_str}{auc_str}")
            fold_results.append(m)
        else:
            print(f"  No trades in test window")

    # Summary
    print("\n" + "=" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 70)
    if not fold_results:
        print("  No folds produced trades."); return

    metrics = ['wr', 'expectancy', 'return', 'max_dd']
    labels  = ['Win rate (%)', 'Expectancy ($)', 'Return (%)', 'Max DD (%)']
    for key, label in zip(metrics, labels):
        vals = [f[key] for f in fold_results]
        print(f"  {label:<20} mean={np.mean(vals):+.1f}  "
              f"std={np.std(vals):.1f}  "
              f"min={np.min(vals):+.1f}  max={np.max(vals):+.1f}")

    consistent = sum(1 for f in fold_results if f['return'] > 0)
    print(f"\n  Profitable folds: {consistent}/{len(fold_results)}")
    if consistent == len(fold_results):
        print("  ✓ Edge holds across all folds")
    elif consistent >= len(fold_results) * 0.75:
        print("  ~ Edge mostly consistent — monitor underperforming folds")
    else:
        print("  ✗ Inconsistent across folds — possible curve-fitting or regime sensitivity")

    # Degradation check: does performance drop from early folds to late folds?
    if len(fold_results) >= 4:
        first_half = np.mean([f['return'] for f in fold_results[:len(fold_results)//2]])
        second_half = np.mean([f['return'] for f in fold_results[len(fold_results)//2:]])
        diff = second_half - first_half
        print(f"\n  First-half avg return: {first_half:+.1f}%")
        print(f"  Second-half avg return: {second_half:+.1f}%")
        if abs(diff) < 10:
            print("  ✓ No significant degradation across time")
        elif diff < 0:
            print(f"  ⚠ Performance dropped {abs(diff):.1f}% in later folds — possible regime shift")
        else:
            print(f"  ↑ Performance improved {diff:.1f}% in later folds")
    print()


if __name__ == '__main__':
    main()
