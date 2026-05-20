#!/usr/bin/env python3
"""
Retrain the ML model on 2 years of H1 data across all 5 pairs.
Uses walk-forward cross-validation and proper cross-asset feature alignment.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
import joblib
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.ml_model import make_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PAIRS = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CHF': 'USDCHF=X',
}

CROSS_ASSETS = {
    'dxy':  'DX-Y.NYB',
    'gold': 'GC=F',
    'tnx':  '^TNX',
    'vix':  '^VIX',
}

SPREAD_PIPS = {
    'EUR/USD': 1.0,
    'GBP/USD': 1.5,
    'USD/JPY': 1.2,
    'AUD/USD': 1.5,
    'USD/CHF': 1.8,
}


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_forex(pair, ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval='1h', progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[['open','high','low','close','volume']].dropna(subset=['close'])
    logger.info(f"{pair}: {len(df)} H1 bars")
    return df


def fetch_cross_assets(start, end):
    result = {}
    for name, ticker in CROSS_ASSETS.items():
        df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].rename(columns={'Close': name})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        result[name] = df
        logger.info(f"{name}: {len(df)} daily bars")
    return result


# make_features is imported from src.ml_model — single source of truth

# ── Walk-forward validation ──────────────────────────────────────────────────

def walk_forward_score(X, y, n_splits=5):
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    aucs   = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pos_weight = (1 - y_tr.mean()) / y_tr.mean() if y_tr.mean() > 0 else 1.0
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
            num_leaves=25,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=80,
            reg_alpha=0.2,
            reg_lambda=0.2,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(period=-1)])
        preds      = model.predict(X_te)
        proba      = model.predict_proba(X_te)[:, 1]
        acc        = accuracy_score(y_te, preds)
        auc        = roc_auc_score(y_te, proba)
        scores.append(acc)
        aucs.append(auc)
        logger.info(f"  Fold {fold+1}: accuracy={acc:.2%}  AUC={auc:.3f}  "
                    f"(train={len(X_tr)}, test={len(X_te)})")
    return np.mean(scores), np.mean(aucs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    end   = datetime.utcnow()
    start = end - timedelta(days=729)  # Yahoo Finance H1 limit ~2 years

    logger.info("=" * 60)
    logger.info("FOREX ML MODEL RETRAINING")
    logger.info(f"Period: {start.date()} → {end.date()}")
    logger.info("=" * 60)

    # 1. Fetch cross-asset data (daily — will be forward-filled to H1)
    logger.info("\n[1/4] Fetching cross-asset data...")
    cross = fetch_cross_assets(start - timedelta(days=10), end)

    # Build a combined cross-asset daily DataFrame
    cross_df = pd.DataFrame()
    for name, df in cross.items():
        col = f'{name}_close'
        df.columns = [col]
        cross_df = cross_df.join(df, how='outer') if not cross_df.empty else df.copy()
        cross_df.columns = list(cross_df.columns[:-1]) + [col] if len(cross_df.columns) > 1 else [col]

    # Re-build properly
    cross_df = pd.DataFrame(index=pd.date_range(start - timedelta(days=10), end, freq='D'))
    for name, df in cross.items():
        cross_df[f'{name}_close'] = df.iloc[:, 0].reindex(cross_df.index, method='ffill')

    # 2. Fetch forex pairs
    logger.info("\n[2/4] Fetching H1 forex data...")
    all_features = []
    for pair, ticker in PAIRS.items():
        df = fetch_forex(pair, ticker, start, end)
        if df is None:
            logger.warning(f"Skipping {pair} — no data")
            continue

        # Merge cross-asset data by date (forward-fill daily into hourly)
        df = df.join(cross_df.reindex(df.index, method='ffill'), how='left')

        # Build features
        spread = SPREAD_PIPS.get(pair, 1.5)
        feat   = make_features(df, spread_pips=spread)
        feat['pair'] = pair
        all_features.append(feat)
        logger.info(f"{pair}: {len(feat)} usable feature rows")

    if not all_features:
        logger.error("No data fetched — aborting")
        return

    # 3. Combine, walk-forward validate
    logger.info("\n[3/4] Walk-forward cross-validation...")
    combined = pd.concat(all_features).sort_index()

    feature_cols = [c for c in combined.columns if c not in ('target', 'pair')]
    X = combined[feature_cols]
    y = combined['target']

    logger.info(f"Total samples: {len(X)}  |  Features: {len(feature_cols)}")
    logger.info(f"Class balance: {y.mean():.1%} BUY opportunities")

    mean_acc, mean_auc = walk_forward_score(X, y, n_splits=5)
    logger.info(f"\nWalk-forward results: Accuracy={mean_acc:.2%}  AUC={mean_auc:.3f}")

    # 4. Final model — train on all data
    logger.info("\n[4/4] Training final model on full dataset...")
    pos_weight_final = (1 - y.mean()) / y.mean() if y.mean() > 0 else 1.0
    logger.info(f"Class balance: {y.mean():.1%} wins → scale_pos_weight={pos_weight_final:.2f}")
    final_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=25,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=80,
        reg_alpha=0.2,
        reg_lambda=0.2,
        scale_pos_weight=pos_weight_final,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    # Feature importance
    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)
    logger.info("\nTop 10 most important features:")
    for feat_name, score in top10.items():
        logger.info(f"  {feat_name:<25} {score:.0f}")

    # Save model
    payload = {
        'model':         final_model,
        'feature_names': feature_cols,
        'trained_at':    datetime.utcnow().isoformat(),
        'n_samples':     len(X),
        'walk_forward':  {'accuracy': mean_acc, 'auc': mean_auc},
    }
    joblib.dump(payload, 'config/ml_model.pkl')
    logger.info("\nModel saved to config/ml_model.pkl")

    # Recommend confidence threshold based on AUC
    if mean_auc >= 0.58:
        threshold = 0.58
    elif mean_auc >= 0.54:
        threshold = 0.55
    else:
        threshold = 0.53
        logger.warning("AUC below 0.54 — model has weak edge. Consider more features or longer history.")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Samples trained on : {len(X):,}")
    logger.info(f"  Walk-forward accuracy: {mean_acc:.2%}")
    logger.info(f"  Walk-forward AUC     : {mean_auc:.3f}")
    logger.info(f"  Recommended threshold: {threshold}")
    logger.info(f"  Update src/main.py → SignalEngine(ml_confidence_threshold={threshold})")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
