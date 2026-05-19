#!/usr/bin/env python3
"""
Train ML model on all 5 forex pairs using H1 bars + daily cross-asset features.
Yahoo Finance caps H1 history at ~2 years, so we use 2 years of H1 data.
Cross-assets (DXY, Gold, TNX, VIX) are fetched at daily resolution and
forward-filled onto each H1 bar by date.

Run: python3 -m src.train_model
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

from src.data_fetcher import ForexDataFetcher
from src.ml_model import MLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_cross_assets(df, cross_assets):
    """
    Merge daily cross-asset closes onto an H1 OHLCV df.
    Matches on calendar date (ignores intraday time component).
    """
    merged = df.copy()
    merged['_date_key'] = pd.to_datetime(merged['date']).dt.normalize()

    for name, cdf in cross_assets.items():
        col = f'{name}_close'
        cdf2 = cdf.copy()
        cdf2['_date_key'] = pd.to_datetime(cdf2['date']).dt.normalize()
        cdf2 = cdf2[['_date_key', 'close']].rename(columns={'close': col})
        merged = merged.merge(cdf2, on='_date_key', how='left')
        merged[col] = merged[col].ffill()

    merged.drop(columns=['_date_key'], inplace=True)
    return merged


def main():
    end = datetime.now()
    start = end - timedelta(days=365 * 4)

    logger.info("Fetching 4 years of daily data for all 5 pairs...")
    all_data = ForexDataFetcher.fetch_all_pairs(start, end, interval='1d')

    logger.info("Fetching daily cross-asset data (DXY, Gold, TNX, VIX)...")
    cross_assets = ForexDataFetcher.fetch_cross_assets(start, end, interval='1d')

    if not all_data:
        logger.error("No forex data fetched — aborting")
        return

    logger.info(f"Cross-assets fetched: {list(cross_assets.keys())}")

    model = MLModel(model_path='config/ml_model.pkl')
    feature_frames = []

    for pair, df in all_data.items():
        merged = merge_cross_assets(df, cross_assets)
        feat = model.create_features(merged)
        logger.info(f"  {pair}: {len(df)} H1 bars → {len(feat)} feature rows")
        feature_frames.append(feat)

    if not feature_frames:
        logger.error("No feature rows generated — aborting")
        return

    all_features = pd.concat(feature_frames, ignore_index=True)
    logger.info(f"Combined: {len(all_features)} feature rows, {len(all_features.columns)-1} features")

    all_features = all_features.sample(frac=1, random_state=42).reset_index(drop=True)

    X = all_features.drop('target', axis=1)
    y = all_features['target']

    logger.info(f"Class balance — up: {y.mean():.1%}  down: {(1-y).mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
    )

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba)

    logger.info(f"\nTest accuracy : {accuracy:.2%}")
    logger.info(f"AUC           : {auc:.2%}")
    logger.info(f"Prob range    : {y_proba.min():.3f} – {y_proba.max():.3f}  (mean {y_proba.mean():.3f})")
    logger.info(f"Above 0.53    : {(y_proba > 0.53).sum()} / {len(y_proba)}  ({(y_proba > 0.53).mean():.1%})")
    logger.info(f"Above 0.60    : {(y_proba > 0.60).sum()} / {len(y_proba)}  ({(y_proba > 0.60).mean():.1%})")

    feature_names = X.columns.tolist()
    joblib.dump({'model': clf, 'feature_names': feature_names}, 'config/ml_model.pkl')
    logger.info("Model saved to config/ml_model.pkl")


if __name__ == '__main__':
    main()
