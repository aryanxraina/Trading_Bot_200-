"""
train_meta_learner.py
Generates vote-level training data from your 47 stocks, then trains
the meta-learner (logistic regression over vote features).

What this does:
  1. For each stock and each day in the backtest window:
     - Run all 9 voters to get their signals
     - Record the actual next-day return (ground truth)
  2. Build a training matrix: [33 vote features] -> [1 = UP, 0 = DOWN]
  3. Train/val split (70/30 by time, NOT random — avoids lookahead)
  4. Train logistic regression
  5. Print learned weights (which voters matter most)
  6. Save model to models/saved/

Runtime: ~5-10 minutes for 47 stocks x 1000 days each.
         Most time is XGBoost/LSTM inference.

Usage:
    python -m scripts.train_meta_learner
    python -m scripts.train_meta_learner --stocks RELIANCE INFY TCS    # subset
    python -m scripts.train_meta_learner --start 2023-01-01             # date range
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.preprocess import preprocess_symbol, get_feature_columns
from src.models.ensemble import EnsembleModel, CONFIDENCE_THRESHOLD
from src.models.vote_extractor import signal_to_features, get_feature_names, VOTER_ORDER
from src.models.meta_learner import MetaLearner
from src.utils.logger import get_logger

logger = get_logger("train_meta_learner")


def get_stock_list(custom_stocks: list[str] = None) -> list[str]:
    """Get list of stocks to process."""
    if custom_stocks:
        return custom_stocks
    # Read from nifty50 symbols file or use processed files
    symbols_file = Path("config/nifty50_symbols.txt")
    if symbols_file.exists():
        with symbols_file.open() as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    # Fallback: scan processed directory
    processed = Path("data/processed")
    if processed.exists():
        return sorted([f.stem.replace("_features", "")
                      for f in processed.glob("*_features.csv")])
    return []


def generate_vote_features(
    ensemble: EnsembleModel,
    symbol: str,
    start_date: str = "2023-01-01",
    lookback: int = 200,
) -> pd.DataFrame:
    """
    For a single stock, walk through each day and extract vote features.

    Returns DataFrame with columns: [33 vote features] + date + symbol + actual_return + signal
    """
    try:
        df = preprocess_symbol(symbol, save=False)
    except Exception as e:
        logger.warning(f"{symbol}: preprocess failed: {e}")
        return pd.DataFrame()

    if len(df) < lookback + 100:
        logger.warning(f"{symbol}: too few rows ({len(df)}), skipping")
        return pd.DataFrame()

    # Find start index
    start_idx = 0
    if start_date:
        mask = df.index >= start_date
        if mask.any():
            start_idx = mask.argmax()
    start_idx = max(start_idx, lookback)

    rows = []
    feature_names = get_feature_names()

    for i in range(start_idx, len(df) - 1):  # -1 because we need next day return
        window = df.iloc[:i + 1]  # all data up to this point

        # Get ensemble signals (all 9 voters)
        try:
            signal = ensemble.predict(window, symbol)
        except Exception:
            continue

        # Extract vote features
        regime = ensemble._get_regime(window)
        features = signal_to_features(signal.model_signals, regime)

        # Ground truth: next day return
        next_return = float(df.iloc[i + 1].get("daily_return", 0.0))
        if "next_day_return" in df.columns:
            next_return = float(df.iloc[i]["next_day_return"])

        actual_up = 1 if next_return > 0 else 0

        row = {
            "date": df.index[i],
            "symbol": symbol,
            "actual_return": next_return,
            "actual_signal": actual_up,
            **features,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        logger.info(f"{symbol}: {len(result)} vote feature rows generated")
    return result


def main(
    stocks: list[str] = None,
    start_date: str = "2023-01-01",
    threshold: float = 0.55,
    top_n: int = 10,
):
    print("=" * 60)
    print("  META-LEARNER TRAINING")
    print("=" * 60)

    # Get stock list
    stock_list = get_stock_list(stocks)
    if not stock_list:
        print("No stocks found. Check config/nifty50_symbols.txt or data/processed/")
        return

    if top_n and len(stock_list) > top_n:
        # Use top N most liquid stocks for faster training
        stock_list = stock_list[:top_n]

    print(f"\nStocks: {len(stock_list)}")
    print(f"Start date: {start_date}")
    print(f"Threshold: {threshold}")

    # Load ensemble (loads XGBoost + LSTM once)
    print("\n[1/4] Loading ensemble models...")
    ensemble = EnsembleModel(min_votes=1, backtest_mode=True)  # min_votes=1 so all signals fire
    ensemble.load_models()

    # Generate vote features for each stock
    print(f"\n[2/4] Generating vote features for {len(stock_list)} stocks...")
    all_data = []
    for i, symbol in enumerate(stock_list):
        print(f"  [{i+1}/{len(stock_list)}] {symbol}...", end=" ", flush=True)
        df = generate_vote_features(ensemble, symbol, start_date)
        if not df.empty:
            all_data.append(df)
            print(f"{len(df)} rows")
        else:
            print("skipped")

    if not all_data:
        print("No training data generated. Check your data/processed/ folder.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values("date").reset_index(drop=True)

    print(f"\nTotal training data: {len(full_df)} rows across {full_df['symbol'].nunique()} stocks")
    print(f"Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    print(f"Class balance: {full_df['actual_signal'].mean():.1%} UP")

    # Save training data for inspection
    data_dir = Path("data/meta_learner")
    data_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(data_dir / "vote_features.csv", index=False)
    print(f"Training data saved to {data_dir / 'vote_features.csv'}")

    # Train/val split by TIME (not random!)
    print(f"\n[3/4] Training meta-learner...")
    feature_cols = get_feature_names()
    split_idx = int(len(full_df) * 0.7)

    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["actual_signal"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["actual_signal"].values

    print(f"  Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")

    # Train
    ml = MetaLearner(threshold=threshold)
    metrics = ml.train(X_train, y_train, X_val, y_val)

    print(f"\n[4/4] Results:")
    print(f"  Train accuracy:       {metrics['train_accuracy']:.1%}")
    if metrics.get("val_accuracy_all") is not None:
        print(f"  Val accuracy (all):   {metrics['val_accuracy_all']:.1%}")
    if metrics.get("val_accuracy_fired") is not None:
        print(f"  Val accuracy (fired): {metrics['val_accuracy_fired']:.1%}")
        print(f"  Val fire rate:        {metrics['val_fire_rate']:.1%}")
        print(f"  Val fired count:      {metrics['val_fired_count']}")

    # Show weights
    ml.print_weights()

    # Compare vs old system
    print("\n=== Comparison: Old vs Meta-Learner ===")
    # Old system: simple vote counting with min_votes=3
    old_preds = []
    for _, row in val_df.iterrows():
        up = int(row["votes_up"])
        down = int(row["votes_down"])
        if up >= 3:
            old_preds.append(1)
        elif down >= 3:
            old_preds.append(0)
        else:
            old_preds.append(-1)  # NO_TRADE

    old_preds = np.array(old_preds)
    old_fires = old_preds != -1
    if old_fires.sum() > 0:
        old_acc = (old_preds[old_fires] == y_val[old_fires]).mean()
        print(f"  Old system (3-vote count):")
        print(f"    Accuracy on trades:  {old_acc:.1%}")
        print(f"    Fire rate:           {old_fires.mean():.1%}")
        print(f"    Trade count:         {old_fires.sum()}")
    else:
        print(f"  Old system: no trades fired")

    meta_preds = ml.predict_signal(X_val)
    meta_fires = np.array([1 if p != "NO_TRADE" else 0 for p in meta_preds])
    if meta_fires.sum() > 0:
        meta_pred_binary = np.array([1 if p == "UP" else 0 for p in meta_preds])
        meta_acc = (meta_pred_binary[meta_fires == 1] == y_val[meta_fires == 1]).mean()
        print(f"\n  Meta-learner (learned weights):")
        print(f"    Accuracy on trades:  {meta_acc:.1%}")
        print(f"    Fire rate:           {meta_fires.mean():.1%}")
        print(f"    Trade count:         {meta_fires.sum()}")

        improvement = meta_acc - old_acc if old_fires.sum() > 0 else 0
        print(f"\n  Improvement: {improvement:+.1%}")
    else:
        print(f"\n  Meta-learner: no trades fired (lower threshold needed)")

    # Save
    ml.save()
    print(f"\nModel saved to models/saved/meta_learner.pkl")
    print(f"Weights saved to models/saved/meta_learner_weights.json")
    print("\nDone. The meta-learner is ready to replace the vote-counting logic.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stocks", nargs="+", default=None,
                   help="List of stock symbols to train on (default: top 10)")
    p.add_argument("--start", default="2023-01-01",
                   help="Backtest start date (default: 2023-01-01)")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Confidence threshold for trading (default: 0.55)")
    p.add_argument("--all-stocks", action="store_true",
                   help="Use all 47 stocks (slower but better)")
    args = p.parse_args()
    top_n = None if args.all_stocks else 10
    main(stocks=args.stocks, start_date=args.start,
         threshold=args.threshold, top_n=top_n)
