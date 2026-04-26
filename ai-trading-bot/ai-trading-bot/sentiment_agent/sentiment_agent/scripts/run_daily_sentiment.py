"""
run_daily_sentiment.py
End-to-end daily pipeline:
  1. Fetch all RSS feeds
  2. Score each headline through FinBERT
  3. Aggregate per-(stock, date) with time decay
  4. Merge into per-stock processed feature CSVs
  5. Save to data/sentiment/sentiment_daily.csv

Run this:
  - Once at start to backfill (requires historical news, which RSS feeds don't
    give you — so backfill will only have ~1-2 weeks).
  - Daily at 8:00 AM IST as a cron job before market open.
  - Optionally again at 6:00 PM IST to capture intraday news for next-day signal.

Usage:
    python -m scripts.run_daily_sentiment
    python -m scripts.run_daily_sentiment --skip-fetch  # rescore cached news only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.sentiment.news_scraper import fetch_all, save_jsonl, NewsItem
from src.sentiment.sentiment_model import SentimentModel
from src.sentiment.sentiment_aggregator import aggregate, to_dataframe, merge_into_features
from src.utils.logger import get_logger

logger = get_logger("run_daily_sentiment")

DATA_DIR = Path("data/sentiment")
RAW_NEWS_FILE = DATA_DIR / "news_raw.jsonl"
SCORED_NEWS_FILE = DATA_DIR / "news_scored.jsonl"
DAILY_SCORES_FILE = DATA_DIR / "sentiment_daily.csv"

PROCESSED_DIR = Path("data/processed")


def step_fetch() -> int:
    """Fetch RSS feeds, append to raw JSONL. Returns count of new items."""
    items = fetch_all()
    return save_jsonl(items, RAW_NEWS_FILE)


def step_score(rescore_all: bool = False) -> pd.DataFrame:
    """
    Run FinBERT on every raw item and write scored JSONL.
    With caching, re-running this is cheap (only new items hit the model).
    """
    if not RAW_NEWS_FILE.exists():
        logger.warning(f"{RAW_NEWS_FILE} missing — run with --fetch first")
        return pd.DataFrame()

    raw_items: list[dict] = []
    with RAW_NEWS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                raw_items.append(json.loads(line))
            except ValueError:
                continue

    # Skip items with no detected stocks — saves model time.
    stocked = [it for it in raw_items if it.get("stocks")]
    logger.info(f"Loaded {len(raw_items)} raw items | {len(stocked)} mention Nifty stocks")

    sm = SentimentModel()
    sm.load()

    # Score titles (and summaries if you want richer signal — for now title only,
    # since summary is often duplicate of body and adds latency)
    titles = [it["title"] for it in stocked]
    preds = sm.predict_batch(titles, batch_size=16)

    scored: list[dict] = []
    for it, pred in zip(stocked, preds):
        scored.append({**it, **pred})

    # Write scored JSONL fresh (overwrite — it's deterministic from raw + model)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SCORED_NEWS_FILE.open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(scored)} scored items → {SCORED_NEWS_FILE}")

    return pd.DataFrame(scored)


def step_aggregate(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored items into per-(date, symbol) sentiment scores."""
    if scored_df.empty:
        return pd.DataFrame()
    rows = aggregate(scored_df.to_dict("records"))
    df = to_dataframe(rows)
    df.to_csv(DAILY_SCORES_FILE, index=False)
    logger.info(f"Wrote {len(df)} (date, symbol) rows → {DAILY_SCORES_FILE}")
    if not df.empty:
        logger.info(f"  Stocks covered: {df['symbol'].nunique()}")
        logger.info(f"  Date range:    {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def step_merge(sentiment_df: pd.DataFrame) -> int:
    """For each processed feature CSV, append sentiment columns."""
    if not PROCESSED_DIR.exists():
        logger.warning(f"{PROCESSED_DIR} not found — skipping merge step")
        return 0

    files = sorted(PROCESSED_DIR.glob("*_features.csv"))
    merged = 0
    for fp in files:
        symbol = fp.stem.replace("_features", "")
        df = pd.read_csv(fp, parse_dates=["date"] if "date" in pd.read_csv(fp, nrows=0).columns else [0])
        # Make sure index is the date column
        if "date" in df.columns:
            df = df.set_index("date")
        merged_df = merge_into_features(df, sentiment_df, symbol)
        # Save with sentiment columns alongside (don't overwrite original by default)
        out_path = fp.with_name(f"{symbol}_features_with_sent.csv")
        merged_df.to_csv(out_path)
        merged += 1
    logger.info(f"Merged sentiment into {merged} stock files → *_features_with_sent.csv")
    return merged


def main(skip_fetch: bool = False, skip_merge: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_fetch:
        n = step_fetch()
        logger.info(f"FETCH: {n} new headlines")
    else:
        logger.info("FETCH: skipped")

    scored = step_score()
    if scored.empty:
        logger.warning("No scored items — exiting")
        return

    daily = step_aggregate(scored)

    if not skip_merge and not daily.empty:
        step_merge(daily)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-fetch", action="store_true",
                   help="Don't fetch new RSS items, just rescore + aggregate cached.")
    p.add_argument("--skip-merge", action="store_true",
                   help="Don't merge into processed feature CSVs.")
    args = p.parse_args()
    main(skip_fetch=args.skip_fetch, skip_merge=args.skip_merge)
