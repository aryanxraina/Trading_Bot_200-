# Sentiment Agent — Setup Guide

Drop-in 8th vote for your existing 7-vote ensemble. Adds non-price information
(news sentiment) so XGBoost stops being blind to the world outside OHLCV.

## What's in the box

```
src/sentiment/
  news_scraper.py          ← RSS scraper (Moneycontrol, ET, Mint, BS) — no API keys
  sentiment_model.py       ← FinBERT wrapper + cache
  sentiment_aggregator.py  ← Per-stock daily score with time decay
  sentiment_signal.py      ← Vote function (drop-in to ensemble.py)
  ensemble_patch.py        ← Step-by-step diff to apply to ensemble.py
scripts/
  run_daily_sentiment.py   ← End-to-end pipeline (cron this)
tests/
  test_sentiment.py        ← 22 unit tests, runs in 1 second
```

## Step 1 — Copy files into your project

Drop the `src/sentiment/` folder into your existing `src/`. Drop `scripts/` and
`tests/` at the repo root. Your tree becomes:

```
ai-trading-bot/
  src/
    sentiment/        ← NEW
      __init__.py
      news_scraper.py
      sentiment_model.py
      sentiment_aggregator.py
      sentiment_signal.py
      ensemble_patch.py
    data/             ← existing
    models/           ← existing
    ...
  scripts/
    run_daily_sentiment.py
  tests/
    test_sentiment.py
  data/
    sentiment/        ← created automatically on first run
```

## Step 2 — Install dependencies

```bash
conda activate trading
pip install transformers torch feedparser python-dateutil
```

You already have `torch` from the LSTM build, so this mostly adds `transformers`
(~50MB) and `feedparser` (tiny).

First run of FinBERT downloads `ProsusAI/finbert` (~440MB) to your HuggingFace
cache. One-time. After that it's local.

## Step 3 — Run unit tests (no model download needed)

```bash
python -m tests.test_sentiment
```

Should print 22 ✅ checks in under a second. If they all pass, the math is
sound — the only remaining concern is whether the live RSS feeds work for you
(some block Indian IPs occasionally; if so, swap them in `news_scraper.FEEDS`).

## Step 4 — Smoke test the scraper

```bash
python -m src.sentiment.news_scraper
```

Should print ~50-200 headlines from 5 RSS sources, with detected stock symbols
shown for the first 5. If you get 0 items, your network is blocking RSS — try a
VPN or swap the feed URLs.

## Step 5 — First sentiment build

```bash
python -m scripts.run_daily_sentiment
```

This will:
1. Fetch RSS feeds → `data/sentiment/news_raw.jsonl`
2. Download FinBERT (one-time, ~440MB)
3. Score every Nifty-stock headline → `data/sentiment/news_scored.jsonl`
4. Aggregate per (date, symbol) → `data/sentiment/sentiment_daily.csv`
5. Merge into `data/processed/<SYMBOL>_features_with_sent.csv`

Expect 1-3 minutes on first run. Subsequent runs hit the cache and finish in
seconds.

## Step 6 — Patch ensemble.py

Open `src/sentiment/ensemble_patch.py` — it's a documentation file with the
exact 4 changes to make in your `src/models/ensemble.py`. The patch is small:
1 import line, 1 dict entry, 1 constant bump (optional), 2 lines in summary().

## Step 7 — Backtest with the new vote

In your notebook 05, change the data load to use the `*_features_with_sent.csv`
files instead of `*_features.csv`. Then re-run the same backtest.

What to look for:
- **Trade count should DROP** (sentiment abstains often → fewer noisy trades)
- **Win rate should rise 1-3pp** if sentiment is helping
- **Sharpe should rise** even if total return doesn't
- If trade count drops but win rate doesn't change → sentiment is mostly noise
  and not adding edge yet. Get more news sources before giving up.

## Step 8 — Production schedule

Add two cron jobs (or Windows Task Scheduler):

```cron
# Pre-market sentiment refresh — 8:00 AM IST every weekday
0 8 * * 1-5 cd /path/to/ai-trading-bot && python -m scripts.run_daily_sentiment

# Post-close refresh for next-day signal — 6:00 PM IST every weekday
0 18 * * 1-5 cd /path/to/ai-trading-bot && python -m scripts.run_daily_sentiment
```

## Honest limitations

1. **No historical backfill.** RSS feeds only return the last few days. So
   your first backtest will only have sentiment for ~the last week of data.
   To backfill years, you need either a paid news API (NewsAPI, Tiingo) or
   to scrape archives (slow, fragile, possibly ToS-violating).

2. **Sparse coverage on smaller Nifty 50 stocks.** Reliance/Infosys/HDFC get
   10+ headlines a day. UPL/GAIL/ZEEL might get 0-1 a day. The `reliable` flag
   correctly handles this — those stocks just won't get a sentiment vote.

3. **FinBERT is English-only.** Hindi business news is missed. Acceptable for
   Nifty 50 since major coverage is in English.

4. **No sarcasm / nuanced events handling.** "Adani denies allegations" is
   classified positive by FinBERT (because "denies" is positive sentiment in
   isolation). For these edge cases, the LLM-based Research Agent (next build)
   is the right fix, not a better classifier.

## How this connects to the master plan

This is **Agent 4 (Sentiment Agent)** from the multi-agent architecture I laid
out. Once shipped, the next pieces in priority order:

1. **Research Agent** (LLM morning briefing) — 2-day build
2. **Refined Signal Agent** (meta-learner over the 8 votes) — 3-day build
3. **Crypto fork** (clone + Binance feed) — 1-week build
4. **Vision Agent** (CNN on chart images) — 2-week build
5. **Execution Agent** (RL policy) — 3-week build
