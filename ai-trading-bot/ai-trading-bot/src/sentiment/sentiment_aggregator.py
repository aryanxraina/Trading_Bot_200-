"""
sentiment_aggregator.py
Turns raw scored headlines into per-stock daily sentiment scores.

Inputs:
  A list of dicts (one per scored headline):
    {
      "stocks":    ["INFY", "TCS"],
      "label":     "positive" | "negative" | "neutral",
      "score":     float,                 # FinBERT confidence
      "probs":     {"positive":.., "negative":.., "neutral":..},
      "published": ISO 8601 string,
      "source":    "moneycontrol" | ...,
    }

Outputs:
  Per (date, symbol):
    {
      "symbol":          "INFY",
      "date":            "2026-04-26",
      "sent_score":      float in [-1, +1],   # signed, decay-weighted
      "sent_strength":   float in [0, 1],     # |score| but capped
      "sent_count":      int,                 # raw headline count for the day
      "sent_pos_share":  float,               # fraction of pos headlines
      "sent_neg_share":  float,
      "sent_top_source": "moneycontrol" | ..., # most reputable source seen
    }

Key design choices:
  1. SIGNED SCORE: pos contributes +prob, neg contributes -prob, neutral = 0.
     This means an article with prob {pos:0.7, neg:0.1, neu:0.2} contributes
     +0.6, NOT just +1. Confidence-weighted naturally.

  2. TIME DECAY: an article from 12 hours ago weighs ~0.7 of one from now.
     Half-life is configurable. This makes "yesterday's news" naturally fade.

  3. SOURCE WEIGHTING: a Moneycontrol/ET headline weighs more than a generic
     blog. Tunable per-source. This is a defense against low-quality scrapers
     polluting the score.

  4. NEUTRAL FLOOR: if there are <2 headlines for a stock on a given day,
     we mark it as data-poor and the ensemble vote becomes "neutral / abstain"
     rather than firing on a single noisy headline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("sentiment_aggregator")

IST = timezone(timedelta(hours=5, minutes=30))

# Source reputation weights (0-1). Tweak after a few weeks of live data.
SOURCE_WEIGHTS: dict[str, float] = {
    "moneycontrol": 1.00,
    "et_markets":   1.00,
    "et_stocks":    1.00,
    "livemint":     0.90,
    "bs_markets":   0.90,
    "default":      0.70,
}

# Time-decay half-life in hours.
# Half-life of 24h means a headline from yesterday counts 50% of one from now.
DEFAULT_HALF_LIFE_HOURS = 24.0

# If fewer than this many headlines exist for a (stock, day), the score is unreliable.
MIN_HEADLINES_FOR_SIGNAL = 2


@dataclass
class StockDaySentiment:
    symbol: str
    date: str            # YYYY-MM-DD (IST)
    sent_score: float    # [-1, +1]
    sent_strength: float # [0, 1]
    sent_count: int
    sent_pos_share: float
    sent_neg_share: float
    sent_top_source: str
    reliable: bool       # True iff sent_count >= MIN_HEADLINES_FOR_SIGNAL


def _signed_contribution(probs: dict) -> float:
    """Convert FinBERT probs to a signed scalar in [-1, +1]."""
    pos = probs.get("positive", 0.0)
    neg = probs.get("negative", 0.0)
    return pos - neg  # neutral cancels itself out


def _decay_weight(published_iso: str, ref_time: datetime, half_life_hours: float) -> float:
    """Exponential decay. Returns weight in (0, 1]."""
    try:
        from dateutil import parser as dtparser
        dt = dtparser.parse(published_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
    except (ValueError, TypeError):
        return 0.5  # if we can't parse it, give it middling weight
    age_hours = max(0.0, (ref_time - dt).total_seconds() / 3600.0)
    # weight = 0.5 ^ (age / half_life)
    return math.pow(0.5, age_hours / half_life_hours)


def aggregate(
    scored_items: Iterable[dict],
    ref_date: str | None = None,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
) -> list[StockDaySentiment]:
    """
    Aggregate scored headlines into per-(stock, day) sentiment rows.

    Args:
        scored_items:    iterable of dicts with keys: stocks, probs, published, source
        ref_date:        IST date string YYYY-MM-DD; if None, uses today (IST).
                         Decay is computed from end-of-day on this date.
        half_life_hours: half-life of the decay curve, in hours.

    Returns:
        list of StockDaySentiment, one per (date, symbol).
    """
    if ref_date is None:
        ref_date = datetime.now(IST).strftime("%Y-%m-%d")
    ref_time = datetime.strptime(ref_date, "%Y-%m-%d").replace(
        hour=23, minute=59, tzinfo=IST
    )

    # group: (date, symbol) -> list of contributions
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for item in scored_items:
        stocks = item.get("stocks") or []
        if not stocks:
            continue
        published = item.get("published", "")
        # IST date of the headline
        try:
            from dateutil import parser as dtparser
            dt = dtparser.parse(published).astimezone(IST)
            day = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            day = ref_date

        # Only aggregate items published on/before the ref_date
        if day > ref_date:
            continue

        decay = _decay_weight(published, ref_time, half_life_hours)
        source_w = SOURCE_WEIGHTS.get(item.get("source", "default"),
                                      SOURCE_WEIGHTS["default"])
        signed = _signed_contribution(item.get("probs", {}))

        for symbol in stocks:
            buckets[(day, symbol)].append({
                "signed":  signed,
                "weight":  decay * source_w,
                "label":   item.get("label", "neutral"),
                "source":  item.get("source", "default"),
                "source_w": source_w,
            })

    out: list[StockDaySentiment] = []
    for (day, symbol), contribs in buckets.items():
        total_w = sum(c["weight"] for c in contribs)
        if total_w == 0:
            continue
        weighted_score = sum(c["signed"] * c["weight"] for c in contribs) / total_w
        # Clip to [-1, 1] just in case
        weighted_score = max(-1.0, min(1.0, weighted_score))

        n = len(contribs)
        n_pos = sum(1 for c in contribs if c["label"] == "positive")
        n_neg = sum(1 for c in contribs if c["label"] == "negative")

        top_source = max(contribs, key=lambda c: c["source_w"])["source"]

        out.append(StockDaySentiment(
            symbol=symbol,
            date=day,
            sent_score=round(weighted_score, 4),
            sent_strength=round(abs(weighted_score), 4),
            sent_count=n,
            sent_pos_share=round(n_pos / n, 3),
            sent_neg_share=round(n_neg / n, 3),
            sent_top_source=top_source,
            reliable=(n >= MIN_HEADLINES_FOR_SIGNAL),
        ))
    return out


def to_dataframe(rows: list[StockDaySentiment]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "date", "symbol", "sent_score", "sent_strength", "sent_count",
            "sent_pos_share", "sent_neg_share", "sent_top_source", "reliable"
        ])
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def merge_into_features(
    features_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Merge sentiment columns into a per-stock feature DataFrame.

    features_df: indexed by date (your existing preprocess_symbol output)
    sentiment_df: long-form (date, symbol, sent_*) — output of to_dataframe()

    Missing days fill with neutral defaults so the column is never NaN.
    """
    df = features_df.copy()
    if df.index.name is None:
        df.index.name = "date"
    s = sentiment_df[sentiment_df["symbol"] == symbol].copy()
    if s.empty:
        # No sentiment data — fill with neutrals
        df["sent_score"] = 0.0
        df["sent_strength"] = 0.0
        df["sent_count"] = 0
        df["sent_reliable"] = False
        return df

    s = s.set_index("date")[
        ["sent_score", "sent_strength", "sent_count", "reliable"]
    ].rename(columns={"reliable": "sent_reliable"})

    df = df.merge(s, left_index=True, right_index=True, how="left")
    df["sent_score"] = df["sent_score"].fillna(0.0)
    df["sent_strength"] = df["sent_strength"].fillna(0.0)
    df["sent_count"] = df["sent_count"].fillna(0).astype(int)
    df["sent_reliable"] = df["sent_reliable"].fillna(False).astype(bool)
    return df


if __name__ == "__main__":
    # Smoke test with synthetic data
    import json
    sample = [
        {
            "stocks": ["INFY"],
            "probs": {"positive": 0.85, "negative": 0.05, "neutral": 0.10},
            "label": "positive",
            "published": datetime.now(IST).isoformat(),
            "source": "moneycontrol",
        },
        {
            "stocks": ["INFY"],
            "probs": {"positive": 0.10, "negative": 0.75, "neutral": 0.15},
            "label": "negative",
            "published": (datetime.now(IST) - timedelta(hours=20)).isoformat(),
            "source": "et_markets",
        },
        {
            "stocks": ["RELIANCE"],
            "probs": {"positive": 0.20, "negative": 0.20, "neutral": 0.60},
            "label": "neutral",
            "published": datetime.now(IST).isoformat(),
            "source": "livemint",
        },
    ]
    rows = aggregate(sample)
    df = to_dataframe(rows)
    print(df.to_string(index=False))
