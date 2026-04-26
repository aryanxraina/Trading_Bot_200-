"""
research_signal.py
Drop-in 9th vote for ensemble.py.

Same interface as sentiment_signal.py and all other voters:
Returns {vote, confidence, available, ...}

This vote represents the MARKET-LEVEL bias (not stock-specific).
Think of it as: "should we be trading at all today, and in which direction?"

Unlike the other 8 voters which look at per-stock data, this one looks at:
  - FII/DII institutional flows
  - India VIX (fear gauge)
  - US market overnight performance
  - Crude oil
  - Nifty momentum
  - USD/INR

Logic:
  - bias_score > +0.15 and confidence != "low" -> vote UP
  - bias_score < -0.15 and confidence != "low" -> vote DOWN
  - otherwise -> abstain

The threshold is intentionally low (0.15, not 0.25) because this is just
ONE vote in a 9-vote ensemble. It shouldn't be the decider — it should
nudge the consensus.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("research_signal")

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_DIR = Path("data/research")

BIAS_THRESHOLD = 0.15       # |bias_score| must exceed this to vote
CONFIDENCE_FLOOR = 0.54     # minimum confidence when threshold barely crossed
CONFIDENCE_CEILING = 0.70   # cap — market bias is broad, shouldn't dominate


def _load_latest_bias() -> dict | None:
    """Load the most recent market bias from cache."""
    if not CACHE_DIR.exists():
        return None

    today = datetime.now(IST).strftime("%Y-%m-%d")
    bias_path = CACHE_DIR / f"market_bias_{today}.json"
    if bias_path.exists():
        with bias_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Try yesterday (weekend/holiday)
    for days_back in range(1, 4):
        date = (datetime.now(IST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        bias_path = CACHE_DIR / f"market_bias_{date}.json"
        if bias_path.exists():
            with bias_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


def research_signal(df: pd.DataFrame = None) -> dict:
    """
    Produce a market-level vote based on the latest research briefing.

    Note: unlike other signals, this IGNORES the df parameter.
    It reads from the cached market bias file instead.
    df is accepted to match the voter interface but not used.
    """
    try:
        bias = _load_latest_bias()
        if bias is None:
            return {
                "vote": None,
                "confidence": 0.5,
                "available": False,
                "bias_score": 0.0,
                "reason": "no_bias_file",
            }

        bias_score = float(bias.get("bias_score", 0.0))
        confidence_level = bias.get("confidence", "low")
        factors = int(bias.get("factors_available", 0))

        # Don't vote if data quality is poor
        if confidence_level == "low" or factors < 3:
            return {
                "vote": None,
                "confidence": 0.5,
                "available": True,
                "bias_score": bias_score,
                "factors": factors,
                "reason": "low_confidence",
            }

        # Don't vote if bias is too weak
        if abs(bias_score) < BIAS_THRESHOLD:
            return {
                "vote": None,
                "confidence": 0.5,
                "available": True,
                "bias_score": bias_score,
                "factors": factors,
                "reason": "below_threshold",
            }

        # Map |bias_score| in [THRESHOLD, 1.0] to confidence in [FLOOR, CEILING]
        strength = abs(bias_score)
        normalized = (strength - BIAS_THRESHOLD) / (1.0 - BIAS_THRESHOLD)
        normalized = max(0.0, min(1.0, normalized))
        conf = CONFIDENCE_FLOOR + normalized * (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)

        vote = "UP" if bias_score > 0 else "DOWN"

        return {
            "vote": vote,
            "confidence": round(conf, 4),
            "available": True,
            "bias_score": bias_score,
            "bias_label": bias.get("bias_label", "unknown"),
            "recommendation": bias.get("recommendation", "hold"),
            "factors": factors,
            "reason": "ok",
        }

    except Exception as e:
        logger.error(f"research_signal failed: {e}")
        return {
            "vote": None,
            "confidence": 0.5,
            "available": False,
            "bias_score": 0.0,
            "reason": f"error:{e}",
        }


if __name__ == "__main__":
    result = research_signal()
    print(f"Vote: {result['vote']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Bias score: {result['bias_score']}")
    print(f"Available: {result['available']}")
    print(f"Reason: {result['reason']}")
