"""
sentiment_signal.py
Drop-in vote function for ensemble.py.

This function has the EXACT same interface as your existing _xgb_signal,
_lstm_signal, _rsi_signal, etc. so adding it to the ensemble is a 3-line patch.

Returns:
    {
      "vote":       "UP" | "DOWN" | None,
      "confidence": float in [0.5, 0.75],
      "available":  bool,
      "sent_score": float,    # raw underlying score for logging
      "sent_count": int,
    }

Trigger logic:
  - Need a sentiment row for today (or last available business day)
  - Need sent_count >= MIN_HEADLINES (default 2) — otherwise abstain
  - Need |sent_score| >= MIN_SCORE_THRESHOLD (default 0.20) — otherwise abstain
  - Vote UP if sent_score > 0, DOWN if < 0
  - Confidence scales linearly with |sent_score|, capped at 0.75
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("sentiment_signal")

MIN_HEADLINES = 2
MIN_SCORE_THRESHOLD = 0.20  # |sent_score| below this → abstain
CONFIDENCE_FLOOR = 0.55     # baseline confidence when threshold barely crossed
CONFIDENCE_CEILING = 0.75   # cap to prevent over-betting on a single signal


def sentiment_signal(df: pd.DataFrame) -> dict:
    """
    Read the latest row of df and produce a vote.

    Expects df to have these columns (added by sentiment_aggregator.merge_into_features):
        sent_score      (float, [-1, 1])
        sent_strength   (float, [0, 1])
        sent_count      (int)
        sent_reliable   (bool)
    """
    try:
        if not all(c in df.columns for c in ("sent_score", "sent_count", "sent_reliable")):
            return {"vote": None, "confidence": 0.5, "available": False,
                    "sent_score": 0.0, "sent_count": 0,
                    "reason": "no_sentiment_columns"}

        row = df.iloc[-1]
        sent_score = float(row["sent_score"])
        sent_count = int(row["sent_count"])
        reliable = bool(row["sent_reliable"])

        if not reliable or sent_count < MIN_HEADLINES:
            return {"vote": None, "confidence": 0.5, "available": True,
                    "sent_score": sent_score, "sent_count": sent_count,
                    "reason": "insufficient_headlines"}

        if abs(sent_score) < MIN_SCORE_THRESHOLD:
            return {"vote": None, "confidence": 0.5, "available": True,
                    "sent_score": sent_score, "sent_count": sent_count,
                    "reason": "below_threshold"}

        # Map |sent_score| in [MIN_SCORE_THRESHOLD, 1.0] →
        # confidence in [CONFIDENCE_FLOOR, CONFIDENCE_CEILING]
        strength = abs(sent_score)
        normalized = (strength - MIN_SCORE_THRESHOLD) / (1.0 - MIN_SCORE_THRESHOLD)
        normalized = max(0.0, min(1.0, normalized))
        confidence = CONFIDENCE_FLOOR + normalized * (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)

        vote = "UP" if sent_score > 0 else "DOWN"

        return {
            "vote": vote,
            "confidence": round(confidence, 4),
            "available": True,
            "sent_score": sent_score,
            "sent_count": sent_count,
            "reason": "ok",
        }
    except Exception as e:
        logger.error(f"sentiment_signal failed: {e}")
        return {"vote": None, "confidence": 0.5, "available": False,
                "sent_score": 0.0, "sent_count": 0, "reason": f"error:{e}"}


if __name__ == "__main__":
    # Synthetic tests
    cases = [
        ("Strong positive",  {"sent_score": 0.65, "sent_count": 5, "sent_reliable": True}),
        ("Strong negative",  {"sent_score": -0.55, "sent_count": 8, "sent_reliable": True}),
        ("Weak positive",    {"sent_score": 0.10, "sent_count": 4, "sent_reliable": True}),
        ("Few headlines",    {"sent_score": 0.80, "sent_count": 1, "sent_reliable": False}),
        ("No data",          {"sent_score": 0.0, "sent_count": 0, "sent_reliable": False}),
    ]
    for name, row in cases:
        df = pd.DataFrame([row])
        out = sentiment_signal(df)
        print(f"{name:>20s} → vote={out['vote']!s:>5s} conf={out['confidence']:.3f}  reason={out['reason']}")
