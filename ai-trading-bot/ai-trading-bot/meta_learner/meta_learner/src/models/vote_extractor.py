"""
vote_extractor.py
Extracts a flat feature vector from ensemble's 9 voters for the meta-learner.

The ensemble currently does: count UP votes >= 3 -> trade UP.
That's dumb — it treats RSI the same as XGBoost.

The meta-learner replaces this with a logistic regression that LEARNS:
  - Which voters are accurate in which conditions
  - How to weight confident votes vs unsure ones
  - Which voter combinations are predictive

This module converts the raw signal dict from ensemble.predict() into a
numeric feature vector that the meta-learner can train on.

Feature vector (per prediction):
  - 9 vote features:  +1 (UP), -1 (DOWN), 0 (abstain) for each voter
  - 9 confidence features: raw confidence [0, 1] for each voter
  - 9 available features: 1 if voter had data, 0 if not
  - 3 aggregate features: votes_up, votes_down, vote_spread
  - 2 regime features: is_bull, is_bear
  - 1 vix feature: vix_high flag
  Total: 33 features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOTER_ORDER = [
    "xgboost", "lstm", "rsi", "macd", "volume",
    "trend_filter", "nifty", "sentiment", "research"
]


def signal_to_features(model_signals: dict, regime: dict = None) -> dict:
    """
    Convert ensemble model_signals dict into a flat feature dict.

    Args:
        model_signals: the .model_signals from EnsembleSignal
        regime: the regime dict from ensemble._get_regime()

    Returns:
        dict of feature_name -> float value
    """
    features = {}

    votes_up = 0
    votes_down = 0

    for voter in VOTER_ORDER:
        sig = model_signals.get(voter, {})
        vote = sig.get("vote")
        conf = sig.get("confidence", 0.5)
        avail = 1.0 if sig.get("available", False) else 0.0

        # Vote encoding: +1 UP, -1 DOWN, 0 abstain
        if vote == "UP":
            vote_val = 1.0
            votes_up += 1
        elif vote == "DOWN":
            vote_val = -1.0
            votes_down += 1
        else:
            vote_val = 0.0

        features[f"vote_{voter}"] = vote_val
        features[f"conf_{voter}"] = float(conf)
        features[f"avail_{voter}"] = avail

    # Aggregate features
    features["votes_up"] = float(votes_up)
    features["votes_down"] = float(votes_down)
    features["vote_spread"] = float(votes_up - votes_down)

    # Regime features
    if regime:
        features["is_bull"] = 1.0 if regime.get("trend") == "bull" else 0.0
        features["is_bear"] = 1.0 if regime.get("trend") == "bear" else 0.0
        features["vix_high"] = 1.0 if regime.get("vix_high", False) else 0.0
    else:
        features["is_bull"] = 0.0
        features["is_bear"] = 0.0
        features["vix_high"] = 0.0

    return features


def get_feature_names() -> list[str]:
    """Return ordered list of all meta-learner feature names."""
    names = []
    for voter in VOTER_ORDER:
        names.append(f"vote_{voter}")
        names.append(f"conf_{voter}")
        names.append(f"avail_{voter}")
    names.extend(["votes_up", "votes_down", "vote_spread",
                  "is_bull", "is_bear", "vix_high"])
    return names


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    names = get_feature_names()
    return np.array([features.get(n, 0.0) for n in names])


if __name__ == "__main__":
    # Smoke test with sample signals
    sample_signals = {
        "xgboost":      {"vote": "DOWN", "confidence": 0.52, "available": True},
        "lstm":         {"vote": "UP",   "confidence": 0.55, "available": True},
        "rsi":          {"vote": None,   "confidence": 0.50, "available": True},
        "macd":         {"vote": "DOWN", "confidence": 0.55, "available": True},
        "volume":       {"vote": None,   "confidence": 0.50, "available": True},
        "trend_filter": {"vote": "DOWN", "confidence": 0.58, "available": True},
        "nifty":        {"vote": None,   "confidence": 0.50, "available": True},
        "sentiment":    {"vote": None,   "confidence": 0.50, "available": False},
        "research":     {"vote": "DOWN", "confidence": 0.54, "available": True},
    }
    regime = {"trend": "bear", "vix_high": False}
    feats = signal_to_features(sample_signals, regime)
    print(f"Feature count: {len(feats)}")
    for k, v in feats.items():
        print(f"  {k:>25s}: {v:+.3f}")
