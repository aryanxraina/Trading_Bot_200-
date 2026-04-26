"""
test_meta_learner.py
Tests for vote extractor and meta-learner. No model loading needed.

Run with:
    python -m tests.test_meta_learner
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.models.vote_extractor import (
    signal_to_features, get_feature_names, features_to_array, VOTER_ORDER
)
from src.models.meta_learner import MetaLearner

PASS = "+"
FAIL = "X"


def test(name, cond):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}")
    assert cond, f"FAILED: {name}"


# -------------------------------------------------------------------
# Vote Extractor
# -------------------------------------------------------------------
print("\n[Vote Extractor]")

sample = {
    "xgboost":      {"vote": "UP",   "confidence": 0.62, "available": True},
    "lstm":         {"vote": "DOWN", "confidence": 0.55, "available": True},
    "rsi":          {"vote": None,   "confidence": 0.50, "available": True},
    "macd":         {"vote": "UP",   "confidence": 0.65, "available": True},
    "volume":       {"vote": None,   "confidence": 0.50, "available": True},
    "trend_filter": {"vote": "UP",   "confidence": 0.58, "available": True},
    "nifty":        {"vote": None,   "confidence": 0.50, "available": True},
    "sentiment":    {"vote": None,   "confidence": 0.50, "available": False},
    "research":     {"vote": "DOWN", "confidence": 0.54, "available": True},
}
regime = {"trend": "bull", "vix_high": False}

feats = signal_to_features(sample, regime)
names = get_feature_names()

test("Feature count is 33",        len(feats) == 33)
test("Feature names count is 33",  len(names) == 33)
test("XGBoost UP = +1",            feats["vote_xgboost"] == 1.0)
test("LSTM DOWN = -1",             feats["vote_lstm"] == -1.0)
test("RSI abstain = 0",            feats["vote_rsi"] == 0.0)
test("Sentiment unavail = 0",      feats["avail_sentiment"] == 0.0)
test("XGBoost conf correct",       feats["conf_xgboost"] == 0.62)
test("Votes UP = 3",               feats["votes_up"] == 3.0)
test("Votes DOWN = 2",             feats["votes_down"] == 2.0)
test("Vote spread = +1",           feats["vote_spread"] == 1.0)
test("Is bull = 1",                feats["is_bull"] == 1.0)
test("Is bear = 0",                feats["is_bear"] == 0.0)
test("VIX high = 0",               feats["vix_high"] == 0.0)

# Test with bear regime
bear_feats = signal_to_features(sample, {"trend": "bear", "vix_high": True})
test("Bear regime = 1",            bear_feats["is_bear"] == 1.0)
test("VIX high = 1",               bear_feats["vix_high"] == 1.0)

# Test with no regime
no_regime = signal_to_features(sample)
test("No regime -> defaults 0",    no_regime["is_bull"] == 0.0)

# Array conversion
arr = features_to_array(feats)
test("Array shape is (33,)",        arr.shape == (33,))
test("Array first val = XGB vote",  arr[0] == 1.0)

# Empty signals
empty_feats = signal_to_features({})
test("Empty signals -> votes 0",   empty_feats["votes_up"] == 0.0 and empty_feats["votes_down"] == 0.0)


# -------------------------------------------------------------------
# Meta-Learner
# -------------------------------------------------------------------
print("\n[Meta-Learner Training]")

np.random.seed(42)
n = 300
n_feats = len(get_feature_names())
X = np.random.randn(n, n_feats)
# Make feature 0 (xgboost vote) and feature 9 (macd vote) predictive
y = ((X[:, 0] * 0.5 + X[:, 9] * 0.3 + np.random.randn(n) * 0.3) > 0).astype(int)

split = int(n * 0.7)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

ml = MetaLearner(threshold=0.55)
metrics = ml.train(X_train, y_train, X_val, y_val)

test("Training succeeded",          ml.trained is True)
test("Train accuracy > 55%",        metrics["train_accuracy"] > 0.55)
test("Has val accuracy",            metrics.get("val_accuracy_all") is not None)
test("Fire rate > 0",               metrics["fire_rate"] > 0)

# Predictions
proba = ml.predict_proba(X_val)
test("Proba shape correct",         proba.shape == (len(X_val),))
test("Proba in [0,1]",              proba.min() >= 0 and proba.max() <= 1)

signals = ml.predict_signal(X_val)
test("Signals length correct",      len(signals) == len(X_val))
test("Signals are valid",           all(s in ("UP", "DOWN", "NO_TRADE") for s in signals))

# Single prediction
single = ml.predict_single(feats)
test("Single has direction",         single["direction"] in ("UP", "DOWN", "NO_TRADE"))
test("Single has confidence",        0.4 <= single["confidence"] <= 1.0)
test("Single has meta_proba",        0 <= single["meta_proba"] <= 1)


# -------------------------------------------------------------------
# Weights inspection
# -------------------------------------------------------------------
print("\n[Weights]")
weights_df = ml.get_weights()
test("Weights has 33 rows",         len(weights_df) == n_feats)
test("Has weight column",           "weight" in weights_df.columns)
test("Sorted by abs_weight",        weights_df["abs_weight"].is_monotonic_decreasing)

# The most important feature should be vote_xgboost or conf_xgboost
# since we made X[:,0] (= vote_xgboost) the strongest predictor
top_feat = weights_df.iloc[0]["feature"]
test("Top feature related to XGB",  "xgboost" in top_feat or "macd" in top_feat)


# -------------------------------------------------------------------
# Save/Load
# -------------------------------------------------------------------
print("\n[Save/Load]")
import src.models.meta_learner as mlmod
orig_path = mlmod.MODEL_PATH
orig_scaler = mlmod.SCALER_PATH
orig_weights = mlmod.WEIGHTS_PATH

tmp = Path(tempfile.mkdtemp())
mlmod.MODEL_PATH = tmp / "meta_learner.pkl"
mlmod.SCALER_PATH = tmp / "meta_learner_scaler.pkl"
mlmod.WEIGHTS_PATH = tmp / "meta_learner_weights.json"
mlmod.MODEL_DIR = tmp

ml.save()
test("Model file exists",           mlmod.MODEL_PATH.exists())
test("Scaler file exists",          mlmod.SCALER_PATH.exists())
test("Weights JSON exists",         mlmod.WEIGHTS_PATH.exists())

ml2 = MetaLearner()
ml2.load()
proba2 = ml2.predict_proba(X_val[:5])
test("Loaded model matches",        np.allclose(proba[:5], proba2))

# Restore paths
mlmod.MODEL_PATH = orig_path
mlmod.SCALER_PATH = orig_scaler
mlmod.WEIGHTS_PATH = orig_weights
mlmod.MODEL_DIR = orig_path.parent


print("\n" + "=" * 50)
print(f"  All tests passed [{PASS}]")
print("=" * 50)
