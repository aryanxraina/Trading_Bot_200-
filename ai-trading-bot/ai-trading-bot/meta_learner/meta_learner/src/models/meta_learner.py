"""
meta_learner.py
Logistic regression that learns optimal vote weights from backtest data.

Instead of:  "if votes_UP >= 3: trade UP"   (dumb equal-weight counting)
It learns:   "XGBoost UP + MACD crossover + bear regime = 62% chance of DOWN"

Architecture:
  - Input: 33 features from vote_extractor.py
  - Model: L2-regularized logistic regression (sklearn)
  - Output: P(UP) probability
  - Decision: P(UP) > threshold -> UP, P(UP) < (1-threshold) -> DOWN, else NO_TRADE

Why logistic regression and not a neural net or another XGBoost:
  1. We have 9 voters. That's 33 features. You don't need deep learning for 33 features.
  2. LR is interpretable — you can read the coefficients and see WHICH voters matter.
  3. LR trains in <1 second. You can retrain weekly with zero cost.
  4. LR won't overfit on small vote-level training data the way a tree model would.
  5. The coefficients literally ARE the learned vote weights. That's the whole point.

Usage:
    from src.models.meta_learner import MetaLearner
    ml = MetaLearner()
    ml.train(X_train, y_train)    # X = vote features, y = actual UP/DOWN
    proba = ml.predict_proba(X_test)
    ml.save()
    ml.load()
    ml.print_weights()            # see which voters matter
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from src.models.vote_extractor import get_feature_names
from src.utils.logger import get_logger

logger = get_logger("meta_learner")

MODEL_DIR = Path("models/saved")
MODEL_PATH = MODEL_DIR / "meta_learner.pkl"
SCALER_PATH = MODEL_DIR / "meta_learner_scaler.pkl"
WEIGHTS_PATH = MODEL_DIR / "meta_learner_weights.json"


class MetaLearner:
    """
    Thin wrapper around sklearn LogisticRegression.
    Learns optimal vote weights from historical ensemble predictions.
    """

    def __init__(self, threshold: float = 0.55, C: float = 1.0):
        """
        Args:
            threshold: P(UP) above this -> vote UP. Below (1-threshold) -> vote DOWN.
                       Between -> NO_TRADE (abstain).
                       Default 0.55 means we need 55% confidence to fire.
            C: Regularization strength (lower = more regularization).
               Default 1.0 is sklearn default. Decrease if overfitting.
        """
        self.threshold = threshold
        self.C = C
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names = get_feature_names()
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the meta-learner.

        Args:
            X: (n_samples, 33) feature matrix from vote_extractor
            y: (n_samples,) binary labels (1=UP, 0=DOWN)
            X_val: optional validation set features
            y_val: optional validation set labels

        Returns:
            dict with training metrics
        """
        # Scale features (important for LR — votes are [-1,1] but confidence is [0,1])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train
        self.model = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",  # handles slight class imbalance
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        self.trained = True

        # Training metrics
        train_pred = self.model.predict(X_scaled)
        train_acc = accuracy_score(y, train_pred)
        train_proba = self.model.predict_proba(X_scaled)[:, 1]

        # How often does it fire (not abstain)?
        fires = ((train_proba > self.threshold) | (train_proba < (1 - self.threshold))).sum()
        fire_rate = fires / len(y)

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "train_samples": len(y),
            "train_up_pct": round(y.mean(), 4),
            "fire_rate": round(fire_rate, 4),
            "n_features": X.shape[1],
        }

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            val_proba = self.model.predict_proba(X_val_scaled)[:, 1]

            # Accuracy ONLY on trades the meta-learner would take
            val_fires = (val_proba > self.threshold) | (val_proba < (1 - self.threshold))
            if val_fires.sum() > 0:
                fired_pred = (val_proba[val_fires] > 0.5).astype(int)
                fired_actual = y_val[val_fires]
                fired_acc = accuracy_score(fired_actual, fired_pred)
                metrics["val_accuracy_all"] = round(val_acc, 4)
                metrics["val_accuracy_fired"] = round(fired_acc, 4)
                metrics["val_fire_rate"] = round(val_fires.sum() / len(y_val), 4)
                metrics["val_fired_count"] = int(val_fires.sum())
            else:
                metrics["val_accuracy_all"] = round(val_acc, 4)
                metrics["val_accuracy_fired"] = None
                metrics["val_fire_rate"] = 0.0

        logger.info(f"Meta-learner trained: {metrics}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(UP) for each row."""
        if not self.trained:
            raise RuntimeError("Meta-learner not trained. Call train() or load() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_signal(self, X: np.ndarray) -> list[str]:
        """
        Returns list of "UP" / "DOWN" / "NO_TRADE" for each row.
        Uses threshold-based decision (not argmax).
        """
        proba = self.predict_proba(X)
        signals = []
        for p in proba:
            if p > self.threshold:
                signals.append("UP")
            elif p < (1 - self.threshold):
                signals.append("DOWN")
            else:
                signals.append("NO_TRADE")
        return signals

    def predict_single(self, features: dict) -> dict:
        """
        Predict from a single feature dict (output of signal_to_features).
        Returns: {direction, confidence, meta_proba}
        """
        from src.models.vote_extractor import features_to_array
        X = features_to_array(features).reshape(1, -1)
        proba = float(self.predict_proba(X)[0])

        if proba > self.threshold:
            direction = "UP"
            confidence = proba
        elif proba < (1 - self.threshold):
            direction = "DOWN"
            confidence = 1 - proba
        else:
            direction = "NO_TRADE"
            confidence = max(proba, 1 - proba)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "meta_proba": round(proba, 4),
        }

    def get_weights(self) -> pd.DataFrame:
        """
        Return learned weights as a DataFrame.
        Positive weight = this feature predicts UP.
        Negative weight = this feature predicts DOWN.
        Large absolute weight = this feature matters a lot.
        """
        if not self.trained:
            raise RuntimeError("Not trained yet")
        coefs = self.model.coef_[0]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "weight": coefs,
            "abs_weight": np.abs(coefs),
        }).sort_values("abs_weight", ascending=False).reset_index(drop=True)
        return df

    def print_weights(self) -> None:
        """Pretty-print the learned vote weights."""
        df = self.get_weights()
        print("\n=== Meta-Learner Weights (most important first) ===")
        print(f"{'Feature':>25s}  {'Weight':>8s}  {'Direction':>10s}")
        print("-" * 50)
        for _, row in df.head(15).iterrows():
            direction = "-> UP" if row["weight"] > 0 else "-> DOWN"
            print(f"{row['feature']:>25s}  {row['weight']:+8.4f}  {direction:>10s}")
        print()

    def save(self) -> None:
        """Save model, scaler, and weights to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open("wb") as f:
            pickle.dump(self.model, f)
        with SCALER_PATH.open("wb") as f:
            pickle.dump(self.scaler, f)
        # Also save weights as JSON for easy inspection
        weights = self.get_weights()
        weights_dict = {row["feature"]: round(row["weight"], 6)
                        for _, row in weights.iterrows()}
        with WEIGHTS_PATH.open("w") as f:
            json.dump(weights_dict, f, indent=2)
        logger.info(f"Meta-learner saved to {MODEL_DIR}")

    def load(self) -> None:
        """Load model and scaler from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No saved meta-learner at {MODEL_PATH}")
        with MODEL_PATH.open("rb") as f:
            self.model = pickle.load(f)
        with SCALER_PATH.open("rb") as f:
            self.scaler = pickle.load(f)
        self.trained = True
        logger.info("Meta-learner loaded")


if __name__ == "__main__":
    # Smoke test with random data
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, len(get_feature_names()))
    y = (X[:, 0] + X[:, 3] + np.random.randn(n) * 0.5 > 0).astype(int)

    ml = MetaLearner(threshold=0.55)
    split = int(n * 0.7)
    metrics = ml.train(X[:split], y[:split], X[split:], y[split:])
    print(f"\nMetrics: {metrics}")
    ml.print_weights()
    ml.save()
    print("Save/load test:")
    ml2 = MetaLearner()
    ml2.load()
    print(f"  Loaded model matches: {np.allclose(ml.predict_proba(X[:5]), ml2.predict_proba(X[:5]))}")
