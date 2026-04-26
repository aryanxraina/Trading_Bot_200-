"""
xgboost_model.py
XGBoost classifier for next-day price direction prediction.
Works on tabular features — much better than LSTM for financial indicators.

Architecture:
  Input  : 38 engineered features (RSI, MACD, Bollinger, VWAP, Delivery%, etc.)
  Model  : XGBoost binary classifier with class balancing
  Output : P(next day UP) → signal (1=UP, 0=DOWN)

Usage:
    from src.models.xgboost_model import XGBoostModel
    model = XGBoostModel()
    model.train(train_df, feature_cols)
    results = model.evaluate(test_df, feature_cols)
    model.save()
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from src.utils.logger import get_logger

logger = get_logger("xgboost")

MODEL_DIR = "models/saved"

# Best params for financial time series (tuned for NSE data)
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,   # auto-set during training
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}


class XGBoostModel:
    def __init__(self, params: dict = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []
        self.feature_importance_: pd.DataFrame | None = None

    def train(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        val_df: pd.DataFrame = None,
    ) -> dict:
        """
        Train XGBoost on training data.

        Args:
            train_df    : preprocessed training DataFrame
            feature_cols: list of feature column names
            val_df      : optional validation DataFrame for early stopping

        Returns:
            dict with training info
        """
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols].values
        y_train = train_df["signal"].values

        # Clean NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

        # Auto-balance classes (handles slight UP/DOWN imbalance)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        self.params["scale_pos_weight"] = round(n_neg / max(n_pos, 1), 3)
        logger.info(f"Class balance: {n_neg} DOWN / {n_pos} UP | scale_pos_weight={self.params['scale_pos_weight']}")

        eval_set = None
        if val_df is not None:
            X_val = np.nan_to_num(val_df[feature_cols].values, nan=0.0, posinf=1.0, neginf=-1.0)
            y_val = val_df["signal"].values
            eval_set = [(X_val, y_val)]

        self.model = xgb.XGBClassifier(**{k: v for k, v in self.params.items()
                                          if k != "early_stopping_rounds"},
                                       early_stopping_rounds=self.params["early_stopping_rounds"])

        logger.info(f"Training XGBoost | train={len(X_train)} | features={len(feature_cols)}")

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        best_iter = getattr(self.model, "best_iteration", self.params["n_estimators"])
        logger.info(f"Training complete | best iteration: {best_iter}")

        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return {"best_iteration": best_iter, "n_features": len(feature_cols)}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns P(UP) for each row."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = np.nan_to_num(df[self.feature_cols].values, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.model.predict_proba(X)[:, 1]

    def predict_signal(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary signals (1=UP, 0=DOWN)."""
        return (self.predict_proba(df) > threshold).astype(int)

    def evaluate(self, test_df: pd.DataFrame, feature_cols: list[str] = None) -> dict:
        """Full evaluation with accuracy, precision, recall, F1."""
        cols = feature_cols or self.feature_cols
        y_true = test_df["signal"].values
        y_pred = self.predict_signal(test_df)
        proba = self.predict_proba(test_df)

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=["DOWN", "UP"],
            output_dict=True
        )

        logger.info(f"Accuracy: {acc:.3f}")
        logger.info(f"\n{classification_report(y_true, y_pred, target_names=['DOWN', 'UP'])}")

        return {
            "accuracy": round(acc, 4),
            "precision_up": round(report["UP"]["precision"], 4),
            "recall_up": round(report["UP"]["recall"], 4),
            "recall_down": round(report["DOWN"]["recall"], 4),
            "f1_up": round(report["UP"]["f1-score"], 4),
            "f1_down": round(report["DOWN"]["f1-score"], 4),
            "total_samples": len(y_true),
            "mean_proba": round(float(proba.mean()), 4)
        }

    def top_features(self, n: int = 15) -> pd.DataFrame:
        """Returns top N most important features."""
        if self.feature_importance_ is None:
            raise RuntimeError("Train the model first.")
        return self.feature_importance_.head(n)

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        n_splits: int = 5
    ) -> dict:
        """
        Walk-forward cross validation using TimeSeriesSplit.
        More robust than single train/test split for financial data.
        """
        X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=1.0, neginf=-1.0)
        y = df["signal"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accs = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m = xgb.XGBClassifier(
                **{k: v for k, v in self.params.items()
                   if k not in ("early_stopping_rounds", "n_estimators")},
                n_estimators=200,
                early_stopping_rounds=20
            )
            m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            preds = m.predict(X_te)
            acc = accuracy_score(y_te, preds)
            fold_accs.append(acc)
            logger.info(f"Fold {fold}/{n_splits}: accuracy={acc:.3f}")

        mean_acc = np.mean(fold_accs)
        logger.info(f"CV mean accuracy: {mean_acc:.3f} (+/- {np.std(fold_accs):.3f})")

        return {
            "fold_accuracies": [round(a, 4) for a in fold_accs],
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(float(np.std(fold_accs)), 4)
        }

    def save(self, name: str = "xgboost"):
        """Save model to models/saved/."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = f"{MODEL_DIR}/{name}.pkl"
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "feature_importance": self.feature_importance_,
            "params": self.params
        }, path)
        logger.info(f"XGBoost saved to {path}")

    def load(self, name: str = "xgboost"):
        """Load saved model."""
        path = f"{MODEL_DIR}/{name}.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.feature_importance_ = data["feature_importance"]
        self.params = data["params"]
        logger.info(f"XGBoost loaded from {path}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data.preprocess import preprocess_symbol, get_feature_columns, get_train_test_split

    print("Quick test on RELIANCE...")
    df = preprocess_symbol("RELIANCE")
    feat_cols = get_feature_columns(df)
    train_df, test_df = get_train_test_split(df, test_start="2019-01-01")

    model = XGBoostModel()
    model.train(train_df, feat_cols, val_df=test_df)
    results = model.evaluate(test_df, feat_cols)
    print(results)
    print("\nTop 10 features:")
    print(model.top_features(10))