"""
lstm.py
LSTM model for next-day price direction prediction.
Automatically uses GPU (CUDA) if available — optimised for RTX 5050.

Architecture:
  Input  : 60-day lookback window of N features
  Layer 1: LSTM(128 units) + Dropout(0.3)
  Layer 2: LSTM(64 units)  + Dropout(0.2)
  Layer 3: Dense(32)       + ReLU
  Output : Dense(1)        + Sigmoid → P(next day UP)

Fixes applied:
  - GPU: device now correctly detects CUDA instead of hardcoding CPU
  - Loss: switched from BCELoss → BCEWithLogitsLoss with pos_weight
           to prevent model collapsing to predicting one class always
  - Network: removed Sigmoid from _LSTMNet (BCEWithLogitsLoss applies it internally)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.utils.logger import get_logger

logger = get_logger("lstm")

MODEL_DIR = "models/saved"
LOOKBACK = 60
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2


# ─────────────────────────────────────────
# PyTorch model definition
# ─────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, HIDDEN_SIZE_1, batch_first=True)
        self.drop1 = nn.Dropout(DROPOUT_1)
        self.lstm2 = nn.LSTM(HIDDEN_SIZE_1, HIDDEN_SIZE_2, batch_first=True)
        self.drop2 = nn.Dropout(DROPOUT_2)
        self.fc1 = nn.Linear(HIDDEN_SIZE_2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        # FIX: removed Sigmoid here — BCEWithLogitsLoss applies it internally
        # This prevents numerical instability and class collapse

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])   # take last timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(1)   # raw logits — no sigmoid here


# ─────────────────────────────────────────
# Data preparation helpers
# ─────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Convert flat arrays into (samples, lookback, features) sequences."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─────────────────────────────────────────
# Main LSTM wrapper class
# ─────────────────────────────────────────

class LSTMModel:
    def __init__(self, lookback: int = LOOKBACK):
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.net: _LSTMNet | None = None
        self.input_size: int = 0

        # FIX: was hardcoded to cpu — now correctly detects CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LSTM using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def _prepare(self, df: pd.DataFrame, feature_cols: list[str], fit_scaler: bool = True):
        """Scale features and create sequences."""
        X_raw = df[feature_cols].values
        y_raw = df["signal"].values

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            X_scaled = self.scaler.transform(X_raw)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

        return _make_sequences(X_scaled, y_raw, self.lookback)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        epochs: int = EPOCHS,
        patience: int = 8
    ) -> dict:
        logger.info(f"Preparing sequences (lookback={self.lookback})...")
        X_train, y_train = self._prepare(train_df, feature_cols, fit_scaler=True)
        X_val, y_val = self._prepare(val_df, feature_cols, fit_scaler=False)

        self.input_size = X_train.shape[2]
        self.net = _LSTMNet(self.input_size).to(self.device)

        logger.info(
            f"Model: input={self.input_size} features | "
            f"train={len(X_train)} | val={len(X_val)} samples"
        )

        train_ds = TensorDataset(
            torch.tensor(X_train).to(self.device),
            torch.tensor(y_train).to(self.device)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val).to(self.device),
            torch.tensor(y_val).to(self.device)
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # FIX: BCEWithLogitsLoss + pos_weight prevents class collapse
        # pos_weight = ratio of negative to positive samples
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # ── Train
            self.net.train()
            train_losses = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validate
            self.net.eval()
            val_losses, val_preds_all, val_labels_all = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = self.net(xb)
                    val_losses.append(criterion(logits, yb).item())
                    probs = torch.sigmoid(logits)
                    val_preds_all.extend((probs.cpu() > 0.5).float().numpy())
                    val_labels_all.extend(yb.cpu().numpy())

            t_loss = np.mean(train_losses)
            v_loss = np.mean(val_losses)
            v_acc = accuracy_score(val_labels_all, val_preds_all)
            scheduler.step(v_loss)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {t_loss:.4f} | "
                    f"Val Loss: {v_loss:.4f} | "
                    f"Val Acc: {v_acc:.3f}"
                )

            # Early stopping on val loss
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.net.load_state_dict(best_state)

        logger.info(f"Training complete — Best val accuracy: {max(history['val_acc']):.3f}")
        return history

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Returns P(UP) probabilities for each row after lookback warmup."""
        if self.net is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X, _ = self._prepare(df, feature_cols, fit_scaler=False)
        tensor = torch.tensor(X).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict_signal(self, df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.5) -> np.ndarray:
        """Returns binary signals (1=UP, 0=DOWN)."""
        return (self.predict(df, feature_cols) > threshold).astype(int)

    def evaluate(self, df: pd.DataFrame, feature_cols: list[str]) -> dict:
        """Full evaluation with accuracy, precision, recall."""
        X, y_true = self._prepare(df, feature_cols, fit_scaler=False)
        tensor = torch.tensor(X).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        y_pred = (probs > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)

        report = classification_report(y_true, y_pred, target_names=["DOWN", "UP"], output_dict=True)
        logger.info(f"Accuracy: {acc:.3f}")
        logger.info(f"\n{classification_report(y_true, y_pred, target_names=['DOWN','UP'])}")

        return {
            "accuracy": round(acc, 4),
            "precision_up": round(report["UP"]["precision"], 4),
            "recall_up": round(report["UP"]["recall"], 4),
            "f1_up": round(report["UP"]["f1-score"], 4),
            "total_samples": len(y_true)
        }

    def save(self, name: str = "lstm"):
        """Save model weights and scaler to models/saved/."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_size": self.input_size,
            "lookback": self.lookback
        }, f"{MODEL_DIR}/{name}.pt")
        joblib.dump(self.scaler, f"{MODEL_DIR}/{name}_scaler.pkl")
        logger.info(f"Model saved to {MODEL_DIR}/{name}.pt")

    def load(self, name: str = "lstm"):
        """Load saved model weights and scaler."""
        pt_path = f"{MODEL_DIR}/{name}.pt"
        scaler_path = f"{MODEL_DIR}/{name}_scaler.pkl"

        checkpoint = torch.load(pt_path, map_location=self.device)
        self.input_size = checkpoint["input_size"]
        self.lookback = checkpoint["lookback"]

        self.net = _LSTMNet(self.input_size).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded from {pt_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data.preprocess import preprocess_symbol, get_feature_columns, get_train_test_split

    print("Quick test on RELIANCE...")
    df = preprocess_symbol("RELIANCE")
    feat_cols = get_feature_columns(df)
    train_df, test_df = get_train_test_split(df, test_start="2022-01-01")

    model = LSTMModel(lookback=60)
    history = model.train(train_df, test_df, feat_cols, epochs=10)
    results = model.evaluate(test_df, feat_cols)
    print(results)