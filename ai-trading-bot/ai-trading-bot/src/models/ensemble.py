"""
ensemble.py - Updated with improved voting logic:
  1. Nifty index return as market context feature
  2. Trend filter — longs only when price > SMA20
  3. Take-profit at 2.5% (applied in backtest notebook)

Changes from v2:
  - CONFIDENCE_THRESHOLD: 0.55 → 0.52 (more signals fire)
  - RSI thresholds: 35/65 → 40/60 (votes more often)
  - Nifty threshold: 1% → 0.3% (votes more often)
  - Volume threshold: 1.2 → 1.1 (votes more often)
  - TAKE_PROFIT_PCT: 0.015 → 0.025 (2.5% — better reward:risk)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.data.preprocess import get_feature_columns

logger = get_logger("ensemble")

MIN_VOTES = 2
CONFIDENCE_THRESHOLD = 0.52   # CHANGED: was 0.55 — more signals fire
TAKE_PROFIT_PCT = 0.025        # CHANGED: was 0.015 — 2.5% take-profit (reward:risk = 2.5:1)


@dataclass
class EnsembleSignal:
    symbol: str
    direction: str
    confidence: float
    votes_up: int
    votes_down: int
    total_votes: int
    regime: str
    veto: bool
    veto_reason: str
    model_signals: dict


class EnsembleModel:
    def __init__(self, min_votes: int = MIN_VOTES, backtest_mode: bool = True):
        self.min_votes = min_votes
        self.backtest_mode = backtest_mode
        self.xgb_model = None
        self.lstm_model = None
        self.models_loaded = False

    def load_models(self):
        try:
            from src.models.xgboost_model import XGBoostModel
            self.xgb_model = XGBoostModel()
            self.xgb_model.load("xgboost")
            logger.info("XGBoost loaded")
        except Exception as e:
            logger.warning(f"XGBoost load failed: {e}")
            self.xgb_model = None

        try:
            from src.models.lstm import LSTMModel
            self.lstm_model = LSTMModel()
            self.lstm_model.load("lstm")
            logger.info("LSTM loaded")
        except Exception as e:
            logger.warning(f"LSTM load failed: {e}")
            self.lstm_model = None

        self.models_loaded = True
        logger.info(f"Ensemble ready | min_votes={self.min_votes} | backtest={self.backtest_mode}")

    def _xgb_signal(self, df: pd.DataFrame) -> dict:
        if self.xgb_model is None:
            return {"vote": None, "confidence": 0.5, "available": False}
        try:
            feat_cols = [c for c in self.xgb_model.feature_cols if c in df.columns]
            if len(feat_cols) < 10:
                return {"vote": None, "confidence": 0.5, "available": False}
            original_cols = self.xgb_model.feature_cols
            self.xgb_model.feature_cols = feat_cols
            proba = float(self.xgb_model.predict_proba(df.tail(1))[0])
            self.xgb_model.feature_cols = original_cols
            vote = "UP" if proba >= 0.5 else "DOWN"
            return {"vote": vote, "confidence": proba if vote == "UP" else 1 - proba,
                    "raw_proba": proba, "available": True}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _lstm_signal(self, df: pd.DataFrame) -> dict:
        if self.lstm_model is None:
            return {"vote": None, "confidence": 0.5, "available": False}
        try:
            feat_cols = get_feature_columns(df)
            if len(df) < self.lstm_model.lookback + 5:
                return {"vote": None, "confidence": 0.5, "available": False}
            proba = float(self.lstm_model.predict(df, feat_cols)[-1])
            vote = "UP" if proba >= 0.5 else "DOWN"
            return {"vote": vote, "confidence": proba if vote == "UP" else 1 - proba,
                    "raw_proba": proba, "available": True}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _rsi_signal(self, df: pd.DataFrame) -> dict:
        try:
            if "rsi_14" not in df.columns:
                return {"vote": None, "confidence": 0.5, "available": False}
            rsi = float(df["rsi_14"].iloc[-1])
            # CHANGED: thresholds 35/65 → 40/60 so RSI votes more often
            if rsi < 40:
                return {"vote": "UP", "confidence": min(0.9, 0.5 + (40 - rsi) / 80),
                        "rsi": rsi, "available": True}
            elif rsi > 60:
                return {"vote": "DOWN", "confidence": min(0.9, 0.5 + (rsi - 60) / 80),
                        "rsi": rsi, "available": True}
            else:
                return {"vote": None, "confidence": 0.5, "rsi": rsi,
                        "available": True, "neutral": True}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _macd_signal(self, df: pd.DataFrame) -> dict:
        try:
            if "macd" not in df.columns or "macd_signal" not in df.columns or len(df) < 2:
                return {"vote": None, "confidence": 0.5, "available": False}
            hist_now = float(df["macd"].iloc[-1]) - float(df["macd_signal"].iloc[-1])
            hist_prev = float(df["macd"].iloc[-2]) - float(df["macd_signal"].iloc[-2])
            if hist_prev < 0 and hist_now > 0:
                return {"vote": "UP", "confidence": 0.65, "available": True, "crossover": True}
            elif hist_prev > 0 and hist_now < 0:
                return {"vote": "DOWN", "confidence": 0.65, "available": True, "crossover": True}
            elif hist_now > 0:
                return {"vote": "UP", "confidence": 0.55, "available": True}
            else:
                return {"vote": "DOWN", "confidence": 0.55, "available": True}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _volume_signal(self, df: pd.DataFrame) -> dict:
        try:
            if "volume_ratio" not in df.columns or "daily_return" not in df.columns:
                return {"vote": None, "confidence": 0.5, "available": False}
            vol_ratio = float(df["volume_ratio"].iloc[-1])
            daily_ret = float(df["daily_return"].iloc[-1])
            # CHANGED: threshold 1.2 → 1.1 so volume votes more often
            if vol_ratio < 1.1:
                return {"vote": None, "confidence": 0.5, "available": True, "neutral": True}
            confidence = min(0.75, 0.5 + vol_ratio * 0.05)
            vote = "UP" if daily_ret > 0 else "DOWN"
            return {"vote": vote, "confidence": confidence, "vol_ratio": vol_ratio, "available": True}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _trend_filter_signal(self, df: pd.DataFrame) -> dict:
        """
        Trend filter — only vote UP when price is above SMA20.
        Only vote DOWN when price is below SMA20.
        """
        try:
            if "above_sma20" in df.columns:
                above = int(df["above_sma20"].iloc[-1])
            elif "close_vs_sma20" in df.columns:
                above = 1 if float(df["close_vs_sma20"].iloc[-1]) > 0 else 0
            else:
                return {"vote": None, "confidence": 0.5, "available": False}

            if above == 1:
                return {"vote": "UP", "confidence": 0.58, "available": True, "above_sma20": True}
            else:
                return {"vote": "DOWN", "confidence": 0.58, "available": True, "above_sma20": False}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _nifty_signal(self, df: pd.DataFrame) -> dict:
        """
        Market context vote using Nifty 5d return.
        CHANGED: threshold 1% → 0.3% so it votes more often.
        """
        try:
            if "nifty_return_5d" not in df.columns:
                return {"vote": None, "confidence": 0.5, "available": False}
            nifty_5d = float(df["nifty_return_5d"].iloc[-1])
            # CHANGED: was >0.01 / <-0.01, now >0.003 / <-0.003
            if nifty_5d > 0.003:
                return {"vote": "UP", "confidence": 0.55, "available": True,
                        "nifty_5d": nifty_5d}
            elif nifty_5d < -0.003:
                return {"vote": "DOWN", "confidence": 0.55, "available": True,
                        "nifty_5d": nifty_5d}
            else:
                return {"vote": None, "confidence": 0.5, "available": True, "neutral": True}
        except Exception:
            return {"vote": None, "confidence": 0.5, "available": False}

    def _get_regime(self, df: pd.DataFrame = None) -> dict:
        if self.backtest_mode and df is not None:
            try:
                close = df["close"]
                ma_period = min(200, len(close) - 1)
                ma = close.rolling(ma_period).mean().iloc[-1]
                current = close.iloc[-1]
                trend = "bull" if current > ma * 1.02 else ("bear" if current < ma * 0.98 else "sideways")
                vol = df["volatility_20d"].iloc[-1] if "volatility_20d" in df.columns else 0.01
                vix_high = vol > 0.02
                multiplier = (0.5 if vix_high else 1.0) * (0.5 if trend == "bear" else 1.0)
                return {
                    "trend": trend,
                    "vix_high": vix_high,
                    "position_size_multiplier": multiplier,
                    "allow_longs": trend != "bear",
                    "allow_shorts": trend == "bear"
                }
            except Exception:
                pass
        else:
            try:
                from src.data.regime_detector import get_nifty_regime
                return get_nifty_regime()
            except Exception:
                pass

        return {"trend": "bull", "vix_high": False, "position_size_multiplier": 1.0,
                "allow_longs": True, "allow_shorts": True}

    def _check_regime_veto(self, direction: str, regime: dict) -> tuple:
        if regime.get("trend") == "bear" and direction == "UP":
            return True, "Bear market — no long positions"
        if direction == "UP" and not regime.get("allow_longs", True):
            return True, "Regime does not allow long positions"
        return False, ""

    def predict(self, df: pd.DataFrame, symbol: str = "") -> EnsembleSignal:
        if not self.models_loaded:
            self.load_models()

        signals = {
            "xgboost":      self._xgb_signal(df),
            "lstm":         self._lstm_signal(df),
            "rsi":          self._rsi_signal(df),
            "macd":         self._macd_signal(df),
            "volume":       self._volume_signal(df),
            "trend_filter": self._trend_filter_signal(df),
            "nifty":        self._nifty_signal(df),
        }

        votes_up, votes_down, confidences = 0, 0, []
        for sig in signals.values():
            if sig.get("vote") == "UP" and sig.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                votes_up += 1
                confidences.append(sig["confidence"])
            elif sig.get("vote") == "DOWN" and sig.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                votes_down += 1
                confidences.append(sig["confidence"])

        avg_confidence = float(np.mean(confidences)) if confidences else 0.5

        if votes_up >= self.min_votes:
            direction = "UP"
        elif votes_down >= self.min_votes:
            direction = "DOWN"
        else:
            direction = "NO_TRADE"

        regime = self._get_regime(df)
        veto, veto_reason = False, ""
        if direction != "NO_TRADE":
            veto, veto_reason = self._check_regime_veto(direction, regime)
            if veto:
                direction = "NO_TRADE"

        return EnsembleSignal(
            symbol=symbol,
            direction=direction,
            confidence=round(avg_confidence, 4),
            votes_up=votes_up,
            votes_down=votes_down,
            total_votes=votes_up + votes_down,
            regime=regime.get("trend", "unknown"),
            veto=veto,
            veto_reason=veto_reason,
            model_signals=signals
        )

    def predict_batch(self, symbol_dfs: dict) -> dict:
        if not self.models_loaded:
            self.load_models()
        results, tradeable = {}, []
        for symbol, df in symbol_dfs.items():
            try:
                signal = self.predict(df, symbol)
                results[symbol] = signal
                if signal.direction != "NO_TRADE":
                    tradeable.append((symbol, signal))
            except Exception as e:
                logger.error(f"{symbol}: ensemble failed — {e}")
        logger.info(f"Batch: {len(results)} symbols | {len(tradeable)} tradeable")
        return results

    def summary(self, results: dict) -> pd.DataFrame:
        rows = []
        for symbol, sig in results.items():
            rows.append({
                "symbol": symbol,
                "direction": sig.direction,
                "confidence": sig.confidence,
                "votes_up": sig.votes_up,
                "votes_down": sig.votes_down,
                "regime": sig.regime,
                "veto": sig.veto,
                "xgb_proba": sig.model_signals.get("xgboost", {}).get("raw_proba"),
                "rsi": sig.model_signals.get("rsi", {}).get("rsi"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("confidence", ascending=False)
        return df


# Convenience constant for backtest notebooks
TAKE_PROFIT_PCT = 0.025   # CHANGED: was 0.015