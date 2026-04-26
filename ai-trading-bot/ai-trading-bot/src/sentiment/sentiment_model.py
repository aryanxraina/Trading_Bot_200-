"""
sentiment_model.py
FinBERT wrapper for financial-text sentiment classification.

Model: ProsusAI/finbert  (3-class: positive / negative / neutral)
       Pretrained on financial news, returns calibrated probabilities.

Why FinBERT, not generic BERT or an LLM:
  - Generic BERT misclassifies "earnings beat" as neutral or even negative
    ("beat" is often violent in normal text). FinBERT learned the financial
    register and handles "beat", "miss", "guidance cut", "upgrade" correctly.
  - LLM API calls are 100-1000x slower and cost money. FinBERT runs free on
    CPU at ~5-20 headlines/sec.

Caching:
  We hash (model_name + text) and cache the prediction. FinBERT is deterministic
  so this is safe. On a 5000-headline backfill, the second run takes seconds.

Usage:
    from src.sentiment.sentiment_model import SentimentModel
    sm = SentimentModel()
    sm.load()
    out = sm.predict("Infosys Q4 profit beats street estimates by 8%")
    # → {"label": "positive", "score": 0.91, "probs": {...}}
    batch = sm.predict_batch(["...", "...", "..."])
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from src.utils.logger import get_logger

logger = get_logger("sentiment_model")

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]

CACHE_DIR = Path("data/sentiment/cache")
CACHE_FILE = CACHE_DIR / "finbert_cache.jsonl"


def _hash_text(text: str) -> str:
    return hashlib.sha1(f"{MODEL_NAME}::{text}".encode("utf-8")).hexdigest()


class SentimentModel:
    """Thin FinBERT wrapper. Loads lazily, caches results."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu",
                 cache_path: Path = CACHE_FILE):
        self.model_name = model_name
        self.device = device
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._cache[rec["hash"]] = rec["pred"]
                except (KeyError, ValueError):
                    continue
        logger.info(f"Loaded {len(self._cache)} cached predictions")

    def _append_cache(self, h: str, pred: dict) -> None:
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": h, "pred": pred}) + "\n")

    def load(self) -> None:
        """Lazy import + model load. Heavy step."""
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch required. "
                "Install with: pip install transformers torch"
            ) from e
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch
        logger.info("FinBERT ready")

    def predict(self, text: str) -> dict:
        """
        Returns {
          "label": "positive" | "negative" | "neutral",
          "score":  float in [0, 1]      # confidence of chosen label
          "probs":  {"positive": .., "negative": .., "neutral": ..}
        }
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 1.0,
                    "probs": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}

        h = _hash_text(text)
        if h in self._cache:
            return self._cache[h]

        if self._model is None:
            self.load()

        torch = self._torch
        # Truncate to 512 tokens (FinBERT max)
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        # FinBERT label order in the original config is ["positive","negative","neutral"]
        # but verify against id2label to be safe.
        id2label = self._model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}
        prob_map = {id2label[i].lower(): float(p) for i, p in enumerate(probs)}
        # Force the keys we expect
        for k in ("positive", "negative", "neutral"):
            prob_map.setdefault(k, 0.0)

        label = max(prob_map, key=prob_map.get)
        pred = {
            "label": label,
            "score": round(prob_map[label], 4),
            "probs": {k: round(v, 4) for k, v in prob_map.items()},
        }
        self._cache[h] = pred
        self._append_cache(h, pred)
        return pred

    def predict_batch(self, texts: Iterable[str], batch_size: int = 16) -> list[dict]:
        """Batch inference. Caches at the per-text level, so partial cache hits are OK."""
        texts = list(texts)
        results: list[dict | None] = [None] * len(texts)

        # Resolve cache hits first
        to_predict_idx: list[int] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                results[i] = {"label": "neutral", "score": 1.0,
                              "probs": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}
                continue
            h = _hash_text(t)
            cached = self._cache.get(h)
            if cached is not None:
                results[i] = cached
            else:
                to_predict_idx.append(i)

        if not to_predict_idx:
            return results  # type: ignore

        if self._model is None:
            self.load()
        torch = self._torch

        for start in range(0, len(to_predict_idx), batch_size):
            chunk_idx = to_predict_idx[start:start + batch_size]
            chunk_texts = [texts[i] for i in chunk_idx]
            inputs = self._tokenizer(
                chunk_texts, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

            id2label = self._model.config.id2label
            for j, i in enumerate(chunk_idx):
                row_probs = probs[j]
                prob_map = {id2label[k].lower(): float(p) for k, p in enumerate(row_probs)}
                for k in ("positive", "negative", "neutral"):
                    prob_map.setdefault(k, 0.0)
                label = max(prob_map, key=prob_map.get)
                pred = {
                    "label": label,
                    "score": round(prob_map[label], 4),
                    "probs": {k: round(v, 4) for k, v in prob_map.items()},
                }
                results[i] = pred
                h = _hash_text(texts[i])
                self._cache[h] = pred
                self._append_cache(h, pred)

        return results  # type: ignore


if __name__ == "__main__":
    sm = SentimentModel()
    samples = [
        "Infosys Q4 profit beats street estimates by 8%; raises FY26 guidance",
        "Reliance Industries posts disappointing quarterly results, misses estimates",
        "HDFC Bank announces new branch openings across tier-2 cities",
        "Nifty 50 closes flat amid mixed global cues",
        "Tata Motors faces probe over emissions disclosure",
    ]
    print("Loading FinBERT (first run downloads ~440MB)...")
    sm.load()
    for txt in samples:
        out = sm.predict(txt)
        print(f"\n  {txt}")
        print(f"  → {out['label']:>8s}  conf={out['score']:.2f}  {out['probs']}")
