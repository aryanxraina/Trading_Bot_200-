"""
ensemble_patch_meta.py
========================
How to upgrade ensemble.py to use the meta-learner instead of vote counting.

This is the FINAL patch. After this, your ensemble uses learned weights
instead of "count to 3".

There are TWO modes:
  A) Meta-learner available (trained model exists) -> use learned weights
  B) Meta-learner not available (first time) -> fall back to old vote counting

This means the system never breaks. If you haven't trained the meta-learner
yet, it works exactly as before.

================================================================================
CHANGE 1 - add imports at the top
================================================================================

Find:
    from src.research.research_signal import research_signal

Add below:
    from src.models.vote_extractor import signal_to_features

================================================================================
CHANGE 2 - add meta-learner loading to load_models()
================================================================================

Find this at the end of load_models():
        self.models_loaded = True
        logger.info(f"Ensemble ready | min_votes={self.min_votes} | backtest={self.backtest_mode}")

Replace with:
        # Try loading meta-learner (optional upgrade)
        self.meta_learner = None
        try:
            from src.models.meta_learner import MetaLearner
            ml = MetaLearner()
            ml.load()
            self.meta_learner = ml
            logger.info("Meta-learner loaded - using learned vote weights")
        except Exception:
            logger.info("Meta-learner not found - using vote counting (train with: python -m scripts.train_meta_learner)")

        self.models_loaded = True
        logger.info(f"Ensemble ready | min_votes={self.min_votes} | backtest={self.backtest_mode} | meta={'YES' if self.meta_learner else 'NO'}")

================================================================================
CHANGE 3 - use meta-learner in predict() for the final decision
================================================================================

Find this block in predict():
        if votes_up >= self.min_votes:
            direction = "UP"
        elif votes_down >= self.min_votes:
            direction = "DOWN"
        else:
            direction = "NO_TRADE"

Replace with:
        # Use meta-learner if available, else fall back to vote counting
        if self.meta_learner is not None:
            regime = self._get_regime(df)
            vote_features = signal_to_features(signals, regime)
            meta_result = self.meta_learner.predict_single(vote_features)
            direction = meta_result["direction"]
            avg_confidence = meta_result["confidence"]
            signals["_meta"] = meta_result  # store for logging
        else:
            if votes_up >= self.min_votes:
                direction = "UP"
            elif votes_down >= self.min_votes:
                direction = "DOWN"
            else:
                direction = "NO_TRADE"

================================================================================
THAT'S IT. 3 changes. The rest of the predict() method (regime veto, etc.)
stays exactly the same.
================================================================================

The upgrade path:
  1. Train: python -m scripts.train_meta_learner
  2. Apply these 3 edits to ensemble.py
  3. Run your backtest — compare results
  4. If meta-learner is worse, just delete models/saved/meta_learner.pkl
     and the system falls back to vote counting automatically.

Zero risk. You can always go back.
"""
