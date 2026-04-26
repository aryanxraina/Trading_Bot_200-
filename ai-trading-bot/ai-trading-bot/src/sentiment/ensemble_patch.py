"""
ensemble_patch.py
==================
This is NOT a runnable file. It documents the EXACT 4 changes you need to
make in src/models/ensemble.py to plug in the 8th sentiment vote.

Apply these edits, then restart your backtest notebook (notebook 05).
The voting math is unchanged — sentiment becomes one more vote that needs
confidence >= CONFIDENCE_THRESHOLD to count.

================================================================================
CHANGE 1 — add import at the top of ensemble.py
================================================================================

Locate the import block near the top, after this line:
    from src.data.preprocess import get_feature_columns

ADD this line:
    from src.sentiment.sentiment_signal import sentiment_signal

================================================================================
CHANGE 2 — register sentiment in the signals dict inside predict()
================================================================================

Find this block in EnsembleModel.predict():

    signals = {
        "xgboost":      self._xgb_signal(df),
        "lstm":         self._lstm_signal(df),
        "rsi":          self._rsi_signal(df),
        "macd":         self._macd_signal(df),
        "volume":       self._volume_signal(df),
        "trend_filter": self._trend_filter_signal(df),
        "nifty":        self._nifty_signal(df),
    }

REPLACE WITH:

    signals = {
        "xgboost":      self._xgb_signal(df),
        "lstm":         self._lstm_signal(df),
        "rsi":          self._rsi_signal(df),
        "macd":         self._macd_signal(df),
        "volume":       self._volume_signal(df),
        "trend_filter": self._trend_filter_signal(df),
        "nifty":        self._nifty_signal(df),
        "sentiment":    sentiment_signal(df),     # ← NEW 8th vote
    }

================================================================================
CHANGE 3 — (OPTIONAL but recommended) tighten min_votes from 2 to 3
================================================================================

You're now generating 8 votes. With min_votes=2 you were already overtrading
(noted in the progress report). With sentiment added, bump the threshold:

OLD:
    MIN_VOTES = 2

NEW:
    MIN_VOTES = 3

Rationale: at 8 voters, requiring 3-of-8 agreement is still a low bar (~37.5%)
but materially reduces noise trades. Re-tune with backtest.

================================================================================
CHANGE 4 — (OPTIONAL) add sentiment to summary() for monitoring
================================================================================

Inside summary(), find this dict and add the sentiment row at the bottom:

    rows.append({
        "symbol": symbol,
        ...
        "rsi": sig.model_signals.get("rsi", {}).get("rsi"),
        "sent_score":  sig.model_signals.get("sentiment", {}).get("sent_score"),  # ← NEW
        "sent_count":  sig.model_signals.get("sentiment", {}).get("sent_count"),  # ← NEW
    })

================================================================================
THAT'S IT. NO OTHER CHANGES.
================================================================================

The voting/veto math automatically picks up the new vote because predict() loops
over signals.values() — no hardcoded list of 7 anywhere.

After applying:
  1. Run python -m scripts.run_daily_sentiment to populate sentiment data
  2. Re-run notebook 05 (ensemble backtest) — but on top of the new
     *_features_with_sent.csv files. You'll need to update the backtest's
     preprocess_symbol() call to read those.
  3. Compare backtest results: same 47 stocks, with vs without sentiment vote.
     If sentiment is helping, you'll see:
       - Win rate up by 1-3 percentage points
       - Trade count DOWN (sentiment abstains often → fewer trades)
       - Sharpe up

If you DON'T see those, sentiment is either hurting or your news coverage is
too sparse. The fix is more sources, not retraining FinBERT.
"""
