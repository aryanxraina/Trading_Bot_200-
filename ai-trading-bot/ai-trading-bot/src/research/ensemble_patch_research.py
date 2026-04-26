"""
ensemble_patch_research.py
============================
4 more small edits to ensemble.py to add the 9th vote (research/market bias).
Apply AFTER the sentiment patch you already did.

================================================================================
CHANGE 1 - add import (right below the sentiment import you already added)
================================================================================

Find:
    from src.sentiment.sentiment_signal import sentiment_signal

Add below it:
    from src.research.research_signal import research_signal

================================================================================
CHANGE 2 - add 9th vote in the signals dict
================================================================================

Find:
            "sentiment":    sentiment_signal(df),
        }

Replace with:
            "sentiment":    sentiment_signal(df),
            "research":     research_signal(df),
        }

================================================================================
CHANGE 3 - (OPTIONAL) add research to summary()
================================================================================

Find (the lines you added for sentiment):
                "sent_score": sig.model_signals.get("sentiment", {}).get("sent_score"),
                "sent_count": sig.model_signals.get("sentiment", {}).get("sent_count"),
            })

Replace with:
                "sent_score": sig.model_signals.get("sentiment", {}).get("sent_score"),
                "sent_count": sig.model_signals.get("sentiment", {}).get("sent_count"),
                "market_bias": sig.model_signals.get("research", {}).get("bias_score"),
            })

================================================================================
THAT'S IT. 3 changes. Even simpler than the sentiment patch.
================================================================================

The MIN_VOTES stays at 3. With 9 voters, requiring 3-of-9 agreement (~33%)
is a reasonable bar. You can bump to 4 later if you're overtrading.

After applying:
  1. Run: python -m scripts.run_daily_research
  2. Run: python -m scripts.run_daily_sentiment
  3. Then your ensemble has all 9 votes ready.

Daily schedule (cron / Task Scheduler):
  8:00 AM  ->  python -m scripts.run_daily_research    (market data + bias)
  8:05 AM  ->  python -m scripts.run_daily_sentiment   (news sentiment)
  9:15 AM  ->  Market opens, ensemble runs with all 9 votes
"""
