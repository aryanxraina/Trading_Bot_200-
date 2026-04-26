"""
test_sentiment.py
Lightweight tests. No FinBERT download needed for most of these.

Run with:
    python -m tests.test_sentiment
"""

from datetime import datetime, timedelta, timezone
from src.sentiment.news_scraper import detect_stocks, _hash_url
from src.sentiment.sentiment_aggregator import (
    aggregate, to_dataframe, _signed_contribution, _decay_weight
)
from src.sentiment.sentiment_signal import sentiment_signal
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))

PASS = "✅"
FAIL = "❌"


def test(name: str, cond: bool) -> None:
    print(f"  {PASS if cond else FAIL} {name}")
    assert cond, name


# ─────────────────────────────────────────────────────────────────────────────
# Stock detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Stock Detection]")
test("Detects Reliance",     "RELIANCE" in detect_stocks("Reliance Industries reports Q4 results"))
test("Detects RIL alias",    "RELIANCE" in detect_stocks("RIL stock hits new high"))
test("Detects HDFC Bank",    "HDFCBANK" in detect_stocks("HDFC Bank reports strong NII growth"))
test("No false positive",    "ITC" not in detect_stocks("Witcher game announcement"))
test("Empty input safe",     detect_stocks("") == [])
test("Multi-stock headline", set(detect_stocks("Infosys and TCS both beat estimates")) == {"INFY", "TCS"})


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator math
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Aggregator]")

# Signed contribution
test("Pure positive →  +pos",    abs(_signed_contribution({"positive": 0.9, "negative": 0.05, "neutral": 0.05}) - 0.85) < 1e-9)
test("Pure negative →  -neg",    abs(_signed_contribution({"positive": 0.05, "negative": 0.85, "neutral": 0.10}) - (-0.80)) < 1e-9)
test("Pure neutral  →   0",      abs(_signed_contribution({"positive": 0.1, "negative": 0.1, "neutral": 0.8})) < 1e-9)

# Decay weight
ref = datetime(2026, 4, 26, 23, 59, tzinfo=IST)
now = datetime(2026, 4, 26, 12, 0, tzinfo=IST).isoformat()
yesterday = datetime(2026, 4, 25, 23, 59, tzinfo=IST).isoformat()
test("Recent decay close to 1", _decay_weight(now, ref, 24.0) > 0.65)
test("24h ago ≈ 0.5",            abs(_decay_weight(yesterday, ref, 24.0) - 0.5) < 0.05)

# End-to-end aggregate
sample = [
    {"stocks": ["INFY"],
     "probs": {"positive": 0.85, "negative": 0.05, "neutral": 0.10},
     "label": "positive",
     "published": datetime(2026, 4, 26, 10, 0, tzinfo=IST).isoformat(),
     "source": "moneycontrol"},
    {"stocks": ["INFY"],
     "probs": {"positive": 0.80, "negative": 0.10, "neutral": 0.10},
     "label": "positive",
     "published": datetime(2026, 4, 26, 14, 0, tzinfo=IST).isoformat(),
     "source": "et_markets"},
    {"stocks": ["RELIANCE"],
     "probs": {"positive": 0.10, "negative": 0.85, "neutral": 0.05},
     "label": "negative",
     "published": datetime(2026, 4, 26, 11, 0, tzinfo=IST).isoformat(),
     "source": "moneycontrol"},
]
rows = aggregate(sample, ref_date="2026-04-26")
infy = next(r for r in rows if r.symbol == "INFY")
ril  = next(r for r in rows if r.symbol == "RELIANCE")
test("INFY positive sentiment",     infy.sent_score > 0.5)
test("INFY reliable (2 headlines)", infy.reliable is True)
test("RELIANCE negative",           ril.sent_score < -0.5)
test("RELIANCE not reliable (1)",   ril.reliable is False)


# ─────────────────────────────────────────────────────────────────────────────
# Vote function
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Sentiment Vote]")

def make_df(score, count, reliable):
    return pd.DataFrame([{
        "sent_score": score, "sent_strength": abs(score),
        "sent_count": count, "sent_reliable": reliable
    }])

vote = sentiment_signal(make_df(0.6, 5, True))
test("Strong positive → UP",    vote["vote"] == "UP" and vote["confidence"] > 0.6)

vote = sentiment_signal(make_df(-0.5, 4, True))
test("Strong negative → DOWN",  vote["vote"] == "DOWN")

vote = sentiment_signal(make_df(0.1, 5, True))
test("Weak score → abstain",    vote["vote"] is None and vote["available"] is True)

vote = sentiment_signal(make_df(0.8, 1, False))
test("1 headline → abstain",    vote["vote"] is None and "insufficient" in vote["reason"])

vote = sentiment_signal(pd.DataFrame([{"open": 100}]))
test("Missing cols → unavailable", vote["available"] is False)

# Confidence cap
vote = sentiment_signal(make_df(1.0, 10, True))
test("Conf capped at 0.75",     vote["confidence"] <= 0.75)


# ─────────────────────────────────────────────────────────────────────────────
# Hashing determinism
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Misc]")
test("Same URL same hash",      _hash_url("https://x.com/a") == _hash_url("https://x.com/a"))
test("Different URL diff hash", _hash_url("https://x.com/a") != _hash_url("https://x.com/b"))


print("\n" + "=" * 50)
print(f"  All tests passed {PASS}")
print("=" * 50)
