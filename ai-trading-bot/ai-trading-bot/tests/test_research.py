"""
test_research.py
Tests for the Research Agent. No API calls needed.

Run with:
    python -m tests.test_research
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

# We need to set up a minimal logger before importing
import logging
logging.basicConfig(level=logging.WARNING)

from src.research.market_bias_calculator import (
    calculate_bias, _fii_signal, _vix_signal, _us_market_signal,
    _crude_signal, _nifty_signal, _rupee_signal
)
from src.research.research_signal import research_signal

IST = timezone(timedelta(hours=5, minutes=30))
PASS = "+"
FAIL = "X"


def test(name, cond):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}")
    assert cond, f"FAILED: {name}"


# -------------------------------------------------------------------
# Individual factor signals
# -------------------------------------------------------------------
print("\n[FII Signal]")
test("Strong buying -> +1.0",     _fii_signal(3000) == 1.0)
test("Moderate buying -> +0.5",   _fii_signal(800) == 0.5)
test("Small buying -> +0.2",      _fii_signal(200) == 0.2)
test("Small selling -> -0.2",     _fii_signal(-500) == -0.2)
test("Heavy selling -> -1.0",     _fii_signal(-4000) == -1.0)
test("None -> 0.0",               _fii_signal(None) == 0.0)

print("\n[VIX Signal]")
test("Low VIX (12) -> bullish",   _vix_signal(12) == 0.6)
test("Normal VIX (16) -> 0",      _vix_signal(16) == 0.0)
test("High VIX (22) -> bearish",  _vix_signal(22) == -0.6)
test("Panic VIX (30) -> -1.0",    _vix_signal(30) == -1.0)
test("None -> 0.0",               _vix_signal(None) == 0.0)

print("\n[US Market Signal]")
test("Strong rally -> +1.0",      _us_market_signal(2.0) == 1.0)
test("Moderate up -> +0.5",       _us_market_signal(0.8) == 0.5)
test("Flat -> 0.0",               _us_market_signal(0.1) == 0.0)
test("Moderate down -> -0.5",     _us_market_signal(-0.8) == -0.5)
test("Crash -> -1.0",             _us_market_signal(-2.0) == -1.0)

print("\n[Crude Signal]")
test("Crude spike -> bearish",    _crude_signal(4.0) == -0.8)
test("Crude drop -> bullish",     _crude_signal(-4.0) == 0.5)
test("Small move -> 0",           _crude_signal(0.5) == 0.0)

print("\n[Nifty Signal]")
test("Strong up -> +0.5",         _nifty_signal(1.5) == 0.5)
test("Strong down -> -0.5",       _nifty_signal(-1.5) == -0.5)
test("Flat -> 0",                 _nifty_signal(0.1) == 0.0)

print("\n[Rupee Signal]")
test("Rupee weak -> bearish",     _rupee_signal(0.6) == -0.7)
test("Rupee strong -> bullish",   _rupee_signal(-0.5) == 0.3)
test("Flat -> 0",                 _rupee_signal(0.05) == 0.0)


# -------------------------------------------------------------------
# Full bias calculation
# -------------------------------------------------------------------
print("\n[Full Bias - Bullish scenario]")
bullish_data = {
    "fii_net_cr": 2500,
    "india_vix": 12,
    "sp500_change_pct": 1.0,
    "crude_change_pct": -2.0,
    "nifty_change_pct": 0.8,
    "usdinr_change_pct": -0.4,
}
bias = calculate_bias(bullish_data)
test("Label is bullish",           bias.bias_label == "bullish")
test("Score > 0.3",                bias.bias_score > 0.3)
test("Confidence is high",        bias.confidence == "high")
test("6 factors available",       bias.factors_available == 6)
test("Rec is increase_exposure",  bias.recommendation == "increase_exposure")

print("\n[Full Bias - Bearish scenario]")
bearish_data = {
    "fii_net_cr": -3500,
    "india_vix": 25,
    "sp500_change_pct": -2.0,
    "crude_change_pct": 4.0,
    "nifty_change_pct": -1.5,
    "usdinr_change_pct": 0.8,
}
bias = calculate_bias(bearish_data)
test("Label is bearish",          bias.bias_label == "bearish")
test("Score < -0.5",              bias.bias_score < -0.5)
test("Rec is reduce_exposure",    bias.recommendation == "reduce_exposure")

print("\n[Full Bias - Missing data]")
sparse_data = {
    "india_vix": 15,
    "sp500_change_pct": 0.3,
}
bias = calculate_bias(sparse_data)
test("Label is neutral",          bias.bias_label == "neutral")
test("Only 2 factors",            bias.factors_available == 2)
test("Confidence is low",         bias.confidence == "low")

print("\n[Full Bias - All None]")
empty_data = {}
bias = calculate_bias(empty_data)
test("Score is 0.0",              bias.bias_score == 0.0)
test("Label neutral",             bias.bias_label == "neutral")
test("0 factors",                 bias.factors_available == 0)


# -------------------------------------------------------------------
# Research signal (vote function)
# -------------------------------------------------------------------
print("\n[Research Signal - No cache file]")
# Temporarily point cache to empty dir
import src.research.research_signal as rs
original_cache = rs.CACHE_DIR
rs.CACHE_DIR = Path(tempfile.mkdtemp())

vote = research_signal()
test("No file -> not available",   vote["available"] is False)
test("Reason is no_bias_file",     vote["reason"] == "no_bias_file")

print("\n[Research Signal - Bullish bias file]")
today = datetime.now(IST).strftime("%Y-%m-%d")
bias_file = rs.CACHE_DIR / f"market_bias_{today}.json"
with bias_file.open("w") as f:
    json.dump({"bias_score": 0.45, "bias_label": "bullish",
               "confidence": "high", "factors_available": 5,
               "recommendation": "increase_exposure"}, f)
vote = research_signal()
test("Vote is UP",                 vote["vote"] == "UP")
test("Confidence > 0.54",         vote["confidence"] > 0.54)
test("Available",                  vote["available"] is True)

print("\n[Research Signal - Bearish bias file]")
with bias_file.open("w") as f:
    json.dump({"bias_score": -0.55, "bias_label": "bearish",
               "confidence": "high", "factors_available": 6,
               "recommendation": "reduce_exposure"}, f)
vote = research_signal()
test("Vote is DOWN",               vote["vote"] == "DOWN")
test("Confidence > 0.54",         vote["confidence"] > 0.54)

print("\n[Research Signal - Weak bias]")
with bias_file.open("w") as f:
    json.dump({"bias_score": 0.08, "bias_label": "neutral",
               "confidence": "medium", "factors_available": 4,
               "recommendation": "hold"}, f)
vote = research_signal()
test("Weak bias -> abstain",       vote["vote"] is None)
test("Reason: below_threshold",   vote["reason"] == "below_threshold")

print("\n[Research Signal - Low confidence]")
with bias_file.open("w") as f:
    json.dump({"bias_score": 0.50, "bias_label": "bullish",
               "confidence": "low", "factors_available": 2,
               "recommendation": "hold"}, f)
vote = research_signal()
test("Low conf -> abstain",        vote["vote"] is None)
test("Reason: low_confidence",    vote["reason"] == "low_confidence")

# Restore original cache dir
rs.CACHE_DIR = original_cache


print("\n" + "=" * 50)
print(f"  All tests passed [{PASS}]")
print("=" * 50)
