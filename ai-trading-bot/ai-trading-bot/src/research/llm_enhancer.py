"""
llm_enhancer.py
OPTIONAL Mode 2 — Uses Google Gemini free tier to generate a natural-language
morning briefing from the rule-based data.

This is a NICE-TO-HAVE, not a requirement. The rule-based bias calculator
(Mode 1) works perfectly fine on its own. This adds:
  - Human-readable morning briefing text
  - LLM-detected risk events that rules might miss
  - Nuanced interpretation of combined signals

Cost: FREE (Gemini 1.5 Flash free tier: 15 requests/minute)
Dependency: google-generativeai package (pip install google-generativeai)
Fallback: If Gemini fails, returns None and the system uses Mode 1 only.

Setup:
  1. Go to https://aistudio.google.com/app/apikey
  2. Create a free API key
  3. Add to your .env file: GEMINI_API_KEY=your_key_here
  4. That's it. Optional — system works without it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("llm_enhancer")

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_DIR = Path("data/research")


def _get_gemini_key() -> Optional[str]:
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key else None


def generate_briefing(
    market_data: dict,
    bias: dict,
    news_headlines: list[str] = None,
) -> Optional[dict]:
    """
    Generate an LLM-enhanced morning briefing.

    Args:
        market_data: output of fetch_all_market_data()
        bias: output of calculate_bias().to_dict()
        news_headlines: optional list of today's top headlines

    Returns:
        dict with 'briefing_text' and 'llm_risk_events', or None on failure.
    """
    api_key = _get_gemini_key()
    if not api_key:
        logger.info("No GEMINI_API_KEY set - LLM enhancement skipped (this is fine)")
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        logger.info("google-generativeai not installed - LLM enhancement skipped")
        logger.info("To enable: pip install google-generativeai")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        headlines_text = ""
        if news_headlines:
            headlines_text = "\n".join(f"- {h}" for h in news_headlines[:20])

        prompt = f"""You are a senior equity research analyst covering Indian markets (NSE/BSE).
Generate a concise pre-market briefing for today based on this data:

MARKET DATA:
- Nifty 50: {market_data.get('nifty_close', 'N/A')} ({market_data.get('nifty_change_pct', 'N/A')}%)
- India VIX: {market_data.get('india_vix', 'N/A')} ({market_data.get('vix_change_pct', 'N/A')}%)
- S&P 500 (overnight): {market_data.get('sp500_close', 'N/A')} ({market_data.get('sp500_change_pct', 'N/A')}%)
- Crude Oil (WTI): ${market_data.get('crude_oil', 'N/A')} ({market_data.get('crude_change_pct', 'N/A')}%)
- USD/INR: {market_data.get('usdinr', 'N/A')} ({market_data.get('usdinr_change_pct', 'N/A')}%)
- FII Net: {market_data.get('fii_net_cr', 'N/A')} Cr
- DII Net: {market_data.get('dii_net_cr', 'N/A')} Cr

RULE-BASED BIAS: {bias.get('bias_label', 'N/A')} (score: {bias.get('bias_score', 'N/A')})
RECOMMENDATION: {bias.get('recommendation', 'N/A')}

{"TODAY'S HEADLINES:" + chr(10) + headlines_text if headlines_text else "No headlines available."}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "briefing": "2-3 sentence market outlook",
  "risk_events": ["list of 0-3 specific risk events to watch today"],
  "sectors_to_watch": ["list of 1-3 sectors"],
  "agrees_with_rules": true/false,
  "llm_bias_adjustment": 0.0
}}

The llm_bias_adjustment should be between -0.2 and +0.2. Use it ONLY if the
headlines reveal something the rule-based system missed (e.g. a surprise policy
announcement). Otherwise keep it at 0.0.
"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Clean markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        # Cache the LLM response
        today = datetime.now(IST).strftime("%Y-%m-%d")
        cache_path = CACHE_DIR / f"llm_briefing_{today}.json"
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info(f"LLM briefing generated: {result.get('briefing', '')[:80]}...")
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM enhancement failed (will use rule-based only): {e}")
        return None


if __name__ == "__main__":
    # Test with sample data
    sample_market = {
        "nifty_close": 24500, "nifty_change_pct": -0.5,
        "india_vix": 16.5, "vix_change_pct": 3.2,
        "sp500_close": 5800, "sp500_change_pct": -0.8,
        "crude_oil": 72.5, "crude_change_pct": 1.2,
        "usdinr": 83.5, "usdinr_change_pct": 0.1,
        "fii_net_cr": -1500, "dii_net_cr": 920,
    }
    sample_bias = {
        "bias_label": "bearish", "bias_score": -0.35,
        "recommendation": "reduce_exposure",
    }
    result = generate_briefing(sample_market, sample_bias)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("LLM not available - rule-based mode works fine!")
