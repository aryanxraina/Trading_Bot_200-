"""
run_daily_research.py
End-to-end morning research pipeline:
  1. Fetch macro market data (VIX, FII, S&P, crude, Nifty, USD/INR)
  2. Calculate rule-based market bias
  3. (Optional) Enhance with Gemini LLM briefing
  4. Save bias to cache for ensemble's research_signal to read

Run daily at 8:00 AM IST, BEFORE run_daily_sentiment.py.
Takes ~10-15 seconds (mostly yfinance calls).

Usage:
    python -m scripts.run_daily_research
    python -m scripts.run_daily_research --no-llm    # skip Gemini even if configured
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.research.market_data_fetcher import fetch_all_market_data, load_cached_market_data
from src.research.market_bias_calculator import calculate_bias
from src.utils.logger import get_logger

logger = get_logger("run_daily_research")

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_DIR = Path("data/research")


def main(use_llm: bool = True, use_cached_data: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(IST).strftime("%Y-%m-%d")

    # Step 1 - Fetch market data
    print("=" * 50)
    print(f"  RESEARCH AGENT - {today}")
    print("=" * 50)

    if use_cached_data:
        market_data = load_cached_market_data()
        if market_data is None:
            print("\nNo cached data found. Fetching fresh...")
            market_data = fetch_all_market_data()
    else:
        print("\n[1/3] Fetching market data...")
        market_data = fetch_all_market_data()

    # Print summary
    print(f"\n  Nifty 50:    {market_data.get('nifty_close', 'N/A')} ({market_data.get('nifty_change_pct', 'N/A')}%)")
    print(f"  India VIX:   {market_data.get('india_vix', 'N/A')} ({market_data.get('vix_change_pct', 'N/A')}%)")
    print(f"  S&P 500:     {market_data.get('sp500_close', 'N/A')} ({market_data.get('sp500_change_pct', 'N/A')}%)")
    print(f"  Crude Oil:   ${market_data.get('crude_oil', 'N/A')} ({market_data.get('crude_change_pct', 'N/A')}%)")
    print(f"  USD/INR:     {market_data.get('usdinr', 'N/A')} ({market_data.get('usdinr_change_pct', 'N/A')}%)")
    print(f"  FII Net:     {market_data.get('fii_net_cr', 'N/A')} Cr")
    print(f"  DII Net:     {market_data.get('dii_net_cr', 'N/A')} Cr")

    # Step 2 - Calculate bias
    print("\n[2/3] Calculating market bias...")
    bias = calculate_bias(market_data)
    bias_dict = bias.to_dict()

    # Save bias for research_signal.py to read
    bias_path = CACHE_DIR / f"market_bias_{today}.json"
    with bias_path.open("w", encoding="utf-8") as f:
        json.dump(bias_dict, f, indent=2)

    # Print bias
    label_emoji = {"bullish": "+", "bearish": "-", "neutral": "~"}
    emoji = label_emoji.get(bias.bias_label, "?")
    print(f"\n  MARKET BIAS: [{emoji}] {bias.bias_label.upper()} (score: {bias.bias_score:+.3f})")
    print(f"  Confidence:  {bias.confidence} ({bias.factors_available}/6 factors)")
    print(f"  Action:      {bias.recommendation}")
    print(f"\n  Factor breakdown:")
    print(f"    FII flow:    {bias.fii_signal:+.2f}  (weight 30%)")
    print(f"    VIX:         {bias.vix_signal:+.2f}  (weight 20%)")
    print(f"    US market:   {bias.us_market_signal:+.2f}  (weight 20%)")
    print(f"    Crude oil:   {bias.crude_signal:+.2f}  (weight 10%)")
    print(f"    Nifty:       {bias.nifty_signal:+.2f}  (weight 10%)")
    print(f"    Rupee:       {bias.rupee_signal:+.2f}  (weight 10%)")

    # Step 3 - Optional LLM enhancement
    if use_llm:
        print("\n[3/3] LLM enhancement...")
        try:
            from src.research.llm_enhancer import generate_briefing

            # Load today's news headlines for context
            headlines = []
            news_file = Path("data/sentiment/news_raw.jsonl")
            if news_file.exists():
                import json as jl
                with news_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = jl.loads(line)
                            if today in item.get("published", ""):
                                headlines.append(item.get("title", ""))
                        except (ValueError, KeyError):
                            continue

            result = generate_briefing(market_data, bias_dict, headlines[:20])
            if result:
                print(f"\n  LLM Briefing: {result.get('briefing', 'N/A')}")
                risks = result.get("risk_events", [])
                if risks:
                    print(f"  Risk events:  {', '.join(risks)}")
                sectors = result.get("sectors_to_watch", [])
                if sectors:
                    print(f"  Sectors:      {', '.join(sectors)}")

                # Apply LLM bias adjustment if present
                adj = float(result.get("llm_bias_adjustment", 0.0))
                if abs(adj) > 0.01:
                    old_score = bias_dict["bias_score"]
                    new_score = max(-1.0, min(1.0, old_score + adj))
                    bias_dict["bias_score"] = round(new_score, 4)
                    bias_dict["llm_adjusted"] = True
                    bias_dict["llm_adjustment"] = adj
                    # Re-save with adjustment
                    with bias_path.open("w", encoding="utf-8") as f:
                        json.dump(bias_dict, f, indent=2)
                    print(f"\n  LLM adjusted bias: {old_score:+.3f} -> {new_score:+.3f} (adj: {adj:+.2f})")
            else:
                print("  LLM not available - using rule-based bias only (this is fine)")
        except ImportError:
            print("  LLM module not available - using rule-based bias only")
    else:
        print("\n[3/3] LLM enhancement: skipped (--no-llm)")

    print("\n" + "=" * 50)
    print(f"  Research complete. Bias saved to {bias_path.name}")
    print(f"  Ensemble's research_signal will read this automatically.")
    print("=" * 50)

    return bias_dict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-llm", action="store_true",
                   help="Skip Gemini LLM enhancement, use rule-based only.")
    p.add_argument("--cached", action="store_true",
                   help="Use cached market data instead of fetching fresh.")
    args = p.parse_args()
    main(use_llm=not args.no_llm, use_cached_data=args.cached)
