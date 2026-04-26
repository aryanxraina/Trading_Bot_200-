"""
market_data_fetcher.py (v2 - yfinance-free)
Fetches macro market data from FREE sources. No API keys. No yfinance.

Uses direct HTTP requests to:
  - NSE India website (Nifty, VIX, FII/DII)
  - Google Finance (S&P 500, Crude Oil, USD/INR)

All parsing is simple JSON/HTML — no fragile scraping libraries.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("market_data_fetcher")

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_DIR = Path("data/research")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# NSE India direct (Nifty, VIX) — most reliable for Indian data
# ---------------------------------------------------------------------------

def _nse_session() -> requests.Session:
    """Create a session with NSE cookies (required for API access)."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    # Hit main page to get cookies
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return session


def fetch_nifty50() -> Optional[dict]:
    """Nifty 50 from NSE India API."""
    try:
        session = _nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/allIndices",
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            for idx in data.get("data", []):
                if idx.get("index") == "NIFTY 50":
                    last = float(idx.get("last", 0))
                    change_pct = float(idx.get("percentChange", 0))
                    logger.info(f"Nifty 50: {last} ({change_pct:+.1f}%)")
                    return {
                        "value": round(last, 2),
                        "change_pct": round(change_pct, 2),
                        "date": datetime.now(IST).strftime("%Y-%m-%d"),
                    }
    except Exception as e:
        logger.warning(f"NSE Nifty fetch failed: {e}")
    return None


def fetch_india_vix() -> Optional[dict]:
    """India VIX from NSE India API."""
    try:
        session = _nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/allIndices",
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            for idx in data.get("data", []):
                if "VIX" in idx.get("index", "").upper():
                    last = float(idx.get("last", 0))
                    change_pct = float(idx.get("percentChange", 0))
                    logger.info(f"India VIX: {last} ({change_pct:+.1f}%)")
                    return {
                        "value": round(last, 2),
                        "change_pct": round(change_pct, 2),
                        "date": datetime.now(IST).strftime("%Y-%m-%d"),
                    }
    except Exception as e:
        logger.warning(f"NSE VIX fetch failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Google Finance (S&P 500, Crude, USD/INR) — free, no API key
# ---------------------------------------------------------------------------

def _google_finance_price(ticker: str) -> Optional[dict]:
    """
    Scrape current price from Google Finance.
    ticker format: "INDEXSP:.INX" for S&P, ".NYMEX:CL1!" for crude, etc.
    Uses the simple Google Finance page which returns data in the HTML.
    """
    try:
        url = f"https://www.google.com/finance/quote/{ticker}"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None

        text = resp.text

        # Extract price from data attribute or specific div patterns
        # Google Finance puts the price in a div with data-last-price attribute
        price_match = re.search(r'data-last-price="([0-9.,]+)"', text)
        change_match = re.search(r'data-last-price-change-percent="([0-9.,-]+)"', text)

        if price_match:
            price_str = price_match.group(1).replace(",", "")
            price = float(price_str)
            change_pct = 0.0
            if change_match:
                change_pct = float(change_match.group(1).replace(",", ""))
            return {
                "value": round(price, 2),
                "change_pct": round(change_pct, 2),
                "date": datetime.now(IST).strftime("%Y-%m-%d"),
            }
    except Exception as e:
        logger.warning(f"Google Finance {ticker} failed: {e}")
    return None


def fetch_sp500() -> Optional[dict]:
    """S&P 500 from Google Finance."""
    result = _google_finance_price(".INX:INDEXSP")
    if result:
        logger.info(f"S&P 500: {result['value']} ({result['change_pct']:+.1f}%)")
        return result
    # Fallback: try alternate ticker format
    result = _google_finance_price("SPX:INDEXSP")
    if result:
        logger.info(f"S&P 500: {result['value']} ({result['change_pct']:+.1f}%)")
    return result


def fetch_crude_oil() -> Optional[dict]:
    """WTI Crude Oil from Google Finance."""
    result = _google_finance_price("CL%3DF:NYMEX")
    if result:
        logger.info(f"Crude Oil: ${result['value']} ({result['change_pct']:+.1f}%)")
    return result


def fetch_usdinr() -> Optional[dict]:
    """USD/INR from Google Finance."""
    result = _google_finance_price("USD-INR")
    if result:
        logger.info(f"USD/INR: {result['value']} ({result['change_pct']:+.1f}%)")
    return result


# ---------------------------------------------------------------------------
# FII/DII flows from NSE
# ---------------------------------------------------------------------------

def fetch_fii_dii() -> Optional[dict]:
    """FII/DII daily flows from NSE India."""
    try:
        session = _nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            fii_net = 0.0
            dii_net = 0.0
            for row in data:
                cat = row.get("category", "").upper()
                buy = float(row.get("buyValue", 0))
                sell = float(row.get("sellValue", 0))
                if "FII" in cat or "FPI" in cat:
                    fii_net = round((buy - sell) / 100, 2)
                elif "DII" in cat:
                    dii_net = round((buy - sell) / 100, 2)
            result = {
                "fii_net": fii_net,
                "dii_net": dii_net,
                "date": datetime.now(IST).strftime("%Y-%m-%d"),
            }
            logger.info(f"FII: {fii_net:+.0f}Cr | DII: {dii_net:+.0f}Cr")
            return result
    except Exception as e:
        logger.warning(f"NSE FII/DII API failed: {e}")
    return None


# ---------------------------------------------------------------------------
# All-in-one fetcher
# ---------------------------------------------------------------------------

def fetch_all_market_data() -> dict:
    """
    Fetch all macro data points. Returns a flat dict.
    Missing values are None (the bias calculator handles this gracefully).
    """
    logger.info("Fetching market data...")

    nifty = fetch_nifty50()
    vix = fetch_india_vix()
    sp500 = fetch_sp500()
    crude = fetch_crude_oil()
    usdinr = fetch_usdinr()
    fii_dii = fetch_fii_dii()

    result = {
        "fetch_time": datetime.now(IST).isoformat(),
        "nifty_close": nifty["value"] if nifty else None,
        "nifty_change_pct": nifty["change_pct"] if nifty else None,
        "india_vix": vix["value"] if vix else None,
        "vix_change_pct": vix["change_pct"] if vix else None,
        "sp500_close": sp500["value"] if sp500 else None,
        "sp500_change_pct": sp500["change_pct"] if sp500 else None,
        "crude_oil": crude["value"] if crude else None,
        "crude_change_pct": crude["change_pct"] if crude else None,
        "usdinr": usdinr["value"] if usdinr else None,
        "usdinr_change_pct": usdinr["change_pct"] if usdinr else None,
        "fii_net_cr": fii_dii["fii_net"] if fii_dii else None,
        "dii_net_cr": fii_dii["dii_net"] if fii_dii else None,
    }

    # Count how many data points we got
    available = sum(1 for k, v in result.items()
                    if k not in ("fetch_time",) and v is not None)
    logger.info(f"Got {available}/12 data points")

    # Cache it
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    cache_path = CACHE_DIR / f"market_data_{today}.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Market data cached: {cache_path}")

    return result


def load_cached_market_data(date_str: str = None) -> Optional[dict]:
    """Load cached market data for a date. Falls back to latest available."""
    if date_str is None:
        date_str = datetime.now(IST).strftime("%Y-%m-%d")
    cache_path = CACHE_DIR / f"market_data_{date_str}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    for days_back in range(1, 4):
        prev = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
        prev_path = CACHE_DIR / f"market_data_{prev}.json"
        if prev_path.exists():
            with prev_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


if __name__ == "__main__":
    data = fetch_all_market_data()
    print("\n=== Market Data ===")
    for k, v in data.items():
        print(f"  {k:>20s}: {v}")