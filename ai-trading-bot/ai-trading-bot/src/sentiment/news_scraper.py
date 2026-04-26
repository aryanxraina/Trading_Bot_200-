"""
news_scraper.py
Pulls financial news from free RSS feeds. No API keys required.

Sources:
  - Moneycontrol Markets RSS
  - Economic Times Markets RSS
  - LiveMint Markets RSS
  - Business Standard Markets RSS

Each headline is normalized to:
  {
    "id":        <sha1 hash of url>,        # for caching/dedup
    "source":    "moneycontrol" | "et" | ...,
    "title":     "Infosys Q4 results beat estimates...",
    "summary":   "<short blurb if RSS provides one>",
    "url":       "https://...",
    "published": "2026-04-26T08:30:00+05:30",  # ISO 8601 IST
    "stocks":    ["INFY", "TCS"]              # detected via symbol matcher
  }

Designed to be polite: retries with backoff, respects 429s, caches feed bytes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

import feedparser
import requests
from dateutil import parser as dtparser

from src.utils.logger import get_logger

logger = get_logger("news_scraper")

# Free RSS feeds — verified working as of build time.
# If any of these break, swap them out — the rest of the pipeline doesn't care.
FEEDS: dict[str, str] = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "et_markets":   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks":    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint":     "https://www.livemint.com/rss/markets",
    "bs_markets":   "https://www.business-standard.com/rss/markets-106.rss",
}

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Polite scraping
USER_AGENT = "Mozilla/5.0 (compatible; AITradingBot/1.0; sentiment-agent)"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


@dataclass
class NewsItem:
    id: str
    source: str
    title: str
    summary: str
    url: str
    published: str  # ISO 8601
    stocks: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Symbol detection
# ─────────────────────────────────────────────────────────────────────────────

# Map of NSE symbol → list of aliases the headline might use.
# Keep this small and high-precision. False positives are worse than misses
# because they'll dilute the per-stock sentiment score.
SYMBOL_ALIASES: dict[str, list[str]] = {
    "RELIANCE":  ["Reliance Industries", "RIL", "Reliance"],
    "TCS":       ["TCS", "Tata Consultancy"],
    "INFY":      ["Infosys", "Infy"],
    "HDFCBANK":  ["HDFC Bank"],
    "ICICIBANK": ["ICICI Bank"],
    "SBIN":      ["State Bank of India", "SBI"],
    "AXISBANK":  ["Axis Bank"],
    "KOTAKBANK": ["Kotak Mahindra", "Kotak Bank"],
    "ITC":       ["ITC Ltd", "ITC "],  # trailing space avoids matching "Witcher" etc.
    "HINDUNILVR":["Hindustan Unilever", "HUL"],
    "LT":        ["Larsen & Toubro", "L&T"],
    "MARUTI":    ["Maruti Suzuki", "Maruti"],
    "BAJFINANCE":["Bajaj Finance"],
    "BAJAJFINSV":["Bajaj Finserv"],
    "ASIANPAINT":["Asian Paints"],
    "TITAN":     ["Titan Company", "Titan "],
    "WIPRO":     ["Wipro"],
    "HCLTECH":   ["HCL Tech", "HCL Technologies"],
    "TECHM":     ["Tech Mahindra"],
    "SUNPHARMA": ["Sun Pharma"],
    "DRREDDY":   ["Dr Reddy", "Dr. Reddy"],
    "CIPLA":     ["Cipla"],
    "TATAMOTORS":["Tata Motors"],
    "TATASTEEL": ["Tata Steel"],
    "JSWSTEEL":  ["JSW Steel"],
    "HINDALCO":  ["Hindalco"],
    "VEDL":      ["Vedanta"],
    "ONGC":      ["ONGC", "Oil and Natural Gas"],
    "BPCL":      ["BPCL", "Bharat Petroleum"],
    "IOC":       ["Indian Oil", "IOC "],
    "COALINDIA": ["Coal India"],
    "NTPC":      ["NTPC"],
    "POWERGRID": ["Power Grid", "PowerGrid"],
    "ADANIPORTS":["Adani Ports"],
    "GRASIM":    ["Grasim"],
    "ULTRACEMCO":["UltraTech Cement", "UltraTech"],
    "SHREECEM":  ["Shree Cement"],
    "BHARTIARTL":["Bharti Airtel", "Airtel"],
    "NESTLEIND": ["Nestle India", "Nestle"],
    "BRITANNIA": ["Britannia"],
    "EICHERMOT": ["Eicher Motors", "Royal Enfield"],
    "HEROMOTOCO":["Hero MotoCorp", "Hero Moto"],
    "BAJAJ-AUTO":["Bajaj Auto"],
    "MM":        ["Mahindra & Mahindra", "M&M"],
    "INDUSINDBK":["IndusInd Bank"],
    "GAIL":      ["GAIL"],
    "UPL":       ["UPL Ltd", "UPL "],
    "ZEEL":      ["Zee Entertainment", "Zee Ent"],
}


def detect_stocks(text: str) -> list[str]:
    """Find which Nifty 50 symbols are mentioned in a headline.
    Case-insensitive substring match against curated aliases."""
    if not text:
        return []
    text_lower = text.lower()
    hits: list[str] = []
    for symbol, aliases in SYMBOL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                hits.append(symbol)
                break
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_with_retry(url: str) -> bytes | None:
    """GET a URL with retries and backoff. Returns bytes or None."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** (attempt + 2)
                logger.warning(f"Rate limited on {url} — waiting {wait}s")
                time.sleep(wait)
                continue
            logger.warning(f"HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"Fetch error on {url}: {e} — retry in {wait}s")
            time.sleep(wait)
    logger.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts")
    return None


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _parse_published(entry: dict) -> str:
    """Try several fields, fall back to now()."""
    for field in ("published", "updated", "pubDate", "created"):
        val = entry.get(field)
        if val:
            try:
                dt = dtparser.parse(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST)
                return dt.astimezone(IST).isoformat()
            except (ValueError, TypeError):
                continue
    return datetime.now(IST).isoformat()


def fetch_feed(source: str, url: str) -> list[NewsItem]:
    """Fetch and parse a single RSS feed."""
    raw = _fetch_with_retry(url)
    if raw is None:
        return []
    parsed = feedparser.parse(raw)
    items: list[NewsItem] = []
    for entry in parsed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue
        link = entry.get("link", "").strip()
        if not link:
            continue
        summary = entry.get("summary", "").strip()
        # Strip HTML tags from summary cheaply
        if "<" in summary:
            import re
            summary = re.sub(r"<[^>]+>", "", summary)
        published = _parse_published(entry)
        full_text = f"{title} {summary}"
        stocks = detect_stocks(full_text)

        items.append(NewsItem(
            id=_hash_url(link),
            source=source,
            title=title[:500],       # cap length
            summary=summary[:1000],
            url=link,
            published=published,
            stocks=stocks,
        ))
    logger.info(f"{source}: {len(items)} items ({sum(1 for i in items if i.stocks)} with detected stocks)")
    return items


def fetch_all(feeds: dict[str, str] | None = None) -> list[NewsItem]:
    """Fetch every configured feed, dedupe by URL hash."""
    feeds = feeds or FEEDS
    seen: set[str] = set()
    out: list[NewsItem] = []
    for source, url in feeds.items():
        for item in fetch_feed(source, url):
            if item.id in seen:
                continue
            seen.add(item.id)
            out.append(item)
    logger.info(f"Total unique items: {len(out)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence (append-only JSONL)
# ─────────────────────────────────────────────────────────────────────────────

def save_jsonl(items: Iterable[NewsItem], path: str | Path) -> int:
    """Append items to a JSONL file. Skips dupes by id within the same run."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if path.exists():
        # Load existing IDs to avoid double-writing
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except (KeyError, ValueError):
                    continue
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            if it.id in seen:
                continue
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
            seen.add(it.id)
            written += 1
    logger.info(f"Wrote {written} new items to {path}")
    return written


if __name__ == "__main__":
    # Quick smoke test
    items = fetch_all()
    print(f"Fetched {len(items)} headlines")
    for it in items[:5]:
        print(f"  [{it.source}] {it.title[:80]}  → {it.stocks}")
    save_jsonl(items, "data/sentiment/news_raw.jsonl")
