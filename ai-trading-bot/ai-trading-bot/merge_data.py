"""
merge_data.py
Merges your existing NSE bhavcopy data (2000-2021) with the new
Kaggle intraday dataset (2015-2026) to create complete daily OHLCV files.

What this script does:
1. Reads each *_minute.csv from archive (1) folder
2. Resamples 1-min → daily OHLCV
3. Merges with existing 2000-2021 bhavcopy data
4. Saves complete 2000→2026 daily CSVs to data/raw/
5. Also copies 1-min files to data/intraday/ for Phase 3 use

Run from project root:
    python merge_data.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
ARCHIVE_OLD = r"data\raw"           # existing 2000-2021 bhavcopy CSVs
ARCHIVE_NEW = r"C:\Users\Aryan\Desktop\everything\PROJECTS\200%BOT\ai-trading-bot\data\archive (1)"
OUTPUT_DAILY = r"data\raw"          # final merged daily CSVs go here
OUTPUT_INTRADAY = r"data\intraday"  # 1-min files for Phase 3

# Map from archive (1) filename suffix to your existing symbol names
# e.g. AXISBANK_minute.csv → AXISBANK
# Some names differ between datasets — handle those here
NAME_MAP = {
    "MM": "MM",           # M&M in new data might be MM
    "MAHINDRA": "MM",
    "M&M": "MM",
}


def load_existing(symbol: str) -> pd.DataFrame | None:
    """Load existing bhavcopy daily data for a symbol."""
    path = os.path.join(ARCHIVE_OLD, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        # Standardise column names
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume"
        }
        df = df.rename(columns=col_map)
        keep = ["open", "high", "low", "close", "volume"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "close"])
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"  ⚠️  Failed to load existing {symbol}: {e}")
        return None


def load_intraday(filepath: str) -> pd.DataFrame | None:
    """Load a *_minute.csv file and resample to daily OHLCV."""
    try:
        df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
        df.columns = [c.lower().strip() for c in df.columns]

        # Keep only OHLCV
        keep = ["open", "high", "low", "close", "volume"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "close"])
        df = df.sort_index()

        # Resample 1-min → daily
        daily = df.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna(subset=["open", "close"])

        # Keep only trading days (remove weekends with no data)
        daily = daily[daily["open"].notna() & (daily["volume"] > 0)]

        return daily

    except Exception as e:
        print(f"  ⚠️  Failed to load intraday file: {e}")
        return None


def merge_dataframes(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge old (2000-2021) and new (2015-2026) daily data.
    New data takes priority for overlapping dates.
    """
    if old_df is None and new_df is None:
        return None
    if old_df is None:
        return new_df
    if new_df is None:
        return old_df

    # Find the cutover point — use old data before new data starts
    new_start = new_df.index.min()

    # Keep old data only before new data starts (avoid duplicates)
    old_trimmed = old_df[old_df.index < new_start]

    # Combine
    combined = pd.concat([old_trimmed, new_df], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    return combined


def save_daily(df: pd.DataFrame, symbol: str):
    """Save merged daily CSV to data/raw/."""
    os.makedirs(OUTPUT_DAILY, exist_ok=True)
    path = os.path.join(OUTPUT_DAILY, f"{symbol}.csv")
    df.index.name = "Date"
    df.to_csv(path)


def copy_intraday(filepath: str, symbol: str):
    """Copy 1-min file to data/intraday/ for Phase 3 use."""
    os.makedirs(OUTPUT_INTRADAY, exist_ok=True)
    dest = os.path.join(OUTPUT_INTRADAY, f"{symbol}_1min.csv")
    if not os.path.exists(dest):
        import shutil
        shutil.copy2(filepath, dest)


def get_symbol_from_filename(filename: str) -> str:
    """Extract symbol from filename like RELIANCE_minute.csv → RELIANCE."""
    name = filename.replace("_minute.csv", "").replace(".csv", "")
    return NAME_MAP.get(name, name)


def main():
    print("=" * 60)
    print("AI Trading Bot — Data Merger")
    print("=" * 60)

    # Get existing symbols
    existing_symbols = set()
    if os.path.exists(ARCHIVE_OLD):
        for f in os.listdir(ARCHIVE_OLD):
            if f.endswith(".csv") and not f.endswith(".NS.csv"):
                sym = f.replace(".csv", "")
                skip = {"INFRATEL", "NIFTY50_all", "stock_metadata"}
                if sym not in skip:
                    existing_symbols.add(sym)

    print(f"Existing symbols in data/raw: {len(existing_symbols)}")

    # Get new intraday files
    if not os.path.exists(ARCHIVE_NEW):
        print(f"\n❌ Archive folder not found: {ARCHIVE_NEW}")
        print("Please check the path and try again.")
        return

    new_files = [f for f in os.listdir(ARCHIVE_NEW)
                 if f.endswith(".csv") and "_minute" in f]
    print(f"New intraday files found: {len(new_files)}")

    # Find matching symbols
    matched = []
    for f in new_files:
        symbol = get_symbol_from_filename(f)
        if symbol in existing_symbols:
            matched.append((symbol, f))

    unmatched_new = [get_symbol_from_filename(f) for f in new_files
                     if get_symbol_from_filename(f) not in existing_symbols]

    print(f"Matched symbols (will merge): {len(matched)}")
    print(f"New-only symbols (will add as daily): {len(unmatched_new)}")
    print()

    success, failed = [], []

    # ── Process matched symbols (merge old + new)
    print("── Merging matched symbols...")
    for symbol, filename in matched:
        filepath = os.path.join(ARCHIVE_NEW, filename)
        print(f"  {symbol}...", end=" ")

        old_df = load_existing(symbol)
        new_daily = load_intraday(filepath)

        merged = merge_dataframes(old_df, new_daily)

        if merged is not None and len(merged) > 100:
            save_daily(merged, symbol)
            copy_intraday(filepath, symbol)
            old_end = old_df.index.max().date() if old_df is not None else "N/A"
            new_end = merged.index.max().date()
            print(f"✅ {len(merged)} rows | {merged.index.min().date()} → {new_end}")
            success.append(symbol)
        else:
            print(f"❌ insufficient data")
            failed.append(symbol)

    # ── Process existing symbols with NO new data (keep as-is)
    print("\n── Keeping existing-only symbols unchanged...")
    matched_syms = {s for s, _ in matched}
    for symbol in existing_symbols - matched_syms:
        old_df = load_existing(symbol)
        if old_df is not None:
            print(f"  {symbol}: kept as-is ({len(old_df)} rows)")

    # ── Summary
    print("\n" + "=" * 60)
    print(f"✅ Successfully merged: {len(success)} symbols")
    print(f"❌ Failed: {len(failed)}")
    if failed:
        print(f"   Failed symbols: {failed}")
    print(f"\nIntraday 1-min files saved to: {OUTPUT_INTRADAY}/")
    print(f"Merged daily files saved to: {OUTPUT_DAILY}/")
    print("\nNext step: re-run notebooks/02_feature_engineering.ipynb")
    print("           to reprocess all symbols with new data")


if __name__ == "__main__":
    main()
