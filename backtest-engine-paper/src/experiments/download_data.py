from __future__ import annotations

import argparse
import os
import pandas as pd

from src.utils.io import ensure_dir


def fetch_yfinance(symbol: str, start: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, auto_adjust=True)

    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "t"
    df = df[["open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", type=str, default="2005-01-01")
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    args = ap.parse_args()

    ensure_dir(args.raw_dir)
    ensure_dir(args.processed_dir)

    for sym in args.symbols:
        print(f"Downloading {sym}...")
        df = fetch_yfinance(sym, args.start)

        raw_path = os.path.join(args.raw_dir, f"{sym}.csv")
        df.to_csv(raw_path, index=True)

        processed_path = os.path.join(args.processed_dir, f"{sym}.csv")
        df.reset_index().to_csv(processed_path, index=False)

        print(f"  [OK] {sym}: {len(df)} rows -> {processed_path}")


if __name__ == "__main__":
    main()
