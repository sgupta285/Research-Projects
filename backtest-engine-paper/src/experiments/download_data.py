from __future__ import annotations

import argparse
import os
import pandas as pd

from src.utils.io import ensure_dir

STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Stooq for a symbol."""
    # US tickers usually work in lowercase on Stooq.
    for sym in [symbol.lower(), symbol]:
        url = STOOQ_DAILY_URL.format(sym=sym)
        try:
            df = pd.read_csv(url)
            if len(df) > 0:
                break
        except Exception:
            df = None
    if df is None or len(df) == 0:
        raise RuntimeError(f"Failed to fetch data from Stooq for {symbol}")

    df = df.rename(columns={
        "Date": "t",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
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

    start = pd.to_datetime(args.start)
    ensure_dir(args.raw_dir)
    ensure_dir(args.processed_dir)

    for sym in args.symbols:
        df = fetch_stooq_daily(sym)
        df = df.loc[df.index >= start].copy()

        raw_path = os.path.join(args.raw_dir, f"{sym}.csv")
        df.to_csv(raw_path, index=True)

        processed_path = os.path.join(args.processed_dir, f"{sym}.csv")
        df.reset_index().to_csv(processed_path, index=False)

        print(f"[OK] {sym}: {len(df)} rows -> {raw_path} / {processed_path}")


if __name__ == "__main__":
    main()
