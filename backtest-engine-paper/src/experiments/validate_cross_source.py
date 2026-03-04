from __future__ import annotations

import argparse
import io
import os
from typing import Dict, List, Optional
from urllib.request import urlopen
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_processed_symbols


def fetch_stooq(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    sym = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    with urlopen(url, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    if df is None or len(df) == 0:
        raise RuntimeError(f"Stooq returned no data for {symbol}")

    df = df.rename(
        columns={
            "Date": "t",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if start is not None:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df.index <= pd.to_datetime(end)]
    return df


def validate_against_stooq(
    processed_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    window: int = 60,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        yf = processed_data[sym].copy()
        if start is not None:
            yf = yf.loc[yf.index >= pd.to_datetime(start)]
        if end is not None:
            yf = yf.loc[yf.index <= pd.to_datetime(end)]

        st = fetch_stooq(sym, start=start, end=end)

        y = yf["close"].astype(float).rename("yf_close")
        s = st["close"].astype(float).rename("stooq_close")
        px = pd.concat([y, s], axis=1, join="inner").dropna()
        if len(px) < 3:
            rows.append(
                {
                    "symbol": sym,
                    "n_overlap_prices": int(len(px)),
                    "n_overlap_returns": 0,
                    "close_corr": float("nan"),
                    "return_corr": float("nan"),
                    "rolling_corr_mean": float("nan"),
                    "rolling_corr_min": float("nan"),
                    "rolling_corr_max": float("nan"),
                }
            )
            continue

        r = pd.DataFrame(
            {
                "yf_ret": np.log(px["yf_close"].astype(float)).diff(),
                "st_ret": np.log(px["stooq_close"].astype(float)).diff(),
            },
            index=px.index,
        )
        r = r.dropna()

        if len(r) < 3:
            close_corr = float(px["yf_close"].corr(px["stooq_close"])) if len(px) > 2 else float("nan")
            rows.append(
                {
                    "symbol": sym,
                    "n_overlap_prices": int(len(px)),
                    "n_overlap_returns": int(len(r)),
                    "close_corr": close_corr,
                    "return_corr": float("nan"),
                    "rolling_corr_mean": float("nan"),
                    "rolling_corr_min": float("nan"),
                    "rolling_corr_max": float("nan"),
                }
            )
            continue

        rolling = r["yf_ret"].rolling(window).corr(r["st_ret"]).dropna()
        rows.append(
            {
                "symbol": sym,
                "n_overlap_prices": int(len(px)),
                "n_overlap_returns": int(len(r)),
                "close_corr": float(px["yf_close"].corr(px["stooq_close"])),
                "return_corr": float(r["yf_ret"].corr(r["st_ret"])),
                "rolling_corr_mean": float(rolling.mean()) if len(rolling) else float("nan"),
                "rolling_corr_min": float(rolling.min()) if len(rolling) else float("nan"),
                "rolling_corr_max": float(rolling.max()) if len(rolling) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("symbol")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/tables/cross_source_validation.csv")
    args = ap.parse_args()

    data = load_processed_symbols(args.processed_dir, args.symbols)
    out_df = validate_against_stooq(
        processed_data=data,
        symbols=args.symbols,
        window=int(args.window),
        start=args.start,
        end=args.end,
    )
    ensure_dir(os.path.dirname(args.out) or ".")
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved {args.out}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
