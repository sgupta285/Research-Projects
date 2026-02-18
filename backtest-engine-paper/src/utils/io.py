from __future__ import annotations

import os
from typing import Dict, List
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_processed_symbols(processed_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p = os.path.join(processed_dir, f"{sym}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing processed file: {p}. Run download_data first.")
        df = pd.read_csv(p, parse_dates=["t"]).set_index("t").sort_index()
        data[sym] = df
    return data
