from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


class DataHandler:
    """Provides bars and history *as-of* time t, preventing look-ahead."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data: Dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{sym}: index must be DatetimeIndex")
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{sym}: missing columns {missing}")
            self.data[sym] = df.sort_index()

        self.symbols = sorted(self.data.keys())
        self._timeline = self._build_global_timeline()
        self._cursor = 0

    def _build_global_timeline(self) -> List[pd.Timestamp]:
        idxs = [df.index for df in self.data.values()]
        if not idxs:
            return []
        timeline = idxs[0]
        for ix in idxs[1:]:
            timeline = timeline.union(ix)
        return list(timeline.sort_values())

    def reset(self) -> None:
        self._cursor = 0

    def has_next(self) -> bool:
        return self._cursor < len(self._timeline)

    def next_time(self) -> pd.Timestamp:
        if not self.has_next():
            raise StopIteration
        t = self._timeline[self._cursor]
        self._cursor += 1
        return t

    def get_bar(self, symbol: str, t: pd.Timestamp) -> Optional[Dict[str, float]]:
        df = self.data[symbol]
        if t not in df.index:
            return None
        row = df.loc[t]
        return {c: float(row[c]) for c in REQUIRED_COLS}

    def get_history_asof(self, symbol: str, t: pd.Timestamp) -> pd.DataFrame:
        df = self.data[symbol]
        return df.loc[:t].copy()
