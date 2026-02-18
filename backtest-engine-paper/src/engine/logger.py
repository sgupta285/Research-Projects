from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List
import pandas as pd


class EventLogger:
    """Collects events and can flush them to CSV for reproducibility/debugging."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.rows: List[Dict[str, Any]] = []

    def log(self, evt: Any) -> None:
        if not self.enabled:
            return
        d = asdict(evt)
        d["event_type"] = evt.__class__.__name__
        if "t" in d:
            d["t"] = str(pd.Timestamp(d["t"]))
        self.rows.append(d)

    def flush_csv(self, path: str) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)
