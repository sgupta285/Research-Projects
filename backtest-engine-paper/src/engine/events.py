from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    """A new market bar is available for a symbol at time t."""
    t: pd.Timestamp
    symbol: str
    bar: Dict[str, Any]  # keys: open, high, low, close, volume


@dataclass(frozen=True)
class SignalEvent:
    """A strategy signal generated at time t."""
    t: pd.Timestamp
    symbol: str
    side: str  # "BUY" or "SELL" (SELL means exit/flat in long-only mode)
    strength: float = 1.0


@dataclass(frozen=True)
class OrderEvent:
    """An order to be executed by the execution handler."""
    t: pd.Timestamp
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: int
    order_type: str = "MKT"


@dataclass(frozen=True)
class FillEvent:
    """A filled order with realized price + costs."""
    t: pd.Timestamp
    symbol: str
    side: str
    qty: int
    price: float
    fee: float
    slippage: float
    meta: Optional[Dict[str, Any]] = None
