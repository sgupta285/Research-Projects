from __future__ import annotations

from collections import deque
from typing import Deque, Union

from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent

Event = Union[MarketEvent, SignalEvent, OrderEvent, FillEvent]


class EventQueue:
    """A simple FIFO event queue."""

    def __init__(self):
        self._q: Deque[Event] = deque()

    def put(self, evt: Event) -> None:
        self._q.append(evt)

    def get(self) -> Event:
        return self._q.popleft()

    def empty(self) -> bool:
        return len(self._q) == 0

    def __len__(self) -> int:
        return len(self._q)
