from __future__ import annotations
from typing import Protocol
import datetime as _dt


class IIdGenerator(Protocol):
    """Generates unique, collision-resistant identifiers."""

    def new_id(self) -> str:
        ...


class IClock(Protocol):
    """Provides current time for deterministic testing."""

    def now(self) -> _dt.datetime:
        ...
