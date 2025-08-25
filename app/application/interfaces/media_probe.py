from __future__ import annotations

from typing import Protocol


class IMediaProbe(Protocol):
    async def duration(self, input_path: str) -> float:
        """Return media duration in seconds.
        Implementations may use ffprobe or other tools.
        """
        ...
