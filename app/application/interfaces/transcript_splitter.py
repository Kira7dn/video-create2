from __future__ import annotations

from typing import List, Protocol


class ITranscriptSplitter(Protocol):
    async def split(self, content: str, content_id: str) -> List[str]:
        """Split transcript text into readable chunks.

        Should preserve all words from the original text in order, only adding boundaries.
        """
        ...
