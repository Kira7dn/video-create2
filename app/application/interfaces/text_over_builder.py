from __future__ import annotations

from typing import Any, Dict, List, Protocol


class ITextOverBuilder(Protocol):
    """Builds text_over items from alignment words and text chunks."""

    def build(
        self,
        *,
        word_items: List[Dict[str, Any]],
        chunks: List[str],
        text_over_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return a list of text_over items.

        Each item has at least: text, start_time, duration, word_count.
        """
        ...
