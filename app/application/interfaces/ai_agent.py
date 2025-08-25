from __future__ import annotations

from typing import Protocol, List, Optional


class IKeywordAgent(Protocol):
    """AI keyword extraction adapter used by the application layer.

    Implementations may call external LLMs or local models. The application
    should not depend on any concrete AI SDKs.
    """

    async def extract_keywords(
        self,
        content: str,
        *,
        fields: Optional[List[str]] = None,
        max_keywords: int = 8,
    ) -> List[str]:
        """Return a list of keywords/phrases for image search."""
        ...
