from __future__ import annotations

from typing import Optional, Protocol


class IImageSearch(Protocol):
    """Adapter for searching image URLs by keywords.

    Implementations may call Pixabay, Pexels, etc. The application layer should
    not know about concrete providers.
    """

    def search_image(self, keywords: str, min_width: int, min_height: int) -> Optional[str]:
        """Return a direct image URL for the given keywords or None if not found."""
        ...
