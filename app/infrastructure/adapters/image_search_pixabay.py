from __future__ import annotations

import logging
from typing import Optional

from app.application.interfaces import IImageSearch
from app.core.config import settings
from utils.image_utils import search_pixabay_image

logger = logging.getLogger(__name__)


class PixabayImageSearch(IImageSearch):
    """IImageSearch implementation using the Pixabay API.

    Wraps the existing utility `search_pixabay_image` to keep logic centralized.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.pixabay_api_key

    def search_image(self, keywords: str, min_width: int, min_height: int) -> Optional[str]:
        if not self.api_key:
            logger.debug("PixabayImageSearch: missing API key; returning None")
            return None
        return search_pixabay_image(keywords, self.api_key, min_width, min_height)
