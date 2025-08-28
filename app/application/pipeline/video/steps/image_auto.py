from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces import (
    IAssetDownloader,
    IKeywordAgent,
    IImageSearch,
    IImageProcessor,
)
from app.core.config import settings
from app.core.exceptions import ProcessingError
from utils.image_utils import is_image_size_valid

logger = logging.getLogger(__name__)


class ImageAutoStep(BaseStep):
    """
    Validate segment images and auto-replace invalid ones using Pixabay.
    Optionally enhances keyword search via PydanticAI when enabled.
    """
    name = "image_auto"
    required_keys = ["segments"]

    def __init__(
        self,
        downloader: IAssetDownloader,
        keyword_agent: IKeywordAgent,
        image_search: IImageSearch,
        image_processor: IImageProcessor,
    ) -> None:
        self.keyword_agent = keyword_agent
        self.downloader = downloader
        self.image_search = image_search
        self.image_processor = image_processor
        # Configure retry/timeout for network + processing work
        self.retries = int(getattr(settings, "image_auto_retries", 1))
        self.retry_backoff = float(getattr(settings, "image_auto_retry_backoff", 0.5))
        self.max_backoff = float(getattr(settings, "image_auto_max_backoff", 3.0))
        self.jitter = float(getattr(settings, "image_auto_jitter", 0.2))
        self.use_exponential_backoff = bool(
            getattr(settings, "image_auto_use_exp_backoff", True)
        )
        self.timeout = getattr(settings, "image_auto_timeout", None)

    def _init_ai_agent(self) -> None:  # legacy no-op kept for compatibility
        return

    async def _ai_extract_keywords(
        self, content: str, fields: Optional[List[str]] = None
    ) -> List[str]:
        """Return prioritized keywords for image search.

        Falls back to provided fields when AI is disabled/unavailable.
        """
        fields = fields or []
        if not settings.ai_keyword_extraction_enabled:
            return fields

        try:
            final_keywords = await self.keyword_agent.extract_keywords(
                content, fields=fields, max_keywords=settings.ai_max_keywords_per_prompt
            )
            logger.info("🤖 AI keywords: %s", final_keywords)
            return final_keywords
        except Exception as e:  # noqa: BLE001
            logger.warning("AI keyword extraction failed, fallback to fields: %s", e)
            return fields

    async def _ai_search_image(
        self,
        content: str,
        fields: Optional[List[str]] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
    ) -> Optional[str]:
        """AI-enhanced Pixabay search, with fallback."""
        min_width = min_width or settings.video_min_image_width
        min_height = min_height or settings.video_min_image_height

        keywords_list = await self._ai_extract_keywords(content, fields)

        for keywords in keywords_list:
            url = self.image_search.search_image(keywords, min_width, min_height)
            if url:
                logger.info("✅ Found image with keywords '%s'", keywords)
                return url

        # Final fallback
        return self.image_search.search_image(
            "abstract background", min_width, min_height
        )

    async def _download_image(self, content: str, fields: List[str]) -> Tuple[str, str]:
        """Search and download an image via the injected downloader; return (url, local_path)."""

        new_url = await self._ai_search_image(content=content, fields=fields)
        if not new_url:
            raise ProcessingError(
                f"Không tìm được ảnh phù hợp cho content: '{content}' với fields: {fields}"
            )
        try:
            local_path = await self.downloader.download_asset(new_url, kind="image")
            return new_url, local_path
        except Exception as e:  # noqa: BLE001
            raise ProcessingError(f"Download replacement image failed: {e}") from e

    async def run(self, context: PipelineContext) -> None:  # type: ignore[override]
        """Validate/replace images directly on context.segments."""
        segments: List[dict] = context.get("segments") or []
        keywords = context.get("keywords")  # Optional[List[str]]
        min_width = settings.video_min_image_width
        min_height = settings.video_min_image_height

        new_segments: List[dict] = []
        for segment in segments:
            # If there is a video asset, treat as valid image implicitly
            video_obj = segment.get("video")
            if video_obj:
                valid = True
            else:
                image_obj = segment.get("image", {}) or {}
                image_path = image_obj.get("local_path")
                valid = False
                if image_path:
                    valid = is_image_size_valid(image_path, min_width, min_height)

            merged = segment.copy()
            if not valid:
                # voice_over can be None; guard with fallback dict
                vo = segment.get("voice_over") or {}
                content = vo.get("content", "")
                fields = keywords or []
                new_url, local_path = await self._download_image(
                    content=content, fields=fields
                )
                if "image" in merged and isinstance(merged["image"], dict):
                    merged["image"]["url"] = new_url
                    merged["image"]["local_path"] = local_path
                else:
                    merged["image"] = {"url": new_url, "local_path": local_path}

            # Preprocess image to target render size if there is an image and no video
            # Uses fixed resolution from settings; writes to a local tmp directory
            if not video_obj and merged.get("image"):
                try:
                    width, height = settings.video_resolution_tuple
                    img_path = merged["image"].get("local_path")
                    if img_path:
                        processed_list = await self.image_processor.process(
                            img_path,
                            target_width=width,
                            target_height=height,
                            seg_id=str(merged.get("id")) if merged.get("id") is not None else None,
                            smart_pad_color=True,
                            pad_color_method="average_edge",
                            auto_enhance=True,
                            enhance_brightness=True,
                            enhance_contrast=True,
                            enhance_saturation=True,
                        )
                        if processed_list:
                            # Overwrite final local_path with processed output
                            merged["image"]["local_path"] = processed_list[0]
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Image preprocessing failed, will fallback at render: %s", e
                    )

            new_segments.append(merged)

        context.set("segments", new_segments)
        return

    # Skip when feature is disabled via settings
    def can_skip(self, context: PipelineContext) -> bool:
        try:
            enabled = bool(getattr(settings, "image_auto_enabled", True))
        except Exception:
            enabled = True
        return not enabled
