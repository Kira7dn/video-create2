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
        # Expose pipeline context to helper methods (initialized here to satisfy linters)
        self._ctx: Optional[PipelineContext] = None
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
        # Determine canvas target size by video_type to avoid upscaling
        # Try to access context via closure (set in run); fallback to settings defaults
        try:
            # This attribute will be set in run() just before calling helpers
            ctx: PipelineContext = getattr(self, "_ctx", None)  # type: ignore[name-defined]
            if ctx is not None:
                vd = ctx.get("validated_data") or {}
                vtype = (
                    vd.get("video_type") if isinstance(vd, dict) else None
                ) or "long"
            else:
                vtype = "long"
        except Exception:
            vtype = "long"

        canvas_w, canvas_h = settings.video_resolution_tuple_for(str(vtype))
        # Base constraints: at least canvas size or configured minima (unless caller provided stricter)
        base_min_w = int(min_width or max(settings.video_min_image_width, canvas_w))
        base_min_h = int(min_height or max(settings.video_min_image_height, canvas_h))

        keywords_list = await self._ai_extract_keywords(content, fields)

        # Progressive relax: try with decreasing size requirements
        relax_factors = [1.0, 0.9, 0.8]  # up to 20% relax
        for factor in relax_factors:
            req_w = max(settings.video_min_image_width, int(base_min_w * factor))
            req_h = max(settings.video_min_image_height, int(base_min_h * factor))
            try:
                logger.info(
                    "🖼️ Image search (video_type=%s) with min_size=%dx%d (factor=%.2f)",
                    vtype,
                    req_w,
                    req_h,
                    factor,
                )
            except Exception:
                pass
            for keywords in keywords_list:
                url = self.image_search.search_image(keywords, req_w, req_h)
                if url:
                    logger.info("✅ Found image with keywords '%s' at %dx%d", keywords, req_w, req_h)
                    return url

        # Final fallback
        # Try a generic background with the most relaxed constraints
        last_w = max(settings.video_min_image_width, int(base_min_w * relax_factors[-1]))
        last_h = max(settings.video_min_image_height, int(base_min_h * relax_factors[-1]))
        return self.image_search.search_image("abstract background", last_w, last_h)

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
        # Expose context for helper methods (e.g., to read validated_data.video_type)
        self._ctx = context  # type: ignore[attr-defined]
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
                    # Determine canvas size based on validated video_type (default to 'long')
                    validated = context.get("validated_data") or {}
                    video_type = (
                        validated.get("video_type")
                        if isinstance(validated, dict)
                        else None
                    ) or "long"
                    width, height = settings.video_resolution_tuple_for(str(video_type))
                    fit_mode = "cover" if str(video_type).lower() == "short" else "contain"
                    img_path = merged["image"].get("local_path")
                    if img_path:
                        processed_list = await self.image_processor.process(
                            img_path,
                            target_width=width,
                            target_height=height,
                            mode=fit_mode,
                            seg_id=(
                                str(merged.get("id"))
                                if merged.get("id") is not None
                                else None
                            ),
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
