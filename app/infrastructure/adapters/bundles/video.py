from __future__ import annotations

import os
from app.application.pipeline.video.adapter_bundle import VideoPipelineAdapters
from app.infrastructure.adapters import (
    AssetRepoDownloader,
    FFMpegVideoRenderer,
    S3Uploader,
    PydanticAIKeywordAgent,
    PixabayImageSearch,
    GentleTranscriptionAligner,
    FFmpegMediaProbe,
    ImageProcessor,
)
from app.core.config import settings


def get_video_adapter_bundle(*, temp_dir: str | None = None) -> VideoPipelineAdapters:
    """Provide the adapters container for the video pipeline.

    Kept under infrastructure/adapters/container to reflect it's a concrete
    assembly of adapter implementations.
    """
    # Resolve a safe default work_dir under the base temp directory
    if temp_dir:
        work_dir = temp_dir
    else:
        base_dir = os.getenv("TEMP_BASE_DIR", "data/tmp")
        os.makedirs(base_dir, exist_ok=True)
        work_dir = os.path.join(base_dir, settings.temp_batch_dir)
    assets_dir = os.path.join(work_dir, "assets")
    return VideoPipelineAdapters(
        downloader=AssetRepoDownloader(base_dir=assets_dir),
        renderer=FFMpegVideoRenderer(temp_dir=work_dir),
        uploader=S3Uploader(),
        keyword_agent=PydanticAIKeywordAgent(),
        image_search=PixabayImageSearch(),
        aligner=GentleTranscriptionAligner(temp_dir=work_dir),
        media_probe=FFmpegMediaProbe(),
        image_processor=ImageProcessor(temp_dir=work_dir),
    )
