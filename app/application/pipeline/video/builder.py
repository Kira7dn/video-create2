from __future__ import annotations

from app.application.pipeline.base import Pipeline, make_logging_middleware
from app.application.pipeline.factory import PipelineFactory
from app.application.pipeline.video.steps.validate_input import ValidateInputStep
from app.application.pipeline.video.steps.download_assets import DownloadAssetsStep
from app.application.pipeline.video.steps.image_auto import ImageAutoStep
from app.application.pipeline.video.steps.transcription_align import (
    TranscriptionAlignStep,
)
from app.application.pipeline.video.steps.create_segment_clips import (
    CreateSegmentClipsStep,
)
from app.application.pipeline.video.steps.concatenate_video import ConcatenateVideoStep
from app.application.pipeline.video.steps.upload_final import UploadFinalStep
from app.application.interfaces import IVideoPipelineAdapters


def build_video_pipeline_via_container(
    adapters: IVideoPipelineAdapters,
    *,
    enable_logging_middleware: bool = True,
    fail_fast: bool = True,
) -> Pipeline:

    middlewares = [make_logging_middleware()] if enable_logging_middleware else []
    factory = PipelineFactory(middlewares=middlewares, fail_fast=fail_fast)
    factory.add(ValidateInputStep())
    factory.add(DownloadAssetsStep(adapters.downloader))
    factory.add(
        ImageAutoStep(
            adapters.downloader,
            adapters.keyword_agent,
            adapters.image_search,
            adapters.image_processor,
        )
    )
    factory.add(
        TranscriptionAlignStep(
            adapters.splitter, adapters.aligner, adapters.text_over_builder
        )
    )
    factory.add(CreateSegmentClipsStep(adapters.renderer))
    factory.add(ConcatenateVideoStep(adapters.renderer))
    factory.add(UploadFinalStep(adapters.uploader))

    return factory.build()
