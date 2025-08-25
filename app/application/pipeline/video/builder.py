from __future__ import annotations

from typing import Optional

from app.application.pipeline.base import Pipeline
from app.application.pipeline.factory import PipelineFactory
from app.application.pipeline.video.adapter_bundle import VideoPipelineAdapters
from app.application.interfaces import (
    IAssetDownloader,
    IVideoRenderer,
    IUploader,
    IKeywordAgent,
    IImageSearch,
    ITranscriptionAligner,
    IImageProcessor,
)
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


def build_video_creation_pipeline(
    *,
    downloader: Optional[IAssetDownloader] = None,
    renderer: Optional[IVideoRenderer] = None,
    uploader: Optional[IUploader] = None,
    keyword_agent: Optional[IKeywordAgent] = None,
    image_search: Optional[IImageSearch] = None,
    aligner: Optional[ITranscriptionAligner] = None,
    image_processor: Optional[IImageProcessor] = None,
) -> Pipeline:
    factory = PipelineFactory()
    factory.add(ValidateInputStep())
    factory.add(DownloadAssetsStep(downloader))
    factory.add(ImageAutoStep(downloader, keyword_agent, image_search, image_processor))
    factory.add(TranscriptionAlignStep(aligner))
    factory.add(CreateSegmentClipsStep(renderer))
    factory.add(ConcatenateVideoStep(renderer))
    factory.add(UploadFinalStep(uploader))

    return factory.build()


def build_video_pipeline_via_container(adapters: VideoPipelineAdapters) -> Pipeline:
    # adapters.validate_required(["downloader", "renderer"])  # uploader optional

    return build_video_creation_pipeline(
        downloader=adapters.downloader,
        renderer=adapters.renderer,
        uploader=adapters.uploader,
        keyword_agent=adapters.keyword_agent,
        image_search=adapters.image_search,
        aligner=adapters.aligner,
        image_processor=adapters.image_processor,
    )
