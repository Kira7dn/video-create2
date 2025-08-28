from __future__ import annotations

from typing import Protocol, runtime_checkable

from .asset_repo import IAssetDownloader
from .renderer import IVideoRenderer
from .uploader import IUploader
from .ai_agent import IKeywordAgent
from .image_search import IImageSearch
from .aligner import ITranscriptionAligner
from .image_processor import IImageProcessor
from .transcript_splitter import ITranscriptSplitter
from .text_over_builder import ITextOverBuilder
from .media_probe import IMediaProbe


@runtime_checkable
class IVideoPipelineAdapters(Protocol):
    downloader: IAssetDownloader
    renderer: IVideoRenderer
    uploader: IUploader
    keyword_agent: IKeywordAgent
    image_search: IImageSearch
    aligner: ITranscriptionAligner
    media_probe: IMediaProbe
    image_processor: IImageProcessor
    splitter: ITranscriptSplitter
    text_over_builder: ITextOverBuilder
