from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

from app.application.interfaces import (
    IStorageRepo,
    IAssetDownloader,
    IVideoRenderer,
    IAudioSynthesizer,
    IUploader,
    IIdGenerator,
    IClock,
    IKeywordAgent,
    IImageSearch,
    ITranscriptionAligner,
    IMediaProbe,
    IImageProcessor,
)


@dataclass(slots=True)
class VideoPipelineAdapters:
    """Container for all adapters used by the video creation pipeline.

    This avoids parameter explosion in builders and centralizes validation.
    """

    # Backward-compat alias: some tests/fixtures pass 'assets' instead of 'downloader'
    assets: Optional[IAssetDownloader] = None
    storage: Optional[IStorageRepo] = None
    downloader: Optional[IAssetDownloader] = None
    renderer: Optional[IVideoRenderer] = None
    tts: Optional[IAudioSynthesizer] = None
    uploader: Optional[IUploader] = None
    id_gen: Optional[IIdGenerator] = None
    clock: Optional[IClock] = None
    keyword_agent: Optional[IKeywordAgent] = None
    image_search: Optional[IImageSearch] = None
    aligner: Optional[ITranscriptionAligner] = None
    media_probe: Optional[IMediaProbe] = None
    image_processor: Optional[IImageProcessor] = None

    def __post_init__(self) -> None:
        # If 'assets' provided but 'downloader' is None, map it for compatibility
        if self.assets is not None and self.downloader is None:
            self.downloader = self.assets

    def validate_required(self, required: Iterable[str]) -> None:
        missing = [name for name in required if getattr(self, name, None) is None]
        if missing:
            raise ValueError(f"Missing required adapters: {', '.join(missing)}")

    # Convenience helpers (optional)
    def has_renderer(self) -> bool:
        return self.renderer is not None


    def has_downloader(self) -> bool:
        return self.downloader is not None

    def has_uploader(self) -> bool:
        return self.uploader is not None

    def has_keyword_agent(self) -> bool:
        return self.keyword_agent is not None

    def has_image_search(self) -> bool:
        return self.image_search is not None

    def has_aligner(self) -> bool:
        return self.aligner is not None
