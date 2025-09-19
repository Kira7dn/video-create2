import asyncio
import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.image_auto import ImageAutoStep


class FakeDownloader:
    async def download_asset(self, url: str, kind: str = "image"):
        return "/tmp/dl.jpg"


class FakeKeywordAgent:
    async def extract_keywords(self, content: str, fields=None, max_keywords: int = 3):
        # return fields as-is to keep deterministic
        return fields or ["kw"]


class FakeImageSearch:
    def __init__(self, responses):
        # responses is a dict mapping (w,h) -> url or None
        self.responses = responses
        self.calls = []

    def search_image(self, keywords: str, min_width: int, min_height: int):
        key = (min_width, min_height)
        self.calls.append((keywords, min_width, min_height))
        return self.responses.get(key)


class FakeImageProcessor:
    async def process(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        return ["/tmp/processed.jpg"]


@pytest.mark.asyncio
async def test_ai_search_image_relax_short_video_type():
    # Prepare responses: return URL only at most-relaxed (0.8) for short (1080x1920)
    # Expected sequence for short with min config (1024x576):
    # factor=1.0 -> (1080, 1920)
    # factor=0.9 -> (1024, 1728)
    # factor=0.8 -> (1024, 1536)
    responses = {
        (1080, 1920): None,
        (1024, 1728): None,
        (1024, 1536): "http://example.com/final.jpg",
    }
    search = FakeImageSearch(responses)
    step = ImageAutoStep(
        downloader=FakeDownloader(),
        keyword_agent=FakeKeywordAgent(),
        image_search=search,
        image_processor=FakeImageProcessor(),
    )
    # Provide context with validated_data.video_type='short'
    ctx = PipelineContext(input={"json_data": {}})
    ctx.set("validated_data", {"video_type": "short"})
    step._ctx = ctx  # expose context for helper

    url = await step._ai_search_image(content="c", fields=["kw"])  # type: ignore[attr-defined]
    assert url == "http://example.com/final.jpg"

    # Verify call sequence sizes
    sizes = [(w, h) for _, w, h in search.calls]
    assert sizes[:3] == [(1080, 1920), (1024, 1728), (1024, 1536)]


@pytest.mark.asyncio
async def test_ai_search_image_base_sizes_long_video_type():
    # For long (1280x720), should try (1280,720) first
    responses = {
        (1280, 720): "http://example.com/ok.jpg",
    }
    search = FakeImageSearch(responses)
    step = ImageAutoStep(
        downloader=FakeDownloader(),
        keyword_agent=FakeKeywordAgent(),
        image_search=search,
        image_processor=FakeImageProcessor(),
    )
    ctx = PipelineContext(input={"json_data": {}})
    ctx.set("validated_data", {"video_type": "long"})
    step._ctx = ctx

    url = await step._ai_search_image(content="c", fields=["kw"])  # type: ignore[attr-defined]
    assert url == "http://example.com/ok.jpg"
    # First call should be with (1280, 720)
    assert search.calls[0][1:] == (1280, 720)
