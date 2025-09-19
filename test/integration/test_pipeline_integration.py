"""
Integration tests for the video processing pipeline (new structure).
"""

from pathlib import Path
from types import SimpleNamespace
import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.builder import build_video_pipeline_via_container
from unittest.mock import AsyncMock

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_video_pipeline_happy_path(fake_adapters):
    pipeline = build_video_pipeline_via_container(fake_adapters)

    input_payload = {
        "json_data": {
            "segments": [
                {
                    "id": "s1",
                    "image": {"url": "https://cdn/img1.jpg"},
                    "video": {"url": "https://cdn/vid1.mp4"},
                    "voice_over": {"url": "https://cdn/vo1.mp3"},
                },
                {
                    "id": "s2",
                    "image": {"url": "https://cdn/img2.jpg"},
                },
            ]
        },
        "background_music": {"url": "https://cdn/bg.mp3"},
    }

    ctx = PipelineContext(input=input_payload)
    result = await pipeline.execute(ctx)

    # Execution returns a dict with context
    assert "context" in result
    out_ctx: PipelineContext = result["context"]

    # Validate artifacts populated by steps
    segments = out_ctx.get("segments")
    assert isinstance(segments, list) and len(segments) == 2

    clips = out_ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 2

    final_path = out_ctx.get("final_video_path")
    assert isinstance(final_path, str) and final_path.endswith(".mp4")

    final_url = out_ctx.get("final_video_url")
    assert isinstance(final_url, str) and final_url.startswith("https://")

    # Adapter interactions
    assert fake_adapters.downloader.download_asset.await_count >= 1  # type: ignore[attr-defined]
    # concat_clips should be called at least once (may be called multiple times in some flows)
    assert fake_adapters.renderer.concat_clips.await_count >= 1  # type: ignore[attr-defined]
    # Upload may be attempted multiple times; ensure at least once
    assert fake_adapters.uploader.upload_file.await_count >= 1  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_video_pipeline_invalid_input_raises(fake_adapters):
    pipeline = build_video_pipeline_via_container(fake_adapters)

    # Missing 'segments' in json_data should raise in ValidateInputStep
    bad_input = {"json_data": {"not_segments": []}}
    ctx = PipelineContext(input=bad_input)

    # Pydantic-based message can vary; ensure it mentions 'segments'
    with pytest.raises(ValueError, match="segments"):
        await pipeline.execute(ctx)
