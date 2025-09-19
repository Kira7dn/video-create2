import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.builder import (
    build_video_pipeline_via_container,
)

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_flow_with_bundle(fake_adapters, tmp_path):
    """Run full video pipeline using adapter bundle and assert final artifacts.

    Uses session-scoped fake_adapters (AsyncMock-wrapped) from conftest.
    """

    # Build pipeline via new builder
    pipeline = build_video_pipeline_via_container(fake_adapters)

    # Input expected by ValidateInputStep
    data = {
        "json_data": {
            "segments": [
                {
                    "id": "s1",
                    "image": {"url": "http://example.com/bg.jpg"},
                    # Validator requires voice_over.url when voice_over is present
                    "voice_over": {
                        "url": "http://example.com/vo1.mp3",
                        "text": "hello",
                    },
                },
                {
                    "id": "s2",
                    "video": {"url": "http://example.com/in.mp4"},
                },
            ]
        }
    }

    ctx = PipelineContext(input=data)

    result = await pipeline.execute(ctx)
    out_ctx = result["context"]

    # Final artifacts
    assert out_ctx.get("final_video_path")
    # Uploader in fake adapters returns https://example.com/<name>
    assert str(out_ctx.get("final_video_url")).startswith("https://example.com/")

    # Adapter interactions
    # downloader: at least 2 downloads (bg.jpg and in.mp4)
    assert fake_adapters.downloader.download_asset.await_count >= 2
    # renderer may use process_with_specification instead of render
    assert fake_adapters.renderer.concat_clips.await_count == 1
    # upload at least once (optional if uploader provided). May be 2 when fallback occurs.
    if fake_adapters.uploader:
        assert fake_adapters.uploader.upload_file.await_count >= 1
