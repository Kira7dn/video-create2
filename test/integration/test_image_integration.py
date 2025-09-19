import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.validate_input import ValidateInputStep
from app.application.pipeline.video.steps.download_assets import (
    DownloadAssetsStep,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.integration
async def test_image_download_via_download_assets_step(fake_adapters, tmp_path):
    """Validate and download an image asset via pipeline steps using adapters."""

    # Arrange input for ValidateInputStep -> expects input["json_data"]["segments"]
    context = PipelineContext(
        input={
            "json_data": {
                "segments": [
                    {
                        "id": "s1",
                        "text": "A cute cat",
                        "image": {"url": "https://example.com/cat.jpg"},
                    }
                ]
            }
        }
    )

    # Act: run validate -> download_assets
    await ValidateInputStep().run(context)
    await DownloadAssetsStep(fake_adapters.downloader).run(context)

    # Assert segments contain image.local_path and adapter called once
    segments = context.get("segments")
    assert isinstance(segments, list) and segments, "segments not set"
    image = segments[0].get("image")
    assert image and image.get("local_path"), "image.local_path must be set"
    assert fake_adapters.downloader.download_asset.await_count == 1
