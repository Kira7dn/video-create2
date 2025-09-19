"""Unit tests for ValidateInputStep (unit-level)."""

import pytest
from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.validate_input import ValidateInputStep


@pytest.fixture
def unit_valid_video_data():
    return {
        "niche": "tech",
        "keywords": ["ai", "video"],
        "title": "Sample Video",
        "description": "An example payload for unit tests.",
        "segments": [
            {"id": "seg-1", "image": {"url": "https://example.com/image.jpg"}},
        ],
        "background_music": {"url": "https://example.com/music.mp3", "volume": 0.5},
    }


class TestValidateInputStepUnit:
    def setup_method(self):
        self.step = ValidateInputStep()

    async def _run(self, data):
        ctx = PipelineContext(input={"json_data": data})
        await self.step.run(ctx)
        return ctx.get("validated_data")

    @pytest.mark.asyncio
    async def test_valid_data_passes(self, unit_valid_video_data):
        result = await self._run(unit_valid_video_data)
        assert isinstance(result, dict)
        assert result["segments"][0]["id"] == "seg-1"

    @pytest.mark.asyncio
    async def test_non_dict_data_fails(self):
        with pytest.raises(ValueError):
            await self._run("not a dict")

    @pytest.mark.asyncio
    async def test_missing_segments_fails(self, unit_valid_video_data):
        bad = {**unit_valid_video_data}
        bad.pop("segments")
        with pytest.raises(ValueError) as e:
            await self._run(bad)
        assert "segments" in str(e.value).lower()
