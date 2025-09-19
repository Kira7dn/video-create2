"""Integration tests for ValidateInputStep (custom in-step validator)."""

import pytest

pytestmark = pytest.mark.integration

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.validate_input import ValidateInputStep

pytestmark = pytest.mark.integration


@pytest.fixture
def valid_video_data():
    return {
        "niche": "tech",
        "keywords": ["ai", "video"],
        "title": "Sample Video",
        "description": "An example payload for validation tests.",
        "segments": [
            {
                "id": "seg-1",
                "image": {"url": "https://example.com/image.jpg"},
            }
        ],
        "background_music": {"url": "https://example.com/music.mp3", "volume": 0.5},
    }


class TestValidateInputStep:
    def setup_method(self):
        self.step = ValidateInputStep()

    async def _run(self, data):
        ctx = PipelineContext(input={"json_data": data})
        await self.step.run(ctx)
        return ctx.get("validated_data")

    @pytest.mark.asyncio
    async def test_valid_data_passes(self, valid_video_data):
        result = await self._run(valid_video_data)
        assert isinstance(result, dict)
        assert "segments" in result and isinstance(result["segments"], list)
        assert result["segments"][0]["id"] == valid_video_data["segments"][0]["id"]

    @pytest.mark.asyncio
    async def test_non_dict_data_fails(self):
        with pytest.raises(ValueError) as e:
            await self._run("not a dict")
        assert "dictionary" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_empty_segments_fails(self, valid_video_data):
        data = {**valid_video_data, "segments": []}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        msg = str(e.value).lower()
        # Pydantic: "segments: List should have at least 1 item"
        assert "segments" in msg and ("at least" in msg or "min" in msg)

    @pytest.mark.asyncio
    async def test_segment_missing_id_fails(self, valid_video_data):
        data = {**valid_video_data, "segments": [{"image": {"url": "u"}}]}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        assert "id" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_media_url_required(self, valid_video_data):
        seg = {"id": "s1", "image": {}}
        with pytest.raises(ValueError) as e:
            await self._run({**valid_video_data, "segments": [seg]})
        assert "image.url" in str(e.value).lower() or "url" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_transition_shape(self, valid_video_data):
        seg = {"id": "s1", "image": {"url": "u"}, "transition_in": 1}
        with pytest.raises(ValueError) as e:
            await self._run({**valid_video_data, "segments": [seg]})
        assert "transition_in" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_background_music_must_be_object(self, valid_video_data):
        data = {**valid_video_data, "background_music": "not_an_object"}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        assert "background_music" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_background_music_requires_url(self, valid_video_data):
        data = {**valid_video_data, "background_music": {"volume": 0.3}}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        assert (
            "background_music.url" in str(e.value).lower()
            or "url" in str(e.value).lower()
        )

    @pytest.mark.asyncio
    async def test_background_music_numeric_fields_types(self, valid_video_data):
        bm = {"url": "u", "volume": "loud", "fade_in": "no"}
        data = {**valid_video_data, "background_music": bm}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        msg = str(e.value).lower()
        assert "background_music.volume" in msg or "background_music.fade_in" in msg

    @pytest.mark.asyncio
    async def test_keywords_must_be_list_of_strings(self, valid_video_data):
        data = {**valid_video_data, "keywords": ["ok", 123]}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        assert "keywords" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_optional_top_level_strings_types(self, valid_video_data):
        # With strict string fields, ints should not be coerced and must raise
        data = {**valid_video_data, "title": 123, "description": 456}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        msg = str(e.value).lower()
        assert "title" in msg and "description" in msg

    @pytest.mark.asyncio
    async def test_segment_non_object_fails(self, valid_video_data):
        data = {**valid_video_data, "segments": ["not an object"]}
        with pytest.raises(ValueError) as e:
            await self._run(data)
        # Pydantic loc uses dot notation
        assert "segments.0" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_media_start_end_delay_numeric(self, valid_video_data):
        seg = {"id": "s1", "image": {"url": "u", "start_delay": "x"}}
        with pytest.raises(ValueError) as e:
            await self._run({**valid_video_data, "segments": [seg]})
        assert "start_delay" in str(e.value).lower()

    @pytest.mark.asyncio
    async def test_transitions_field_types(self, valid_video_data):
        seg = {
            "id": "s1",
            "image": {"url": "u"},
            "transition_in": {"type": 1, "duration": "x"},
        }
        with pytest.raises(ValueError) as e:
            await self._run({**valid_video_data, "segments": [seg]})
        msg = str(e.value).lower()
        assert "transition_in.type" in msg or "transition_in.duration" in msg

    @pytest.mark.asyncio
    async def test_aggregate_errors_multiple_fields(self, valid_video_data):
        # Create payload with multiple errors (id wrong type; image.url missing)
        seg = {"id": 1, "image": {}}
        data = {**valid_video_data, "segments": [seg]}
        ctx = PipelineContext(input={"json_data": data})
        with pytest.raises(ValueError) as e:
            await self.step.run(ctx)
        msg = str(e.value).lower()
        # Expect both fields present in aggregated error locations
        assert "segments.0.id" in msg and (
            "segments.0.image.url" in msg or "image.url" in msg
        )
