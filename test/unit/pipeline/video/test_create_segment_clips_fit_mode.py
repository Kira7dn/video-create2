import asyncio
import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.create_segment_clips import (
    CreateSegmentClipsStep,
)
from app.application.interfaces.renderer import IVideoRenderer


class FakeRenderer(IVideoRenderer):
    def __init__(self):
        self.last_spec = None
        self.last_args = {}

    async def duration(self, input_path: str) -> float:  # pragma: no cover - not used
        return 1.0

    async def concat_clips(
        self,
        inputs,
        *,
        output_path: str,
        transition: str | None = None,
        background_music: dict | None = None,
    ) -> str:  # pragma: no cover - not used
        return output_path

    async def render_segment(
        self,
        segment: dict,
        *,
        seg_id: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        # capture the spec/args for assertions
        self.last_spec = segment
        self.last_args = {
            "seg_id": seg_id,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "frame_rate": frame_rate,
        }
        return f"/tmp/{seg_id}/segment_video.mp4"


@pytest.mark.asyncio
async def test_fit_mode_cover_when_video_type_short():
    renderer = FakeRenderer()
    step = CreateSegmentClipsStep(renderer)

    # context with one image segment; ensure local_path present to avoid fetch
    context = PipelineContext(
        input={
            "json_data": {
                "segments": [
                    {
                        "id": "s1",
                        "image": {"url": "", "local_path": "/tmp/placeholder.jpg"},
                    }
                ],
                "video_type": "short",
            }
        }
    )
    # Emulate prior steps
    context.set("segments", context.input["json_data"]["segments"])  # type: ignore[index]
    context.set("validated_data", context.input["json_data"])  # type: ignore[index]

    await step.run(context)

    assert renderer.last_spec is not None
    transforms = renderer.last_spec.get("transformations", [])
    # The first transform should be our canvas_fit with cover mode
    assert any(
        t.get("type") == "canvas_fit" and t.get("fit_mode") == "cover"
        for t in transforms
    ), f"Expected canvas_fit with cover in transforms, got: {transforms}"
