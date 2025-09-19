from __future__ import annotations

import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.create_segment_clips import (
    CreateSegmentClipsStep,
)


class FakeRenderer:
    def __init__(self, durations: dict[str, float]):
        # map of media path -> duration seconds
        self._durations = durations
        self.captured_specs: list[dict] = []

    async def duration(self, path: str) -> float:
        return float(self._durations[path])

    async def process_with_specification(
        self,
        specification: dict,
        *,
        seg_id: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        # capture for assertions
        self.captured_specs.append(specification)
        # return a fake file path
        return f"/tmp/{seg_id}/segment_video.mp4"

    async def render_segment(
        self,
        segment: dict,
        *,
        seg_id: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        # For tests, capture segment spec instead of specification
        self.captured_specs.append(segment)
        # return a fake file path
        return f"/tmp/{seg_id}/segment_video.mp4"


@pytest.mark.asyncio
async def test_image_segment_with_voice_and_fades_duration():
    # voice is longer than base -> duration = fade_in + max(base, voice) + fade_out
    fake = FakeRenderer(
        {
            "/tmp/voice.wav": 5.0,
        }
    )
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "s1",
        "duration": 3.0,  # base
        "image": {"local_path": "/tmp/bg.jpg"},
        "voice_over": {"local_path": "/tmp/voice.wav"},
        "transition_in": {"duration": 0.7},
        "transition_out": {"duration": 0.8},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])

    await step(ctx)

    # Assertions
    assert ctx.has("segment_clips")
    clips = ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 1

    # Verify duration passed to renderer
    assert len(fake.captured_specs) == 1
    spec = fake.captured_specs[0]
    assert spec["primary_source"] == "/tmp/bg.jpg"
    # duration = fade_in (0.7) + max(base 3.0, voice 5.0) + fade_out (0.8) = 6.5
    assert pytest.approx(spec["duration"], rel=1e-6) == 6.5


@pytest.mark.asyncio
async def test_video_segment_duration_equals_media_duration():
    fake = FakeRenderer(
        {
            "/tmp/clip.mp4": 7.25,
        }
    )
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "s2",
        "video": {"local_path": "/tmp/clip.mp4"},
        # fades should not change total for video; code uses original_duration
        "transition_in": {"duration": 1.2},
        "transition_out": {"duration": 1.0},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])

    await step(ctx)

    # Assertions
    assert ctx.has("segment_clips")
    clips = ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 1

    assert len(fake.captured_specs) == 1
    spec = fake.captured_specs[0]
    assert spec["primary_source"] == "/tmp/clip.mp4"
    # New logic: total duration = transition_in + start_delay + media + end_delay + transition_out
    # Here start_delay=end_delay=0, so total = 1.2 + 7.25 + 1.0 = 9.45
    assert pytest.approx(spec["duration"], rel=1e-6) == 9.45


@pytest.mark.asyncio
async def test_image_segment_without_voice_uses_base_plus_fades():
    fake = FakeRenderer({})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "s3",
        "duration": 4.0,  # base
        "image": {"local_path": "/tmp/bg.jpg"},
        # no voice over
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.5},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    # duration = 0.5 + 4.0 + 0.5 = 5.0
    assert pytest.approx(spec["duration"], rel=1e-6) == 5.0


class RaisingRenderer(FakeRenderer):
    async def get_media_duration(self, path: str) -> float:  # type: ignore[override]
        raise RuntimeError("probe failed")


@pytest.mark.asyncio
async def test_video_duration_error_falls_back_to_base():
    # When duration probe fails, step falls back to base_duration from segment
    fake = RaisingRenderer({})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "s4",
        "duration": 4.5,  # base fallback
        "video": {"local_path": "/tmp/clip.mp4"},
        "transition_in": {"duration": 1.0},
        "transition_out": {"duration": 1.0},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    # New logic: total duration = transition_in + start_delay + base + end_delay + transition_out
    # With base=4.5, start/end delay=0, fades 1.0/1.0 => total = 1.0 + 4.5 + 1.0 = 6.5
    spec = fake.captured_specs[0]
    assert pytest.approx(spec["duration"], rel=1e-6) == 6.5


@pytest.mark.asyncio
async def test_missing_image_for_image_segment_raises():
    fake = FakeRenderer({})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "s5",
        # image key missing -> should raise
        "duration": 3.0,
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])

    with pytest.raises(RuntimeError):
        await step(ctx)


@pytest.mark.asyncio
async def test_text_overlay_timings_respect_delay_and_clamp():
    fake = FakeRenderer({})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    # base=4, fade_in=0.5, fade_out=0.5 -> total = 5.0
    # start_delay=0.3 -> delay = 0.5 + 0.3 = 0.8
    # text item: start=1.0, duration=10.0 -> start'=1.0+0.8=1.8, end=min(5.0, 11.8)=5.0, dur'=3.2
    segment = {
        "id": "s6",
        "duration": 4.0,
        "image": {"local_path": "/tmp/bg.jpg"},
        "voice_over": {"start_delay": 0.3},
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.5},
        "text_over": [
            {"text": "Hello", "start_time": 1.0, "duration": 10.0, "font_size": 20},
        ],
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    assert pytest.approx(spec["duration"], rel=1e-6) == 5.0
    # find text_overlay transformation
    overlays = [t for t in spec["transformations"] if t.get("type") == "text_overlay"]
    assert len(overlays) == 1
    timing = overlays[0]["timing"]
    assert pytest.approx(timing["start"], rel=1e-6) == 1.8
    assert pytest.approx(timing["duration"], rel=1e-6) == 3.2


@pytest.mark.asyncio
async def test_multiple_segments_produce_multiple_outputs():
    fake = FakeRenderer(
        {
            "/tmp/clipA.mp4": 2.0,
            "/tmp/clipB.mp4": 3.0,
        }
    )
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segments = [
        {"id": "A", "video": {"local_path": "/tmp/clipA.mp4"}},
        {"id": "B", "video": {"local_path": "/tmp/clipB.mp4"}},
    ]

    ctx = PipelineContext(input={})
    ctx.set("segments", segments)
    await step(ctx)

    clips = ctx.get("segment_clips")
    assert [c["id"] for c in clips] == ["A", "B"]
    assert len(fake.captured_specs) == 2


@pytest.mark.asyncio
async def test_emits_audio_delay_when_start_delay_set():
    fake = FakeRenderer({})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    # base=4, fade_in=0.5, fade_out=0.5, start_delay=0.3s -> voice_delay = 0.5+0.3 = 0.8s => expect audio_delay 800ms
    segment = {
        "id": "s7",
        "duration": 4.0,
        "image": {"local_path": "/tmp/bg.jpg"},
        "voice_over": {"start_delay": 0.3},
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.5},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    audio_delays = [
        t for t in spec["transformations"] if t.get("type") == "audio_delay"
    ]
    assert len(audio_delays) == 1
    assert audio_delays[0]["milliseconds"] == 800


@pytest.mark.asyncio
async def test_video_segment_with_start_end_delay_and_fades_duration():
    fake = FakeRenderer({"/tmp/clip.mp4": 2.0})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "sv1",
        "video": {"local_path": "/tmp/clip.mp4"},
        "voice_over": {"start_delay": 0.3, "end_delay": 0.4},
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.6},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    # total = fade_in + start_delay + media + end_delay + fade_out = 0.5 + 0.3 + 2.0 + 0.4 + 0.6 = 3.8
    assert pytest.approx(spec["duration"], rel=1e-6) == 3.8


@pytest.mark.asyncio
async def test_video_audio_delay_equals_fadein_plus_start_delay():
    fake = FakeRenderer({"/tmp/clip.mp4": 1.0})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "sv2",
        "video": {"local_path": "/tmp/clip.mp4"},
        "voice_over": {"start_delay": 0.3},
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.1},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    audio_delays = [t for t in spec["transformations"] if t.get("type") == "audio_delay"]
    assert len(audio_delays) == 1
    # delay_ms = (fade_in + start_delay) * 1000 = (0.5 + 0.3) * 1000 = 800
    assert audio_delays[0]["milliseconds"] == 800


@pytest.mark.asyncio
async def test_image_segment_with_start_end_delay_and_fades_duration():
    # voice 1.2s, start=0.3, end=0.4 => original_duration = max(1.0, 1.2+0.3+0.4=1.9)=1.9
    # total = fade_in(0.2) + 1.9 + fade_out(0.3) = 2.4
    fake = FakeRenderer({"/tmp/voice.wav": 1.2})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "si1",
        "duration": 1.0,
        "image": {"local_path": "/tmp/bg.jpg"},
        "voice_over": {"local_path": "/tmp/voice.wav", "start_delay": 0.3, "end_delay": 0.4},
        "transition_in": {"duration": 0.2},
        "transition_out": {"duration": 0.3},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    assert pytest.approx(spec["duration"], rel=1e-6) == 2.4


@pytest.mark.asyncio
async def test_video_transition_out_starts_after_core():
    # media=2.0, start=0.3, end=0.4, fade_in=0.5, fade_out=0.6
    # core = start + media + end = 0.3 + 2.0 + 0.4 = 2.7
    # fade_out_start = fade_in + core = 0.5 + 2.7 = 3.2
    fake = FakeRenderer({"/tmp/clip.mp4": 2.0})
    step = CreateSegmentClipsStep(renderer=fake)  # type: ignore[arg-type]

    segment = {
        "id": "sv3",
        "video": {"local_path": "/tmp/clip.mp4"},
        "voice_over": {"start_delay": 0.3, "end_delay": 0.4},
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.6},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    spec = fake.captured_specs[0]
    trans_out = [t for t in spec["transformations"] if t.get("type") == "transition" and t.get("direction") == "out"]
    assert len(trans_out) == 2  # visual + audio
    starts = sorted(set(float(t.get("start", -1)) for t in trans_out))
    assert len(starts) == 1
    assert pytest.approx(starts[0], rel=1e-6) == 3.2
