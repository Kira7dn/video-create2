from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.concatenate_video import ConcatenateVideoStep
from app.infrastructure.adapters.renderer_ffmpeg import FFMpegVideoRenderer
from app.core.exceptions import ProcessingError

pytestmark = pytest.mark.integration


ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def _make_color_clip(
    path: str, seconds: float = 1.0, size: str = "160x120", color: str = "green"
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:d={seconds}",
            path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return float(out.stdout.strip())


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concatenate_two_clips_with_real_ffmpeg(tmp_path):
    clip1 = str(tmp_path / "c1.mp4")
    clip2 = str(tmp_path / "c2.mp4")
    _make_color_clip(clip1, seconds=1.0, color="red")
    _make_color_clip(clip2, seconds=1.2, color="blue")

    ctx = PipelineContext(input={})
    ctx.set(
        "segment_clips",
        [
            {"id": "s1", "path": clip1},
            {"id": "s2", "path": clip2},
        ],
    )
    ctx.set("concat_transition", None)

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)

    dur = _probe_duration(out_path)
    # allow some muxing overhead tolerance
    assert 2.0 <= dur <= 2.6


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concatenate_raises_on_missing_inputs(tmp_path):
    missing1 = str(tmp_path / "missing1.mp4")
    # make only one existing
    clip_ok = str(tmp_path / "ok.mp4")
    _make_color_clip(clip_ok, seconds=0.5, color="yellow")

    ctx = PipelineContext(input={})
    ctx.set(
        "segment_clips",
        [
            {"id": "m1", "path": missing1},
            {"id": "ok", "path": clip_ok},
        ],
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    with pytest.raises(ProcessingError):
        await step(ctx)
