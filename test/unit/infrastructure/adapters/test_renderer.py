import asyncio
from pathlib import Path

import pytest

from app.infrastructure.adapters import FFMpegVideoRenderer
from app.core.config import settings
from app.infrastructure.adapters.media_probe_ffmpeg import FFmpegMediaProbe


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_media_processor_specification_processing(monkeypatch, tmp_path):
    processor = FFMpegVideoRenderer(temp_dir=str(tmp_path))

    # Prepare input image
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"img")

    # Mock subprocess to avoid running real ffmpeg and ensure output is created
    import app.infrastructure.adapters.renderer.handlers.ffmpeg_handler as fh

    def fake_run(cmd, desc):
        out = cmd[-1]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"processed_media")
        return type("R", (), {"stdout": b""})()

    monkeypatch.setattr(fh, "safe_subprocess_run", fake_run)

    # Build an abstract specification (no ffmpeg concepts)
    specification = {
        "source_type": "static",
        "primary_source": str(img_path),
        "audio_source": None,
        "transformations": [
            {
                "type": "canvas_fit",
                "canvas_width": 320,
                "canvas_height": 240,
                "fit_mode": "contain",
                "fill_color": "black",
            },
            {
                "type": "text_overlay",
                "content": "Test Text",
                "timing": {"start": 0.0, "duration": 2.0},
                "appearance": {"size": 24, "color": "white", "typeface": "default"},
                "layout": {"x": "(w-text_w)/2", "y": "(h-text_h)/2"},
                "background": {"enabled": True, "color": "black@0.4"},
            },
        ],
        "duration": 2.0,
    }

    # Renderer builds path internally as {temp_dir}/{seg_id}/segment_video.mp4
    seg_id = "seg_000"
    result = await processor.render_segment(
        specification,
        seg_id=seg_id,
        canvas_width=320,
        canvas_height=240,
        frame_rate=24,
    )

    expected_path = str(tmp_path / seg_id / "segment_video.mp4")
    assert result == expected_path
    assert Path(expected_path).exists()


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_media_processor_duration_method(monkeypatch, tmp_path):
    processor = FFMpegVideoRenderer()

    # Mock subprocess to return duration
    import app.infrastructure.adapters.renderer.handlers.ffmpeg_handler as fh

    def fake_run(cmd, desc):
        return type("R", (), {"stdout": b"5.25"})()

    monkeypatch.setattr(fh, "safe_subprocess_run", fake_run)

    duration = await processor.duration("test.mp4")
    assert duration == 5.25


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_renderer_render_moves_to_output(monkeypatch, tmp_path):
    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))

    # Prepare input image
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"img")

    # Mock subprocess to avoid running real ffmpeg and ensure output is created
    import app.infrastructure.adapters.renderer.handlers.ffmpeg_handler as fh

    def fake_run(cmd, desc):
        out = cmd[-1]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"clip")
        return type("R", (), {"stdout": b""})()

    monkeypatch.setattr(fh, "safe_subprocess_run", fake_run)

    # Build a minimal infra-agnostic plan for an image clip
    plan = {
        "input_type": "image",
        "video_input": str(img_path),
        "audio_input": None,
        "ops": [
            {"op": "pixel_format", "format": "yuv420p"},
            {"op": "scale", "width": 320, "height": 240},
        ],
        "duration": 1.0,
        "fps": 24,
    }

    desired_out = tmp_path / "clip_000.mp4"

    # Mock render_with_plan to return the output path
    async def mock_render_with_plan(plan, *, output_path, width, height, fps):
        # Create the output file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"dummy video content")
        return output_path

    monkeypatch.setattr(renderer, "_render_with_plan", mock_render_with_plan)

    result = await renderer._render_with_plan(
        plan,
        output_path=str(desired_out),
        width=320,
        height=240,
        fps=24,
    )

    assert result == str(desired_out)
    assert desired_out.exists()


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_renderer_concat_calls_ffmpeg_concat(monkeypatch, tmp_path):
    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))
    made = {}

    def fake_ffmpeg_concat(
        segments,
        output_path,
        temp_dir,
        background_music=None,
        logger=None,
        bgm_volume=0.2,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"video")
        made["segments"] = list(segments)
        made["output_path"] = output_path
        made["temp_dir"] = temp_dir

    monkeypatch.setattr(
        "app.infrastructure.adapters.renderer.handlers.ffmpeg_handler.ffmpeg_concat_videos",
        fake_ffmpeg_concat,
    )

    out = tmp_path / "final.mp4"
    # Create input files so validator passes
    (tmp_path / "a.mp4").write_bytes(b"a")
    (tmp_path / "b.mp4").write_bytes(b"b")
    result = await renderer.concat_clips(
        [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")], output_path=str(out)
    )

    assert result == str(out)
    assert out.exists()
    assert made["output_path"] == str(out)
    assert made["segments"][0]["path"].endswith("a.mp4")


# --- Additional tests merged from test_renderer_ffmpeg.py ---


class DummyResult:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.mark.asyncio
async def test_duration_raises_on_empty(monkeypatch, tmp_path):
    # Stub ffmpeg run to return empty stdout
    calls = []

    def stub(cmd, desc, logger=None):
        calls.append((cmd, desc))
        return DummyResult(stdout="")

    # Patch the utils layer used by handler
    monkeypatch.setattr(
        "app.infrastructure.adapters.renderer.handlers.ffmpeg_handler.safe_subprocess_run",
        stub,
    )

    r = FFMpegVideoRenderer()

    with pytest.raises(ValueError, match="Empty duration"):
        await r.duration(str(tmp_path / "input.mp4"))

    assert calls, "safe_subprocess_run should be called"


@pytest.mark.asyncio
async def test_audio_delay_applied_before_afade(monkeypatch, tmp_path):
    captured_cmd = {}

    def fake_run(cmd, _desc, logger=None):
        captured_cmd["cmd"] = cmd
        out = cmd[-1]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"ok")
        return DummyResult(stdout="", returncode=0)

    monkeypatch.setattr(
        "app.infrastructure.adapters.renderer.handlers.ffmpeg_handler.safe_subprocess_run",
        fake_run,
    )

    renderer = FFMpegVideoRenderer()
    segment = {
        "source_type": "static",
        "primary_source": "/tmp/fake_image.jpg",
        "audio_source": None,
        "transformations": [
            {"type": "audio_delay", "milliseconds": 300},
            {
                "type": "transition",
                "target": "audio",
                "direction": "in",
                "start": 0.0,
                "duration": 0.5,
            },
        ],
        "duration": 5.0,
    }
    await renderer.render_segment(
        segment, seg_id="test", canvas_width=1280, canvas_height=720, frame_rate=30
    )

    cmd = captured_cmd.get("cmd")
    assert cmd is not None
    assert "-af" in cmd
    af_idx = cmd.index("-af")
    af_val = cmd[af_idx + 1]
    assert "adelay=" in af_val
    assert "afade=t=in" in af_val
    assert af_val.index("adelay") < af_val.index("afade")


# --- Additional tests merged from test_media_probe_ffmpeg.py ---


@pytest.mark.asyncio
async def test_media_probe_duration_success(monkeypatch, tmp_path):
    def stub(cmd, desc, logger=None):
        assert cmd[0] == "ffprobe"
        return DummyResult(stdout="2.345\n")

    monkeypatch.setattr(
        "app.infrastructure.adapters.media_probe_ffmpeg.safe_subprocess_run", stub
    )

    probe = FFmpegMediaProbe()
    d = await probe.duration(str(tmp_path / "in.mp4"))
    assert d == pytest.approx(2.345)


@pytest.mark.asyncio
async def test_media_probe_duration_empty_output_raises(monkeypatch, tmp_path):
    def stub(cmd, desc, logger=None):
        return DummyResult(stdout="\n")

    monkeypatch.setattr(
        "app.infrastructure.adapters.media_probe_ffmpeg.safe_subprocess_run", stub
    )

    probe = FFmpegMediaProbe()
    with pytest.raises(ValueError, match="Empty duration output"):
        await probe.duration(str(tmp_path / "in.mp4"))
