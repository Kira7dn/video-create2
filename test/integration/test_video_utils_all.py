from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.concatenate_video import ConcatenateVideoStep
from app.infrastructure.adapters.renderer.utils.ffmpeg_utils import ffmpeg_concat_videos
from app.infrastructure.adapters.renderer_ffmpeg import FFMpegVideoRenderer


# -----------------------------
# Common helpers and markers
# -----------------------------
pytestmark = pytest.mark.integration


def run(cmd):
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )


def have_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except Exception:
        return False


ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def mean_volume_db_for_segment(path: str, start: float, duration: float) -> float:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        path,
        "-af",
        "volumedetect",
        "-f",
        "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = (res.stderr or "") + (res.stdout or "")
    m = re.search(r"mean_volume:\s*([-\d\.]+) dB", text)
    assert m, f"Expected volumedetect output, got: {text[-300:]}"
    return float(m.group(1))


def _make_color_clip_with_silent_audio(
    path: str, seconds: float = 2.0, size: str = "160x120", color: str = "green"
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _make_sine_audio(path: str, seconds: float = 3.0, freq: int = 440) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={seconds}",
            "-c:a",
            "aac",
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


def _has_audio_stream(path: str) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-select_streams",
            "a",
            "-of",
            "json",
            path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return '"codec_type": "audio"' in out.stdout


def _silence_regions(
    path: str, noise_db: int = -35, min_silence_d: float = 0.3
) -> list[tuple[float, float]]:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-af",
            f"silencedetect=n={noise_db}dB:d={min_silence_d}",
            "-f",
            "null",
            "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        line = line.strip()
        if "silence_start" in line:
            try:
                start = float(line.split("silence_start:")[-1].strip())
            except Exception:
                start = None
        elif "silence_end" in line and start is not None:
            try:
                parts = line.split("silence_end:")[-1].strip().split(" ")
                end = float(parts[0])
                silences.append((start, end))
            except Exception:
                pass
            finally:
                start = None
    return silences


# -----------------------------
# Smoke test: subprocess runs OK
# -----------------------------
@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
def test_ffmpeg_mixing_subprocess_smoke(tmp_path):
    tmp_path = Path(tmp_path)

    # 1) Create a short 3s black video with simple tone audio (acts as voice track)
    voice_path = tmp_path / "voice.wav"
    video_path = tmp_path / "video.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            str(voice_path),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:d=3",
            "-i",
            str(voice_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video_path),
        ]
    )

    # 2) Create a 5s BGM tone (will be normalized + mixed)
    bgm_path = tmp_path / "bgm.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=5",
            str(bgm_path),
        ]
    )

    # 3) Execute concat + mix with ducking enabled. Expect no exceptions
    out_path = tmp_path / "out.mp4"
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    segments = [{"id": "s1", "path": str(video_path)}]
    ffmpeg_concat_videos(
        video_segments=segments,
        output_path=str(out_path),
        temp_dir=str(workdir),
        background_music={
            "local_path": str(bgm_path),
            "bgm_volume": 0.4,
            "ducking": True,
        },
    )

    # 4) Assert output exists and is non-empty and ffprobe can read duration
    assert out_path.exists() and out_path.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert probe.returncode == 0, f"ffprobe failed: {probe.stderr}"
    assert float(probe.stdout.strip()) > 0.0


# -----------------------------
# Ducking integration test
# -----------------------------
@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.skip(reason="auto-volume disabled by default")
def test_ducking_reduces_mix_loudness_when_voice_present(tmp_path):
    tmp_path = Path(tmp_path)

    # 1) Create an 8s voice track: 0-4s tone (-12 dBFS, loud), 4-8s silence
    tone_path = tmp_path / "voice_tone.wav"
    silence_path = tmp_path / "silence.wav"
    voice_path = tmp_path / "voice_6s.wav"

    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
            "-filter:a",
            "volume=-12dB",
            str(tone_path),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "4",
            str(silence_path),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(tone_path),
            "-i",
            str(silence_path),
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map",
            "[out]",
            str(voice_path),
        ]
    )

    # 2) Wrap into a simple 8s black video with this audio
    video_path = tmp_path / "segment_video.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:d=8",
            "-i",
            str(voice_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video_path),
        ]
    )

    # 3) Create BGM ~12s sine wave
    bgm_path = tmp_path / "bgm.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=12",
            str(bgm_path),
        ]
    )

    # 4) Prepare inputs
    segments = [{"id": "s1", "path": str(video_path)}]

    # 5) Run with ducking enabled
    out_duck = tmp_path / "out_duck.mp4"
    tmp_duck_dir = tmp_path / "tmp_duck"
    tmp_duck_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_concat_videos(
        video_segments=segments,
        output_path=str(out_duck),
        temp_dir=str(tmp_duck_dir),
        background_music={
            "local_path": str(bgm_path),
            "bgm_volume": 0.7,
            "ducking": True,
        },
    )

    # 6) Run with ducking disabled
    out_noduck = tmp_path / "out_noduck.mp4"
    tmp_noduck_dir = tmp_path / "tmp_noduck"
    tmp_noduck_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_concat_videos(
        video_segments=segments,
        output_path=str(out_noduck),
        temp_dir=str(tmp_noduck_dir),
        background_music={
            "local_path": str(bgm_path),
            "bgm_volume": 0.7,
            "ducking": False,
        },
    )

    # 7) Analyze mean volume
    V_duck_voice = mean_volume_db_for_segment(str(out_duck), 0, 4)
    V_duck_novoice = mean_volume_db_for_segment(str(out_duck), 4, 4)
    V_noduck_voice = mean_volume_db_for_segment(str(out_noduck), 0, 4)
    V_noduck_novoice = mean_volume_db_for_segment(str(out_noduck), 4, 4)

    D_voice = V_noduck_voice - V_duck_voice
    D_novoice = V_noduck_novoice - V_duck_novoice

    assert D_voice >= 1.0, (
        f"Insufficient ducking during voice: D_voice={D_voice} dB;"
        f" values: duck={V_duck_voice} dB, no_duck={V_noduck_voice} dB"
    )
    assert abs(D_novoice) <= 0.8, (
        f"Unexpected change without voice: D_novoice={D_novoice} dB;"
        f" values: duck={V_duck_novoice} dB, no_duck={V_noduck_novoice} dB"
    )


# -----------------------------
# Loudness integration test
# -----------------------------
@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.skip(reason="auto-volume disabled by default")
def test_loudnorm_makes_output_near_target(tmp_path):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"

    # 1) Create a short silent video with silent audio (so mix = BGM only)
    main_vid = tmp_path / "main.mp4"
    make_main = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=3",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-shortest",
        "-c:v",
        "libx264",
        "-t",
        "3",
        "-c:a",
        "aac",
        str(main_vid),
    ]
    subprocess.run(
        make_main, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
    )

    # 2) Create a BGM tone (sine) with arbitrary level
    bgm = tmp_path / "bgm.wav"
    make_bgm = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=5",
        "-filter:a",
        "volume=0.7",
        str(bgm),
    ]
    subprocess.run(
        make_bgm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
    )

    segments = [{"id": "s1", "path": str(main_vid)}]
    background_music = {
        "local_path": str(bgm),
        "start_delay": 0.0,
        "end_delay": 0.0,
        "bgm_volume": 0.2,
    }

    # 3) Run our function (uses loudnorm hardcoded targets)
    ffmpeg_concat_videos(
        segments, str(out), str(temp), background_music=background_music
    )

    assert out.exists() and out.stat().st_size > 0

    # 4) Analyze final audio loudness using loudnorm (analysis mode)
    analyze = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(out),
        "-af",
        "loudnorm=I=-16.0:TP=-1.5:LRA=11.0:print_format=json",
        "-f",
        "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    res = subprocess.run(
        analyze, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    text = (res.stderr or "") + (res.stdout or "")

    matches = list(re.finditer(r"\{[\s\S]*?\}", text))
    assert matches, "Expected loudnorm analysis JSON in ffmpeg output"
    data = json.loads(matches[-1].group(0))

    input_i = float(data["input_i"])  # integrated loudness
    input_tp = float(data.get("input_tp", -10))

    expected = -16.0 + 20.0 * math.log10(0.2)
    tol = 2.0
    assert (
        (expected - tol) <= input_i <= (expected + tol)
    ), f"Integrated loudness {input_i} not within [{expected - tol}, {expected + tol}]"

    assert input_tp <= 0.0, f"True peak too high: {input_tp} dBTP"


# -----------------------------
# Pipeline BGM concatenate tests
# -----------------------------
@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concat_with_bgm_basic(tmp_path):
    # Prepare two video clips (with silent audio streams)
    clip1 = str(tmp_path / "c1.mp4")
    clip2 = str(tmp_path / "c2.mp4")
    _make_color_clip_with_silent_audio(clip1, seconds=1.0, color="red")
    _make_color_clip_with_silent_audio(clip2, seconds=1.5, color="blue")

    # Prepare BGM (sine tone)
    bgm = str(tmp_path / "bgm.m4a")
    _make_sine_audio(bgm, seconds=3.0)

    ctx = PipelineContext(input={})
    ctx.set(
        "segment_clips",
        [
            {"id": "s1", "path": clip1},
            {"id": "s2", "path": clip2},
        ],
    )
    ctx.set("concat_transition", None)
    ctx.set(
        "background_music",
        {"local_path": bgm, "start_delay": 0.0, "end_delay": 0.0},
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)
    assert _has_audio_stream(out_path)  # BGM mixed in

    dur = _probe_duration(out_path)
    # allow small overhead
    assert 2.4 <= dur <= 3.0


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concat_with_bgm_delays(tmp_path):
    # Video ~4s total
    clip1 = str(tmp_path / "c1.mp4")
    clip2 = str(tmp_path / "c2.mp4")
    _make_color_clip_with_silent_audio(clip1, seconds=2.0, color="green")
    _make_color_clip_with_silent_audio(clip2, seconds=2.0, color="yellow")

    # BGM 1s (timing focus)
    bgm = str(tmp_path / "bgm.m4a")
    _make_sine_audio(bgm, seconds=1.0)

    start_delay = 1.0
    end_delay = 0.5

    ctx = PipelineContext(input={})
    ctx.set(
        "segment_clips",
        [
            {"id": "s1", "path": clip1},
            {"id": "s2", "path": clip2},
        ],
    )
    ctx.set(
        "background_music",
        {"local_path": bgm, "start_delay": start_delay, "end_delay": end_delay},
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)
    assert _has_audio_stream(out_path)

    total_video = _probe_duration(out_path)
    assert 3.5 <= total_video <= 4.5


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concat_bgm_skipped_when_too_short(tmp_path):
    # Video ~1.0s
    clip1 = str(tmp_path / "c1.mp4")
    _make_color_clip_with_silent_audio(clip1, seconds=1.0, color="purple")

    # BGM present but start_delay makes play duration <= 0.1s -> skip BGM
    bgm = str(tmp_path / "bgm.m4a")
    _make_sine_audio(bgm, seconds=3.0)

    ctx = PipelineContext(input={})
    ctx.set("segment_clips", [{"id": "s1", "path": clip1}])
    ctx.set(
        "background_music", {"local_path": bgm, "start_delay": 1.0, "end_delay": 0.0}
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)

    dur = _probe_duration(out_path)
    assert 0.9 <= dur <= 1.4


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concat_with_bgm_loops_to_cover_video(tmp_path):
    # Video ~4s total
    clip1 = str(tmp_path / "c1.mp4")
    clip2 = str(tmp_path / "c2.mp4")
    _make_color_clip_with_silent_audio(clip1, seconds=2.0, color="green")
    _make_color_clip_with_silent_audio(clip2, seconds=2.0, color="yellow")

    # BGM very short (0.5s) -> must loop to cover most of the video
    bgm = str(tmp_path / "bgm.m4a")
    _make_sine_audio(bgm, seconds=0.5)

    ctx = PipelineContext(input={})
    ctx.set(
        "segment_clips",
        [
            {"id": "s1", "path": clip1},
            {"id": "s2", "path": clip2},
        ],
    )
    ctx.set(
        "background_music", {"local_path": bgm, "start_delay": 0.0, "end_delay": 0.0}
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)
    assert _has_audio_stream(out_path)

    total_video = _probe_duration(out_path)
    silences = _silence_regions(out_path, noise_db=-35, min_silence_d=0.4)

    max_silence = 0.0
    for s, e in silences:
        max_silence = max(max_silence, e - s)
    assert max_silence < 1.5
    assert 3.5 <= total_video <= 4.5


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concat_with_bgm_trim_when_longer(tmp_path):
    # Video total ~2.0s
    clip1 = str(tmp_path / "c1.mp4")
    clip2 = str(tmp_path / "c2.mp4")
    _make_color_clip_with_silent_audio(clip1, seconds=1.0, color="red")
    _make_color_clip_with_silent_audio(clip2, seconds=1.0, color="blue")

    # BGM longer (5s) -> should be trimmed to video duration
    bgm = str(tmp_path / "bgm.m4a")
    _make_sine_audio(bgm, seconds=5.0)

    ctx = PipelineContext(input={})
    ctx.set(
        "segment_clips",
        [
            {"id": "s1", "path": clip1},
            {"id": "s2", "path": clip2},
        ],
    )
    ctx.set(
        "background_music", {"local_path": bgm, "start_delay": 0.0, "end_delay": 0.0}
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)
    assert _has_audio_stream(out_path)

    total_video = _probe_duration(out_path)
    assert 1.8 <= total_video <= 2.4

    silences = _silence_regions(out_path, noise_db=-35, min_silence_d=0.4)
    max_silence = 0.0
    for s, e in silences:
        max_silence = max(max_silence, e - s)
    assert max_silence < 1.0


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concat_bgm_skipped_when_end_delay_exceeds(tmp_path):
    # Video ~1.5s
    clip1 = str(tmp_path / "c1.mp4")
    _make_color_clip_with_silent_audio(clip1, seconds=1.5, color="pink")

    # BGM exists but end_delay so large that play duration <= 0 -> should skip BGM
    bgm = str(tmp_path / "bgm.m4a")
    _make_sine_audio(bgm, seconds=3.0)

    ctx = PipelineContext(input={})
    ctx.set("segment_clips", [{"id": "s1", "path": clip1}])
    ctx.set(
        "background_music", {"local_path": bgm, "start_delay": 0.0, "end_delay": 2.0}
    )

    step = ConcatenateVideoStep(renderer=FFMpegVideoRenderer(temp_dir=str(tmp_path)))  # type: ignore[arg-type]
    await step(ctx)

    out_path = ctx.get("final_video_path")
    assert isinstance(out_path, str) and os.path.exists(out_path)
    dur = _probe_duration(out_path)
    assert 1.3 <= dur <= 1.8
