from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.create_segment_clips import (
    CreateSegmentClipsStep,
)
from app.infrastructure.adapters.renderer_ffmpeg import FFMpegVideoRenderer

pytestmark = pytest.mark.integration

ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.asyncio
async def test_video_segment_with_real_ffmpeg_renderer(tmp_path):
    # Generate a 2s color video as input
    src_video = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=2",
            str(src_video),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Generate a short 2s sine audio to act as voice_over
    voice_path = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=2",
            str(voice_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Enable ffmpeg filtergraph debug to aid troubleshooting
    from app.core.config import settings

    try:
        setattr(settings, "debug_ffmpeg_cmd", True)
    except Exception:
        pass

    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))
    step = CreateSegmentClipsStep(renderer=renderer)  # type: ignore[arg-type]

    segment = {
        "id": "v1",
        "video": {"local_path": str(src_video)},
        # fades shouldn't change total for video case
        "transition_in": {"duration": 0.5},
        "transition_out": {"duration": 0.5},
        "voice_over": {"local_path": str(voice_path)},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])

    await step(ctx)

    clips = ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 1
    out_path = clips[0]["path"]
    assert os.path.exists(out_path)

    # Probe the output to ensure duration ~= 2s
    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            out_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dur = float(probed.stdout.strip())
    assert 1.8 <= dur <= 2.2


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.asyncio
async def test_image_segment_with_voice_over_and_fades_real_ffmpeg(tmp_path):
    # Create background image
    img_path = tmp_path / "bg.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=128x128",
            "-frames:v",
            "1",
            str(img_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Create a 1.2s sine audio as voice-over
    voice_path = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1.2",
            str(voice_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))
    step = CreateSegmentClipsStep(renderer=renderer)  # type: ignore[arg-type]

    # base=1.0, voice=1.2 -> max=1.2; fades 0.2+0.3 -> total=1.7
    segment = {
        "id": "i_voice",
        "duration": 1.0,
        "image": {"local_path": str(img_path)},
        "voice_over": {"local_path": str(voice_path)},
        "transition_in": {"duration": 0.2},
        "transition_out": {"duration": 0.3},
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    clips = ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 1
    out_path = clips[0]["path"]
    assert os.path.exists(out_path)

    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            out_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dur = float(probed.stdout.strip())
    assert 1.5 <= dur <= 1.9


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.asyncio
async def test_image_segment_with_text_overlay_real_ffmpeg(tmp_path):
    # Create background image
    img_path = tmp_path / "bg.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=128x128",
            "-frames:v",
            "1",
            str(img_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))
    step = CreateSegmentClipsStep(renderer=renderer)  # type: ignore[arg-type]

    # base=1.0 with fades 0.1+0.2 -> total=1.3
    segment = {
        "id": "i_text",
        "duration": 1.0,
        "image": {"local_path": str(img_path)},
        "transition_in": {"duration": 0.1},
        "transition_out": {"duration": 0.2},
        "text_over": [
            {
                "text": "Hello",
                "start": 0.2,
                "duration": 0.8,
                "font_size": 20,
                "color": "white",
                "x": "(w-text_w)/2",
                "y": "(h-text_h)/2",
                "box": True,
                "boxcolor": "black@0.4",
            }
        ],
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    clips = ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 1
    out_path = clips[0]["path"]
    assert os.path.exists(out_path)

    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            out_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dur = float(probed.stdout.strip())
    assert 1.1 <= dur <= 1.5


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.asyncio
async def test_image_segment_with_multiple_text_overlays_commas_real_ffmpeg(
    tmp_path, transcript_voice_pairs
):
    """
    Reproduce an FFmpeg filtergraph similar to e2e with many drawtext entries where:
    - text contains commas and apostrophes
    - enable expressions use decimal start/end times
    This aims to catch parser issues like: Invalid chars ',drawtext=...' at end of 'between(...)'
    """
    # Create background image
    img_path = tmp_path / "bg.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=128x128",
            "-frames:v",
            "1",
            str(img_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Use real voice_over and transcript from fixture
    if not transcript_voice_pairs:
        pytest.skip(
            "No transcript/voice pairs discovered under test/temp; run a job to populate."
        )
    sample = transcript_voice_pairs[0]
    voice_path = sample["voice"]
    # Load transcript lines (fall back to synthetic lines if read fails)
    texts: list[str]
    try:
        with open(sample["transcript"], "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # Ensure we have at least a handful; inject commas/apostrophes to stress parsing
        texts = (
            lines[:8]
            if len(lines) >= 8
            else (
                lines + ["Today, we're stress-testing the latest", "AI tools, really?"]
            )[:8]
        )
        # Guarantee presence of commas/apostrophes in some entries
        if texts:
            texts[0] = texts[0] + ", ok?"
        if len(texts) > 5:
            texts[5] = texts[5] + ", we're in!"
    except Exception:
        texts = [
            "Imagine having an AI assistant",
            "that not only drafts your emails",
            "and schedules your meetings",
            "but learns your work habits",
            "to save you hours every week.",
            "Today, we're stress-testing the latest",
            "AI productivity tools to see if",
            "they're truly revolutionizing the modern office—or just hyped.",
        ]

    # Start times mirroring e2e-like decimals; durations ~1.5-2.0s to overlap lightly
    starts = [1.55, 3.12, 4.54, 6.07, 7.36, 10.64, 12.45, 14.28]
    durs = [1.57, 1.42, 1.29, 1.29, 1.65, 1.80, 1.83, 3.60]

    text_over = []
    for t, st, du in zip(texts, starts, durs):
        text_over.append(
            {
                "text": t,  # contains commas / apostrophes in some entries
                "start": st,
                "duration": du,
                "font_size": 42,
                "color": "white",
                "x": "(w-text_w)/2",
                "y": "h-text_h-0.08*h",
                "box": True,
                "boxcolor": "black@0.4",
            }
        )

    # Total duration — make sure long enough to include last overlay and add fades
    base_duration = max(st + du for st, du in zip(starts, durs))
    fade_in = 0.5
    fade_out = 0.5
    total_duration = base_duration + fade_in + fade_out

    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))
    step = CreateSegmentClipsStep(renderer=renderer)  # type: ignore[arg-type]

    segment = {
        "id": "i_text_many",
        "duration": float(total_duration),
        "image": {"local_path": str(img_path)},
        "transition_in": {"duration": fade_in},
        "transition_out": {"duration": fade_out},
        "text_over": text_over,
    }

    ctx = PipelineContext(input={})
    ctx.set("segments", [segment])
    await step(ctx)

    clips = ctx.get("segment_clips")
    assert isinstance(clips, list) and len(clips) == 1
    out_path = clips[0]["path"]
    assert os.path.exists(out_path)

    # Probe the output to ensure duration spans the overlays window
    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            out_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dur = float(probed.stdout.strip())
    assert dur >= base_duration - 0.5


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.asyncio
async def test_image_segments_with_samples_under_test_temp(
    tmp_path, transcript_voice_pairs
):
    """
    Use multiple real samples discovered under test/temp (via transcript_voice_pairs)
    to validate the renderer behavior across diverse, real-world content.
    For each sample found, we build text_over from transcript lines and use the
    corresponding voice_over.mp3, then render a short clip. We assert each render succeeds.
    """
    if not transcript_voice_pairs:
        pytest.skip("No transcript/voice pairs discovered under test/temp")

    # Create a reusable background image
    img_path = tmp_path / "bg.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=128x128",
            "-frames:v",
            "1",
            str(img_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    renderer = FFMpegVideoRenderer(temp_dir=str(tmp_path))
    step = CreateSegmentClipsStep(renderer=renderer)  # type: ignore[arg-type]

    # Limit to first 5 samples to keep CI time reasonable; adjust as desired
    for idx, sample in enumerate(transcript_voice_pairs[:5]):
        seg_id = f"sample_{idx}"
        voice_path = sample.get("voice")
        transcript_path = sample.get("transcript")
        if not voice_path or not transcript_path:
            continue

        # Load transcript lines safely
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            continue
        if not lines:
            continue

        # Build overlays from up to 8 lines, using incremental timings
        texts = lines[:8]
        starts = [1.0 + i * 1.2 for i in range(len(texts))]
        durs = [1.0 for _ in texts]
        text_over = []
        for t, st, du in zip(texts, starts, durs):
            # include commas/apostrophes if any exist; escape will be handled by renderer
            text_over.append(
                {
                    "text": t,
                    "start": float(st),
                    "duration": float(du),
                    "font_size": 32,
                    "color": "white",
                    "x": "(w-text_w)/2",
                    "y": "h-text_h-0.08*h",
                    "box": True,
                    "boxcolor": "black@0.4",
                }
            )

        # Duration to cover overlays + fades
        base_duration = max(st + du for st, du in zip(starts, durs))
        fade_in = 0.3
        fade_out = 0.3
        total_duration = base_duration + fade_in + fade_out

        segment = {
            "id": seg_id,
            "duration": float(total_duration),
            "image": {"local_path": str(img_path)},
            "voice_over": {"local_path": str(voice_path)},
            "transition_in": {"duration": fade_in},
            "transition_out": {"duration": fade_out},
            "text_over": text_over,
        }

        ctx = PipelineContext(input={})
        ctx.set("segments", [segment])
        await step(ctx)

        clips = ctx.get("segment_clips")
        assert isinstance(clips, list) and len(clips) == 1
        out_path = clips[0]["path"]
        assert os.path.exists(out_path)

        # quick probe
        probed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                out_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        dur = float(probed.stdout.strip())
        assert dur >= base_duration - 0.5


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not available")
@pytest.mark.asyncio
async def test_ffmpeg_error_234_with_malformed_drawtext_enable(tmp_path):
    """
    Purposefully reproduce the FFmpeg parse error (return code 234) by building
    a filtergraph similar to the problematic one observed in e2e logs, but
    WITHOUT using any conftest fixtures. This simulates the old behavior where:
      - drawtext enable is quoted and commas are NOT escaped
      - multiple drawtext filters are chained, so FFmpeg reads the next ',drawtext='
        as part of the enable expression tail -> Invalid chars error.
    """
    import subprocess as sp
    from app.core.config import settings

    # 1) Create background image
    img_path = tmp_path / "bg.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=128x128",
            "-frames:v",
            "1",
            str(img_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 2) Create a short mp3 voice (to mimic e2e input types)
    voice_wav = tmp_path / "voice.wav"
    voice_mp3 = tmp_path / "voice.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=2.0",
            str(voice_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(voice_wav), str(voice_mp3)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 3) Build an intentionally malformed vf with quoted enable and unescaped commas
    # Two drawtext filters; the second starts right after a comma, which should be
    # mis-parsed as part of the previous quoted enable expression tail.
    enable1 = "between(t,12.45,14.280000000000001)"  # not escaped, quoted later
    draw1 = (
        "drawtext="
        "fontfile='fonts/Roboto-Black.ttf':text='Today, we\\'re stress-testing the latest':"
        "fontcolor=white:fontsize=42:x=(w-text_w)/2:y=h-text_h-0.08*h:box=1:boxcolor=black@0.4:"
        f"enable='{enable1}'"
    )
    draw2 = (
        "drawtext="
        "fontfile='fonts/Roboto-Black.ttf':text='AI productivity tools to see if':"
        "fontcolor=white:fontsize=42:x=(w-text_w)/2:y=h-text_h-0.08*h:box=1:boxcolor=black@0.4:"
        "enable='between(t,14.28,17.88)'"
    )
    vf_malformed = ",".join(
        [
            "format=yuv420p",
            "fade=t=in:st=0.0:d=0.5",
            "fade=t=out:st=1.5:d=0.5",
            draw1,
            draw2,
        ]
    )

    # 4) Build a direct ffmpeg command to match renderer structure (image + voice)
    out_path = tmp_path / "out.mp4"
    cmd = [
        settings.ffmpeg_binary_path,
        "-y",
        "-threads",
        str(getattr(settings, "ffmpeg_threads", 1) or 1),
        "-loop",
        "1",
        "-i",
        str(img_path),
        "-i",
        str(voice_mp3),
        "-vf",
        vf_malformed,
        "-af",
        "anull",  # audio chain minimal; focus on vf error
        "-t",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
        "-c:v",
        getattr(settings, "video_default_codec", "libx264"),
        "-c:a",
        getattr(settings, "video_default_audio_codec", "aac"),
        "-b:a",
        "192k",
        str(out_path),
    ]

    # 5) Run and EXPECT failure with CalledProcessError (return code may vary by build)
    with pytest.raises(sp.CalledProcessError) as excinfo:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    # Validate that this is indeed the FFmpeg filter parse problem
    err = excinfo.value
    stderr = getattr(err, "stderr", "") or ""
    # We can't guarantee exact return code across builds; accept common failure signatures
    assert (
        ("Invalid chars" in stderr)
        or ("Error initializing filters" in stderr)
        or ("Invalid argument" in stderr)
        or ("Filter not found" in stderr)
        or ("No such filter" in stderr)
    )
