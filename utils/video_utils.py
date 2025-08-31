"""
Video processing utilities for video creation.

This module provides functions for video processing tasks such as concatenation,
transitions, and audio mixing using FFmpeg.
"""

import json
import os
import re
import math
import shutil
from typing import List, Optional, Dict, Any

from utils.subprocess_utils import safe_subprocess_run, SubprocessError


class VideoProcessingError(SubprocessError):
    """Custom exception for video processing errors."""


def ffmpeg_concat_videos(
    video_segments: List[Dict[str, str]],
    output_path: str,
    temp_dir: str,
    background_music: Optional[dict] = None,
    logger: Optional[Any] = None,
    bgm_volume: float = 0.2,
) -> None:
    """
    Concatenate video segments with per-pair transitions, then overlay
    background music (with start_delay, end_delay) as in input_sample.json.

    Args:
        video_segments: list of dicts with 'id' and 'path'.
        transitions: list of dicts with 'type', 'duration', 'from_segment', 'to_segment'.
        background_music: dict with 'url' or 'local_path', 'start_delay', 'end_delay'.
        output_path: Path to save the output video.
        temp_dir: Directory for temporary files.
        logger: Optional logger instance for logging messages.
        default_transition_type: Default transition type if not specified.
        default_transition_duration: Default transition duration in seconds.
        bgm_volume: Volume level for background music (0.0 to 1.0).
    """

    def get_duration(path):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
        try:
            result = safe_subprocess_run(cmd, f"Get duration for {path}", logger)
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            error_msg = f"Failed to parse duration from {path}: {e}"
            if logger:
                logger.error(error_msg)
            raise VideoProcessingError(error_msg) from e

    def validate_inputs():
        """Validate input parameters"""
        if not video_segments:
            raise VideoProcessingError("video_segments cannot be empty")

        if len(video_segments) < 1:
            raise VideoProcessingError("At least 1 video segment is required")

        for i, seg in enumerate(video_segments):
            if not isinstance(seg, dict):
                raise VideoProcessingError(f"Segment {i} must be a dictionary")
            if "path" not in seg:
                raise VideoProcessingError(f"Segment {i} missing 'path' field")
            if not os.path.exists(seg["path"]):
                raise VideoProcessingError(f"Video file not found: {seg['path']}")

        # Validate output directory exists and is writable
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            raise VideoProcessingError(f"Output directory not found: {output_dir}")

        # Check write permissions (simplified for production)
        if output_dir:
            try:
                # Quick check without creating temporary file
                if not os.access(output_dir, os.W_OK):
                    raise VideoProcessingError(
                        f"No write permission for output directory: {output_dir}"
                    )
            except (OSError, PermissionError) as e:
                error_msg = f"Cannot write to output directory {output_dir}: {e}"
                raise VideoProcessingError(error_msg) from e

    # Validate inputs before processing
    validate_inputs()

    # 1. Concat segments
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for seg in video_segments:
            f.write(f"file '{os.path.abspath(seg['path'])}'\n")
    temp_path = os.path.join(temp_dir, "concat_output.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "1",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_path,
        "-c",
        "copy",
        temp_path,
    ]
    safe_subprocess_run(ffmpeg_cmd, "Concat without transition", logger)
    if logger:
        logger.info(f"Final video concat: {temp_path}")

    # 2. Overlay background music if provided
    if background_music and background_music.get("local_path"):
        bgm_path = background_music.get("local_path")
        start_delay = float(background_music.get("start_delay", 0) or 0)
        video_duration = get_duration(temp_path)
        if logger:
            logger.info("🔈 Auto-volume: loudnorm (default)")

        # Always pre-normalize BGM using EBU R128 loudness (two-pass when possible)
        # Hardcoded targets as requested
        target_i = -16.0  # LUFS
        target_tp = -1.5  # dBTP
        target_lra = 11.0  # LU

        def _analyze_loudness(path: str) -> Optional[dict]:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-i",
                path,
                "-af",
                f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
                "-f",
                "null",
                "NUL" if os.name == "nt" else "/dev/null",
            ]
            try:
                res = safe_subprocess_run(cmd, f"Loudnorm analyze for {path}", logger)
                text = (res.stderr or "") + (res.stdout or "")
                # Extract last JSON object in output (ffmpeg prints to stderr)
                matches = list(re.finditer(r"\{[\s\S]*?\}", text))
                if not matches:
                    return None
                data = json.loads(matches[-1].group(0))
                needed = {
                    "input_i": data.get("input_i"),
                    "input_lra": data.get("input_lra"),
                    "input_tp": data.get("input_tp"),
                    "input_thresh": data.get("input_thresh"),
                    "target_offset": data.get("target_offset"),
                }
                if all(v is not None for v in needed.values()):
                    return needed
                return None
            except Exception as e:
                if logger:
                    logger.warning(f"Loudnorm analyze failed: {e}")
                return None

        analysis = _analyze_loudness(bgm_path)
        try:
            normalized_bgm_path = os.path.join(temp_dir, "bgm_loudnorm.wav")
            if analysis:
                # Two-pass with measured values
                ln_filter = (
                    "loudnorm="
                    f"I={target_i}:TP={target_tp}:LRA={target_lra}:"
                    f"measured_I={analysis['input_i']}:"
                    f"measured_LRA={analysis['input_lra']}:"
                    f"measured_TP={analysis['input_tp']}:"
                    f"measured_thresh={analysis['input_thresh']}:"
                    f"offset={analysis['target_offset']}:"
                    "linear=true:print_format=summary"
                )
            else:
                # Fallback to single-pass loudnorm
                ln_filter = (
                    "loudnorm="
                    f"I={target_i}:TP={target_tp}:LRA={target_lra}:"
                    "linear=true:print_format=summary"
                )
            cmd_norm = [
                "ffmpeg",
                "-y",
                "-i",
                bgm_path,
                "-af",
                ln_filter,
                "-vn",
                "-acodec",
                "pcm_s16le",
                normalized_bgm_path,
            ]
            safe_subprocess_run(cmd_norm, "Normalize BGM loudness", logger)
            if logger:
                logger.info(
                    f"🎚️ BGM loudness normalized to I={target_i} LUFS, TP={target_tp} dB"
                )
            bgm_path = normalized_bgm_path
        except Exception as e:
            if logger:
                logger.warning(f"Loudnorm normalization failed, continue without: {e}")
            normalized_bgm_path = None

        # Mix level for BGM after normalization (no auto detection needed)
        try:
            bgm_volume_factor = float(background_music.get("bgm_volume", bgm_volume))
        except (TypeError, ValueError):
            bgm_volume_factor = bgm_volume
        # Prepare filter for bgm: delay, trim, volume
        # Calculate actual BGM play duration considering both start_delay and end_delay
        end_delay = float(background_music.get("end_delay", 0) or 0)
        bgm_start_time = start_delay
        bgm_end_time = max(0, video_duration - end_delay)
        bgm_play_duration = max(0, bgm_end_time - bgm_start_time)

        if logger:
            logger.info(
                f"🎵 BGM timing: video_duration={video_duration:.2f}s, "
                f"start_delay={start_delay:.2f}s, end_delay={end_delay:.2f}s, "
                f"bgm_play_duration={bgm_play_duration:.2f}s"
            )

        # Safety check: if BGM duration is too small or invalid, skip BGM processing
        if bgm_play_duration <= 0.1:  # Less than 100ms
            if logger:
                logger.warning(
                    f"⚠️ BGM play duration too short ({bgm_play_duration:.2f}s), "
                    "skipping background music"
                )
            # Copy temp_path to output without BGM processing
            if os.path.exists(output_path):
                raise VideoProcessingError(f"Output file already exists: {output_path}")
            shutil.copy2(temp_path, output_path)
            return

        filter_parts = []
        if start_delay > 0:
            delay_ms = int(start_delay * 1000)
            filter_parts.append(f"adelay={delay_ms}|{delay_ms}")

        # Only trim if we have a valid duration to trim to
        if bgm_play_duration > 0:
            filter_parts.append(f"atrim=duration={bgm_play_duration}")
        filter_parts.append(f"volume={bgm_volume_factor}")
        bgm_filter = ",".join(filter_parts)
        # Compute how many loops of bgm are required to cover desired play duration
        # We will loop the bgm source just enough to exceed bgm_play_duration, then trim
        bgm_src_duration = None
        try:
            bgm_src_duration = get_duration(bgm_path)
        except VideoProcessingError:
            # If we cannot probe duration, fallback to single pass (no loop)
            bgm_src_duration = None

        # Compose filter_complex for direct mix
        # Use 'duration=first' to tie mixed audio to the video audio (0:a)
        # Add optional ducking via sidechaincompress (default: enabled)
        # Final limiter to avoid clipping
        enable_ducking = bool(background_music.get("ducking", True))
        if enable_ducking and logger:
            logger.info("🪝 Ducking enabled (sidechaincompress)")

        if enable_ducking:
            # Music as main input, video audio (0:a) as sidechain key
            filter_complex = (
                f"[1:a]{bgm_filter}[bgm]; "
                # Stronger ducking: lower threshold, higher ratio, faster attack, longer release
                f"[bgm][0:a]sidechaincompress=threshold=-35dB:ratio=20:attack=5:release=400:knee=8:link=average:level_sc=1:mix=1[bgmsc]; "
                f"[0:a][bgmsc]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
                f"alimiter=limit=0.95[aout]"
            )
        else:
            filter_complex = (
                f"[1:a]{bgm_filter}[bgm]; "
                f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
                f"alimiter=limit=0.95[aout]"
            )
        temp_final_with_bgm = os.path.join(temp_dir, "final_with_bgm.mp4")
        ffmpeg_mix_cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            "1",
            "-i",
            temp_path,
        ]

        # If we know bgm duration and it's shorter than desired play, loop the input enough times
        if bgm_src_duration and bgm_src_duration > 0 and bgm_play_duration > 0:
            loops_needed = max(0, math.ceil(bgm_play_duration / bgm_src_duration) - 1)
            if loops_needed > 0:
                ffmpeg_mix_cmd.extend(["-stream_loop", str(loops_needed)])

        ffmpeg_mix_cmd.extend(
            [
                "-i",
                bgm_path,
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                temp_final_with_bgm,
            ]
        )
        if logger:
            logger.info(f"Mixing BGM (atomic operation): {' '.join(ffmpeg_mix_cmd)}")
        safe_subprocess_run(ffmpeg_mix_cmd, "Background music mixing", logger)
        # Final output is temp_final_with_bgm
        temp_path = temp_final_with_bgm

    # 3. Copy final result to output_path (do not overwrite if exists)
    if os.path.exists(output_path):
        raise VideoProcessingError(f"Output file already exists: {output_path}")
    shutil.copy2(temp_path, output_path)
