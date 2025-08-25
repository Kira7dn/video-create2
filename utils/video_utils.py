"""
Video processing utilities for video creation.

This module provides functions for video processing tasks such as concatenation,
transitions, and audio mixing using FFmpeg.
"""

import json
import os
import re
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

    def get_mean_volume(audio_path):
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-af",
            "volumedetect",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]
        try:
            result = safe_subprocess_run(
                cmd, f"Get mean volume for {audio_path}", logger
            )
            if not result or not result.stderr:
                return None
            match = re.search(r"mean_volume:\s*(-?\d+(\.\d+)?) dB", result.stderr)
            if match:
                return float(match.group(1))
            return None
        except (OSError, SubprocessError, ValueError) as e:
            if logger:
                logger.warning(f"Failed to get mean volume for {audio_path}: {e}")
            return None

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
        # Auto adjust bgm volume based on mean_volume
        try:
            video_mean_volume = get_mean_volume(temp_path)
            music_mean_volume = get_mean_volume(bgm_path)
            if video_mean_volume is not None and music_mean_volume is not None:
                diff_db = video_mean_volume - music_mean_volume
                bgm_volume_factor = 10 ** (diff_db / 20)
                bgm_volume_factor = max(0.1, min(bgm_volume_factor, 0.5))
                if logger:
                    log_msg = (
                        f"🔊 Auto-adjusted bgm volume factor: {bgm_volume_factor:.2f}"
                    )
                    log_msg += f" (video_mean={video_mean_volume}dB, "
                    log_msg += f"music_mean={music_mean_volume}dB)"
                    logger.info(log_msg)
            else:
                bgm_volume_factor = bgm_volume
                if logger:
                    logger.warning(
                        "⚠️ Could not auto-detect mean_volume, using default bgm volume 0.2"
                    )
        except (OSError, SubprocessError, ValueError) as e:
            bgm_volume_factor = bgm_volume
            if logger:
                logger.warning(
                    f"⚠️ Error auto-adjusting bgm volume: {e}, using default 0.2"
                )
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
        filter_complex = (
            f"[1:a]{bgm_filter}[bgm]; "
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
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
            import math

            loops_needed = max(0, math.ceil(bgm_play_duration / bgm_src_duration) - 1)
            if loops_needed > 0:
                ffmpeg_mix_cmd.extend(["-stream_loop", str(loops_needed)])

        ffmpeg_mix_cmd.extend([
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
        ])
        if logger:
            logger.info(f"Mixing BGM (atomic operation): {' '.join(ffmpeg_mix_cmd)}")
        safe_subprocess_run(ffmpeg_mix_cmd, "Background music mixing", logger)
        # Final output is temp_final_with_bgm
        temp_path = temp_final_with_bgm

    # 3. Copy final result to output_path (do not overwrite if exists)
    if os.path.exists(output_path):
        raise VideoProcessingError(f"Output file already exists: {output_path}")
    shutil.copy2(temp_path, output_path)
