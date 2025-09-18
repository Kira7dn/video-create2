import os
import asyncio
from typing import List, Optional, Dict, Any
import logging

from app.core.config import settings
from app.infrastructure.adapters.renderer.utils.ffmpeg_utils import (
    ffmpeg_concat_videos,
    safe_subprocess_run,
)

logger = logging.getLogger(__name__)


class FFmpegHandler:
    """Handler for FFmpeg operations."""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or "."

    def probe_duration(self, input_path: str) -> float:
        """Probe media duration using ffprobe (sync helper suitable for to_thread)."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        result = safe_subprocess_run(cmd, f"Probe duration for {input_path}")
        duration_str = (result.stdout or "").strip()
        if not duration_str:
            raise ValueError("Empty duration output from ffprobe")
        return float(duration_str)

    async def concat_videos(
        self,
        segments: List[Dict[str, Any]],
        output_path: str,
        *,
        background_music: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Concatenate video segments into a single output using project utility."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        await asyncio.to_thread(
            ffmpeg_concat_videos,
            segments,
            output_path,
            temp_dir=self.temp_dir,
            background_music=background_music,
        )
        return output_path

    async def render_with_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        video_filters: Optional[List[str]] = None,
        audio_filters: Optional[List[str]] = None,
        audio_input: Optional[str] = None,
        input_type: str = "image",  # "video" or "image"
        loop_input: bool = False,
        use_lavfi: bool = False,
        lavfi_spec: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        duration: float = 5.0,
    ) -> str:
        """Render media using FFmpeg with the given parameters."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # Build FFmpeg command
            ffmpeg_bin = getattr(settings, "ffmpeg_binary_path", "ffmpeg")
            threads = str(getattr(settings, "ffmpeg_threads", 1))
            vcodec = getattr(settings, "video_default_codec", "libx264")
            acodec = getattr(settings, "video_default_audio_codec", "aac")
            audio_bitrate = getattr(settings, "video_default_audio_bitrate", "192k")

            cmd: List[str] = [ffmpeg_bin, "-y", "-threads", threads]

            # Inputs and mapping
            if input_type == "video":
                # First audio (or anullsrc), then video input
                if audio_input and os.path.exists(audio_input):
                    cmd.extend(["-i", str(audio_input)])
                else:
                    cmd.extend(
                        [
                            "-f",
                            "lavfi",
                            "-i",
                            "anullsrc=channel_layout=stereo:sample_rate=44100",
                        ]
                    )
                cmd.extend(["-i", str(input_path)])
                map_args = ["-map", "1:v", "-map", "0:a"]
            else:
                # Image path (loop) or lavfi color, plus audio (or anullsrc)
                if use_lavfi:
                    spec = lavfi_spec or str(input_path)
                    cmd.extend(["-f", "lavfi", "-i", spec])
                else:
                    if loop_input:
                        cmd.extend(["-loop", "1"])
                    cmd.extend(["-i", str(input_path)])
                if audio_input and os.path.exists(audio_input):
                    cmd.extend(["-i", str(audio_input)])
                else:
                    cmd.extend(
                        [
                            "-f",
                            "lavfi",
                            "-i",
                            "anullsrc=channel_layout=stereo:sample_rate=44100",
                        ]
                    )
                map_args = ["-map", "0:v", "-map", "1:a"]

            # Add basic video options
            cmd.extend(["-r", str(fps), "-t", str(duration)])

            # If no explicit video filters provided, build sensible defaults from width/height
            # to fit the canvas while preserving aspect ratio. This keeps handler usable
            # without requiring the caller to always compute filters.
            if not video_filters and width and height:
                video_filters = [
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
                ]

            # Add video filters if any
            if video_filters:
                cmd.extend(["-vf", ",".join(video_filters)])

            # Add audio filters if any
            if audio_filters:
                cmd.extend(["-af", ",".join(audio_filters)])

            # Encoding and mapping
            cmd.extend(
                [
                    *map_args,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    vcodec,
                    "-c:a",
                    acodec,
                    "-b:a",
                    audio_bitrate,
                    str(output_path),
                ]
            )

            # Execute command
            logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
            await asyncio.to_thread(safe_subprocess_run, cmd, "Render with FFmpeg")

            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error in FFmpeg rendering: {e}", exc_info=True)
            raise
