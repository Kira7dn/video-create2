from __future__ import annotations
import asyncio
import os
import logging
from pathlib import Path
from typing import Sequence

# Application imports
from app.application.interfaces.renderer import IVideoRenderer
from app.infrastructure.adapters.renderer.mixins.text_processing import (
    TextProcessingMixin,
)
from app.infrastructure.adapters.renderer.mixins.effects import EffectsMixin
from app.infrastructure.adapters.renderer.handlers.ffmpeg_handler import (
    FFmpegHandler,
)
from app.infrastructure.adapters.renderer.utils.subtitles_ass import build_ass_script
from app.infrastructure.adapters.renderer.utils.plan_translator import (
    translate_specification_to_plan,
)

logger = logging.getLogger(__name__)


class FFMpegVideoRenderer(IVideoRenderer, TextProcessingMixin, EffectsMixin):
    """
    FFmpeg-based video renderer implementation.

    This class handles video rendering tasks using FFmpeg, including concatenating clips
    and rendering segments with text and effects.

    Attributes:
        _temp_dir (str | None): Temporary directory for working files.
        _ffmpeg_handler (FFmpegHandler): Handler for FFmpeg operations.

    Public entry points:
        - concat_clips(...): Concatenate multiple video clips (optionally with background music).
        - render_segment(...): Render a single segment (image/video + effects/text) based on specification.

    Internal helpers:
        - _render_with_plan(...): Translate a prepared plan into ffmpeg filters and invoke handler.

    Examples:
        renderer = FFMpegVideoRenderer(temp_dir='/tmp')
        await renderer.concat_clips(['clip1.mp4', 'clip2.mp4'], output_path='output.mp4')
    """

    def __init__(self, *, temp_dir: str | None = None) -> None:
        self._temp_dir = temp_dir
        self._ffmpeg_handler = FFmpegHandler(temp_dir=temp_dir)

    async def duration(self, input_path: str) -> float:
        """Probe media duration using ffprobe."""
        return await asyncio.to_thread(self._ffmpeg_handler.probe_duration, input_path)

    async def concat_clips(
        self,
        inputs: Sequence[str],
        *,
        output_path: str,
        background_music: dict | None = None,
    ) -> str:
        """Concatenate multiple video clips into one with optional background music.

        Args:
            inputs: List of paths to video clips to concatenate.
            output_path: Path for the output concatenated video.
            background_music: Optional dict with background music details.

        Returns:
            Path to the concatenated video file.

        Raises:
            ValueError: If inputs are empty or output_path is invalid.
            RuntimeError: If FFmpeg concatenation fails.
        """
        if not output_path.endswith((".mp4", ".mov", ".avi")):
            output_path = f"{output_path}.mp4"

        # Build segment dicts for ffmpeg_concat_videos
        segments = [
            {"id": f"seg_{i}", "path": str(path)} for i, path in enumerate(inputs)
        ]

        # Create output directory if not exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Delegate to handler which wraps project utility
        return await self._ffmpeg_handler.concat_videos(
            segments,
            output_path,
            background_music=background_music,
        )

    async def render_segment(
        self,
        segment: dict,
        *,
        seg_id: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        """Render a video segment based on specification.

        This method accepts the application-level segment specification and is responsible for:
        - Creating a working directory under `self._temp_dir` using `seg_id`.
        - Resolving the final `output_path` (absolute).
        - Translating the spec to a renderer plan.
        - Executing the internal plan renderer.
        - Returning the output path.

        Args:
            segment: Dict containing segment specification.
            seg_id: Unique identifier for the segment.
            canvas_width: Width of the output canvas.
            canvas_height: Height of the output canvas.
            frame_rate: Frame rate of the output video.

        Returns:
            Absolute path to the rendered video segment.

        Raises:
            ValueError: If segment spec is invalid or dimensions are invalid.
            RuntimeError: If rendering fails.
        """
        # Validation
        if not segment:
            raise ValueError("Segment specification cannot be empty")
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        if not seg_id:
            raise ValueError("Segment ID cannot be empty")

        try:
            # Resolve work_dir and output_path using Path
            temp_dir_path = Path(self._temp_dir or ".")
            work_dir = temp_dir_path / seg_id
            work_dir.mkdir(parents=True, exist_ok=True)
            output_path = (work_dir / "segment_video.mp4").resolve()

            # Translate segment to render plan
            plan = translate_specification_to_plan(
                segment, canvas_width=canvas_width, canvas_height=canvas_height
            )

            # Render using the plan via the internal implementation
            await self._render_with_plan(
                plan=plan,
                output_path=str(output_path),
                width=canvas_width,
                height=canvas_height,
                fps=frame_rate,
            )

            return str(output_path)

        except Exception as e:
            logger.error(f"Error rendering segment: {str(e)}")
            raise RuntimeError(f"Failed to render segment {seg_id}: {e}") from e

    async def _render_with_plan(
        self, plan: dict, output_path: str, width: int, height: int, fps: int
    ) -> str:
        """Internal implementation to render video using a prepared plan.

        Translates the plan into FFmpeg filters and executes rendering.

        Args:
            plan: Dict containing the render plan with inputs and operations.
            output_path: Path for the output video file.
            width: Width of the output video.
            height: Height of the output video.
            fps: Frame rate of the output video.

        Returns:
            Path to the rendered video file.

        Raises:
            ValueError: If no input source is specified in the plan or plan is invalid.
            RuntimeError: If FFmpeg rendering fails.
        """
        # Validation
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a dict")
        ops = plan.get("ops", [])
        if not isinstance(ops, list):
            raise ValueError("Ops must be a list")

        try:
            logger.info(f"Rendering with plan: {plan}")

            # Get main input
            input_path = plan.get("video_input") or plan.get("image_input")
            if not input_path:
                raise ValueError("No input source specified in render plan")

            # Process text operations
            ass_events = self._process_text_operations(
                [op for op in ops if op.get("op") == "draw_text"],
                width,
                height,
            )

            # Process effects with priority ordering
            ordered_ops = sorted(ops, key=lambda op: op.get("priority", 0))
            video_filters, audio_filters = self._process_effects(ordered_ops)
            # Stabilize audio PTS to start at 0 and allow resampling jitter correction
            if audio_filters is None:
                audio_filters = []
            audio_filters.append("aresample=async=1:first_pts=0")

            # Add ASS subtitles if there are text events
            ass_path = None
            if ass_events:
                ass_path = Path(f"{output_path}.ass")
                # build_ass_script writes the ASS file and returns the path
                build_ass_script(ass_events, width, height, ass_path)
                video_filters.append(f"ass={ass_path}")

            # Render with FFmpeg
            result = await self._ffmpeg_handler.render_with_ffmpeg(
                input_path=input_path,
                output_path=output_path,
                video_filters=video_filters,
                audio_filters=audio_filters,
                audio_input=plan.get("audio_input"),
                input_type=str(plan.get("input_type", "image")),
                loop_input=plan.get("loop_image", False)
                or plan.get("input_type", "image") == "image",
                width=width,
                height=height,
                fps=fps,
                duration=float(plan.get("duration", 5.0)),
            )

            return result

        except Exception as e:
            logger.error(f"Error in _render_with_plan: {e}", exc_info=True)
            raise
