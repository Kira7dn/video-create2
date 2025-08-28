from __future__ import annotations
from typing import Protocol, Sequence, Any


class IVideoRenderer(Protocol):
    """Renders a final video from prepared clips, images, audio and transitions."""

    async def duration(self, input_path: str) -> float:
        """Return media duration in seconds for the given input path."""
        pass

    async def concat_clips(
        self, inputs: Sequence[str], *, output_path: str, transition: str | None = None
    ) -> str:
        """Concatenate input clips with optional transition, returning output path."""
        pass

    # Primitive operations
    async def transcode_video(
        self,
        input_path: str,
        *,
        output_path: str,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """Transcode/remux a source video to standard H.264/AAC MP4 at given geometry."""
        pass

    async def still_from_image(
        self,
        image_path: str,
        *,
        output_path: str,
        duration: float,
        width: int,
        height: int,
        fps: int,
        voice_path: str | None = None,
    ) -> str:
        """Create an MP4 clip from a still image, optionally with a voice-over audio track."""
        pass

    async def placeholder(
        self,
        *,
        output_path: str,
        duration: float,
        width: int,
        height: int,
        fps: int,
        color: str = "black",
    ) -> str:
        """Create a placeholder clip (e.g., black screen) when no media is available."""
        pass

    async def render_with_plan(
        self,
        plan: dict,
        *,
        output_path: str,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """Render based on an abstract plan that specifies inputs and filter chains.

        The plan is a renderer-agnostic dict produced by the Application layer.
        Implementations (Infra) translate it to concrete ffmpeg (or other) commands.
        """
        pass

    # Abstract media processor methods
    async def get_media_duration(self, source_path: str) -> float:
        """Return duration in seconds for any media source."""
        pass

    async def combine_media_sequence(
        self, sources: Sequence[str], *, target_path: str, blend_mode: str | None = None
    ) -> str:
        """Combine multiple media sources into a single output."""
        pass

    async def transform_media(
        self,
        source_path: str,
        *,
        target_path: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        """Transform media to target canvas specifications."""
        pass

    async def create_from_static(
        self,
        source_path: str,
        *,
        target_path: str,
        duration: float,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
        audio_source: str | None = None,
    ) -> str:
        """Create timed media from static source with optional audio."""
        pass

    async def create_placeholder(
        self,
        *,
        target_path: str,
        duration: float,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
        fill_color: str = "black",
    ) -> str:
        """Create placeholder media when no source is available."""
        pass

    async def process_with_specification(
        self,
        specification: dict[str, Any],
        seg_id: str,
        *,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        """Process media based on abstract specification.

        The specification is processor-agnostic and contains:
        - source information
        - transformation operations
        - timing and layout instructions
        """
        pass

    async def render_segment(
        self,
        segment: dict,
        *,
        output_path: str,
        temp_dir: str,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """Render a full-featured segment (image/video + transitions + audio + text overlays).

        Implementations may internally use ffmpeg or other tools. The segment dict follows the
        structure used across the pipeline (keys: image/video/voice_over/transition_in/out/text_over, ...).
        """
        pass
