from __future__ import annotations
from typing import Protocol, Sequence


class IVideoRenderer(Protocol):
    """Renders a final video from prepared clips, images, audio and transitions."""

    async def duration(self, input_path: str) -> float:
        """Return media duration in seconds for the given input path."""
        pass

    async def concat_clips(
        self,
        inputs: Sequence[str],
        *,
        output_path: str,
        background_music: dict | None = None,
    ) -> str:
        """Concatenate input clips with optional transition and background music.

        background_music: optional dict with keys like 'local_path', 'start_delay', 'end_delay'.
        Implementations may ignore if None.
        """
        pass

    async def render_segment(
        self,
        segment: dict,
        *,
        seg_id: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        """Render a full-featured segment (image/video + transitions + audio + text overlays)."""
        pass
