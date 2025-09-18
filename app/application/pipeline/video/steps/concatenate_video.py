from __future__ import annotations

import os
import logging

from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces.renderer import IVideoRenderer
from app.core.exceptions import ProcessingError


logger = logging.getLogger(__name__)


class ConcatenateVideoStep(BaseStep):
    name = "concatenate_video"
    required_keys = ["segment_clips"]

    def __init__(self, renderer: IVideoRenderer):
        self.renderer = renderer

    async def run(self, context: PipelineContext) -> None:
        clips = context.get("segment_clips") or []
        if not isinstance(clips, list):
            raise ValueError("segment_clips must be a list")

        # Normalize and validate clip paths
        clip_paths = [
            clip_item.get("path") if isinstance(clip_item, dict) else clip_item
            for clip_item in clips
        ]
        clip_paths = [path for path in clip_paths if isinstance(path, str) and path]
        if not clip_paths:
            raise ProcessingError("No clip paths provided for concatenation")

        # Ensure all inputs exist
        missing = [p for p in clip_paths if not os.path.exists(p)]
        if missing:
            raise ProcessingError(f"Missing clip files for concatenation: {missing}")

        # Ensure we have a unique run_id even when step is invoked outside Pipeline
        video_id = context.ensure_run_id()
        filename = f"final_video_{video_id}.mp4"
        output_path = os.path.join("data", "output", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        background_music = context.get("background_music")

        logger.info("Concatenating %d clips -> %s", len(clip_paths), output_path)

        output_path = await self.renderer.concat_clips(
            clip_paths,
            output_path=output_path,
            background_music=background_music,
        )

        # Verify output exists
        if not os.path.exists(output_path):
            raise ProcessingError(
                f"Concatenation reported success but output not found: {output_path}"
            )
        try:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("✅ Concatenation done: %s (%.2f MB)", output_path, size_mb)
        except Exception:  # size is best-effort
            logger.info("✅ Concatenation done: %s", output_path)
        context.set("final_video_path", output_path)
        return
