from __future__ import annotations

import os
import logging

from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces import IUploader
from app.core.exceptions import UploadError


logger = logging.getLogger(__name__)


class UploadFinalStep(BaseStep):
    name = "upload_final"
    required_keys = ["final_video_path"]

    def __init__(self, uploader: IUploader):
        self.uploader = uploader

    async def run(self, context: PipelineContext) -> None:  # type: ignore[override]
        final_path = context.get("final_video_path")
        if not final_path:
            logger.info("No final video to upload; skipping UploadFinalStep")
            return

        if not isinstance(final_path, str) or not os.path.exists(final_path):
            vid = context.get_run_id()
            raise UploadError(
                f"Final video file not found: {final_path}",
                video_id=str(vid) if vid else None,
            )

        # Provide a sensible destination path hint if supported by the uploader
        run_id = context.get_run_id() or "video"
        dest_path = f"videos/{run_id}.mp4"

        try:
            logger.info("Uploading final video: %s -> %s", final_path, dest_path)
            # Prefer rich signature if supported
            try:
                url = await self.uploader.upload_file(
                    final_path,
                    public=True,
                    dest_path=dest_path,
                    content_type="video/mp4",
                )
            except TypeError:
                # Fallback for simple uploader fakes that don't accept extra kwargs
                url = await self.uploader.upload_file(final_path, public=True)
        except Exception as e:  # noqa: BLE001
            vid = context.get_run_id()
            raise UploadError(
                f"Failed to upload final video: {e}", video_id=str(vid) if vid else None
            ) from e

        context.set("s3_upload_result", {"url": url})
        context.set("final_video_url", url)
        logger.info("Final video uploaded: %s", url)
        return
