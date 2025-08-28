from typing import Dict
from app.application.interfaces.video_adapters import IVideoPipelineAdapters
from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.builder import build_video_pipeline_via_container


class CreateVideoUseCase:
    """Use the Application-layer pipeline; fallback to service if needed.

    This preserves current functionality while we progressively wire
    Infrastructure adapters into the Application pipeline.
    """

    def __init__(self, adapters: IVideoPipelineAdapters) -> None:
        self._adapters = adapters

    async def execute(self, json_data: Dict) -> Dict:
        # Build and run the Application-layer pipeline using the adapters container
        ctx = PipelineContext(
            input={
                "json_data": json_data,
                "segments": json_data.get("segments", []),
                "transitions": json_data.get("transitions", []),
                "background_music": json_data.get("background_music"),
                "keywords": json_data.get("keywords", []),
            }
        )

        pipeline = build_video_pipeline_via_container(self._adapters)
        result = await pipeline.execute(ctx)
        ctx = result["context"]

        final_video_path = ctx.get("final_video_path")
        s3_url = ctx.get("final_video_url")

        return {"video_path": final_video_path, "s3_url": s3_url}
