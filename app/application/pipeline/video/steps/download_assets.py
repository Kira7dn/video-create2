from __future__ import annotations

import asyncio
import logging
from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces import IAssetDownloader
from app.core.config import settings

logger = logging.getLogger(__name__)


class DownloadAssetsStep(BaseStep):
    name = "download_assets"
    required_keys = ["validated_data"]

    def __init__(self, downloader: IAssetDownloader):
        self.downloader = downloader
        # Configure retry/timeout for network-bound operations
        self.retries = int(getattr(settings, "download_step_retries", 2))
        self.retry_backoff = float(
            getattr(settings, "download_step_retry_backoff", 0.5)
        )
        self.max_backoff = float(getattr(settings, "download_step_max_backoff", 3.0))
        self.jitter = float(getattr(settings, "download_step_jitter", 0.2))
        self.use_exponential_backoff = bool(
            getattr(settings, "download_step_use_exp_backoff", True)
        )
        self.timeout = getattr(settings, "download_step_timeout", None)

    async def run(self, context: PipelineContext) -> None:  # type: ignore[override]
        vd = context.get("validated_data")
        if vd is None:
            raise ValueError("validated_data not found")
        segments = vd.get("segments", [])
        background_music = vd.get("background_music") or context.input.get(
            "background_music"
        )
        if not isinstance(segments, list):
            raise ValueError("validated_data.segments must be a list")

        # Normalize asset kinds (remove prefix concept entirely)
        _asset_cfg = getattr(
            settings,
            "segment_asset_types",
            {"image": "img", "video": "vid", "voice_over": "voice"},
        )
        if isinstance(_asset_cfg, dict):
            asset_kinds = list(_asset_cfg.keys())
        elif isinstance(_asset_cfg, (list, tuple, set)):
            asset_kinds = list(_asset_cfg)
        else:
            asset_kinds = ["image", "video", "voice_over"]

        results = []
        coroutines: list = []
        meta: list[tuple[int, str]] = []  # (segment_index, asset_type); -1 for bg music

        # Concurrency limit for downloads
        max_concurrency = getattr(settings, "download_max_concurrent", 8)
        semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def _download_with_limit(url: str, kind: str, seg_id: str | None = None):
            async with semaphore:
                return await self.downloader.download_asset(
                    url, kind=kind, seg_id=seg_id
                )

        for i, segment in enumerate(segments):
            seg = dict(segment)
            seg_id = seg.get("id") or f"seg_{i}"
            # Queue downloads for assets present in this segment
            for asset_type in asset_kinds:
                asset = seg.get(asset_type)
                if isinstance(asset, dict) and asset.get("url"):
                    url = asset["url"]
                    coroutines.append(
                        _download_with_limit(url, kind=asset_type, seg_id=seg_id)
                    )
                    meta.append((i, asset_type))
            results.append(seg)

        # Queue background music download if available
        has_bg = isinstance(background_music, dict) and background_music.get("url")
        if has_bg:
            coroutines.append(
                _download_with_limit(
                    background_music["url"], kind="background_music", seg_id="bg"
                )
            )
            meta.append((-1, "background_music"))
        else:
            # Explicitly reflect absence of background music
            context.set("background_music", None)

        # Execute all downloads concurrently and handle failures
        if coroutines:
            results_list = await asyncio.gather(*coroutines, return_exceptions=True)
            successes = 0
            failures = 0
            for (seg_idx, asset_type), res in zip(meta, results_list):
                failed = isinstance(res, Exception) or not res
                if seg_idx == -1:  # background music
                    if failed:
                        logger.warning(
                            "Background music download failed; setting to None"
                        )
                        context.set("background_music", None)
                    else:
                        context.set("background_music", {**background_music, "local_path": res})  # type: ignore[arg-type]
                        successes += 1
                    continue

                # segment asset
                if failed:
                    # drop the asset field on failure to avoid downstream errors
                    if 0 <= seg_idx < len(results):
                        if asset_type in results[seg_idx]:
                            del results[seg_idx][asset_type]
                    logger.warning(
                        "Failed to download asset '%s' for segment index %s",
                        asset_type,
                        seg_idx,
                    )
                    failures += 1
                else:
                    asset = results[seg_idx].get(asset_type)
                    if isinstance(asset, dict):
                        results[seg_idx][asset_type] = {**asset, "local_path": res}
                    successes += 1

            logger.info(
                "Download summary: %d successes, %d failures", successes, failures
            )

        # Align with downstream expectation: write cleaned segments
        context.set("segments", results)
