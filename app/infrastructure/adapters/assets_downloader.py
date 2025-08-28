from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

from app.application.interfaces.asset_repo import IAssetDownloader
from utils.download_utils import download_file


class AssetRepoDownloader(IAssetDownloader):
    """Download assets using existing utils into a deterministic local folder.

    Files are saved under the pipeline temp folder (temp_dir) in the layout:
    {temp_dir}/{seg_id}/{kind}.{ext}
    """

    temp_dir: Path

    def __init__(self, temp_dir: str | Path = "data/assets") -> None:
        self.temp_dir = Path(temp_dir)
        # Do not create the temp_dir on init to avoid empty folders.
        # It will be created lazily in download_asset() as needed.

    async def download_asset(
        self, url: str, *, kind: str, seg_id: str | None = None
    ) -> str:
        """Download to {temp_dir}/{seg_id}/{kind}.{ext}."""
        clean_url = url.split("?")[0]
        ext = Path(clean_url).suffix or ""
        # Fallback extension if missing
        if not ext:
            # naive inference from kind
            ext = {
                "image": ".jpg",
                "video": ".mp4",
                "voice_over": ".wav",
                "background_music": ".mp3",
            }.get(kind, ".bin")
        target_dir = self.temp_dir / (seg_id or "common")
        dest = target_dir / f"{kind}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = await download_file(url, destination=str(dest), overwrite=True)
        return str(path)

    async def batch_download(self, items: List[Dict[str, Any]]) -> List[str]:
        results: List[str] = []
        for item in items:
            url = str(item.get("url", ""))
            kind = str(item.get("kind", "asset"))
            results.append(await self.download_asset(url, kind=kind))
        return results
