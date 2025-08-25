from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

from app.application.interfaces.asset_repo import IAssetDownloader
from utils.download_utils import download_file


class AssetRepoDownloader(IAssetDownloader):
    """Download assets using existing utils into a deterministic local folder.

    Note: No execution context is provided via interface, so we use a common
    base directory under data/assets/{kind}/basename.
    """

    base_dir: Path

    def __init__(self, base_dir: str | Path = "data/assets") -> None:
        self.base_dir = Path(base_dir)
        # Do not create the base_dir on init to avoid empty folders.
        # It will be created lazily in download_asset() as needed.

    async def download_asset(self, url: str, *, kind: str) -> str:
        # Normalize destination under base_dir/kind/filename
        clean_url = url.split("?")[0]
        dest = self.base_dir / kind / Path(clean_url).name
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
