from __future__ import annotations
from typing import Protocol, Dict, Any


class IAssetDownloader(Protocol):
    """Downloader for external media assets defined in JSON (images, videos, audio)."""

    async def download_asset(self, url: str, *, kind: str) -> str:
        """Download an asset to a local temp path and return that path.
        kind examples: "image", "video", "audio".
        """
        ...

    async def batch_download(self, items: list[Dict[str, Any]]) -> list[str]:
        """Download a batch of assets; return list of local paths matching order."""
        ...
