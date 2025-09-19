import asyncio
from pathlib import Path

import pytest

from app.infrastructure.adapters.assets_downloader import AssetRepoDownloader
import app.infrastructure.adapters.assets_downloader as ad


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_asset_repo_downloader_saves_under_kind(monkeypatch, tmp_path):
    # Arrange
    calls = {}

    async def fake_download_file(url: str, destination: str, **kwargs):
        # Simulate a successful download by creating the destination file
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"data")
        calls["dest"] = destination
        return destination

    # Patch the symbol used inside adapter module
    monkeypatch.setattr(ad, "download_file", fake_download_file)

    # Official: use temp_dir
    adapter = AssetRepoDownloader(temp_dir=tmp_path / "assets")

    # Act
    out = await adapter.download_asset("http://example.com/foo.png", kind="image")

    # Assert: layout {temp_dir}/{seg_id or common}/{kind}.{ext}
    assert Path(out).exists()
    p = Path(out)
    assert p.name == "image.png"
    assert p.parents[0].name == "common"
    assert p.parents[1].name == "assets"  # tmp/assets/common/image.png
