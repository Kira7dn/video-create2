"""
Unit tests for the DownloadAssetsStep (pipeline step).

Covers behavior of concurrent asset downloads, error handling, and
context updates for segments and background_music.
"""

from unittest.mock import AsyncMock
from pathlib import Path

import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.download_assets import DownloadAssetsStep
from app.infrastructure.adapters.assets_downloader import AssetRepoDownloader
import app.infrastructure.adapters.assets_downloader as ad


@pytest.fixture
def fake_downloader():
    """Async downloader mock exposing download_asset(url, kind, seg_id=None)."""

    class _DL:
        async def download_asset(self, url: str, kind: str, seg_id: str | None = None):
            # Simulate returning a local path based on url
            return str(Path("/tmp") / f"{Path(url).name}")

    dl = _DL()
    dl.download_asset = AsyncMock(side_effect=dl.download_asset)  # type: ignore[attr-defined]
    return dl


@pytest.fixture
def ctx() -> PipelineContext:
    return PipelineContext(input={})


class TestDownloadAssetsStep:
    """Test cases for DownloadAssetsStep.run()"""

    @pytest.mark.asyncio
    async def test_download_single_asset_success(self, fake_downloader, ctx):
        step = DownloadAssetsStep(fake_downloader)
        # Arrange validated data
        vd = {
            "segments": [
                {"id": "s1", "image": {"url": "https://cdn/cat.jpg"}},
            ],
            "background_music": {"url": "https://cdn/bg.mp3"},
        }
        ctx.set("validated_data", vd)

        # Act
        await step.run(ctx)

        # Assert segments and local paths
        segs = ctx.get("segments")
        assert isinstance(segs, list) and len(segs) == 1
        assert segs[0]["image"]["local_path"].endswith("cat.jpg")

        # Assert background music set
        bg = ctx.get("background_music")
        assert isinstance(bg, dict) and bg.get("local_path")
        assert bg["local_path"].endswith("bg.mp3")

        # Downloader interactions: 1 image + 1 bg
        assert fake_downloader.download_asset.await_count == 2

    @pytest.mark.asyncio
    async def test_download_multiple_assets_and_kinds(self, fake_downloader, ctx):
        step = DownloadAssetsStep(fake_downloader)
        vd = {
            "segments": [
                {
                    "id": "s1",
                    "image": {"url": "https://cdn/img1.jpg"},
                    "video": {"url": "https://cdn/vid1.mp4"},
                    "voice_over": {"url": "https://cdn/vo1.mp3"},
                },
                {
                    "id": "s2",
                    "image": {"url": "https://cdn/img2.jpg"},
                },
            ],
            "background_music": {"url": "https://cdn/bg.mp3"},
        }
        ctx.set("validated_data", vd)

        await step.run(ctx)

        segs = ctx.get("segments")
        assert isinstance(segs, list) and len(segs) == 2
        # Verify local paths populated
        assert segs[0]["image"]["local_path"].endswith("img1.jpg")
        assert segs[0]["video"]["local_path"].endswith("vid1.mp4")
        assert segs[0]["voice_over"]["local_path"].endswith("vo1.mp3")
        assert segs[1]["image"]["local_path"].endswith("img2.jpg")

        # calls: 4 assets + 1 background
        assert fake_downloader.download_asset.await_count == 5

    @pytest.mark.asyncio
    async def test_error_handling_removes_failed_assets_and_bg(
        self, fake_downloader, ctx
    ):
        async def _side_effect(url: str, kind: str, seg_id: str | None = None):
            if "error" in url:
                raise RuntimeError("boom")
            return f"/tmp/{Path(url).name}"

        fake_downloader.download_asset.side_effect = _side_effect  # type: ignore[attr-defined]
        step = DownloadAssetsStep(fake_downloader)

        vd = {
            "segments": [
                {"id": "ok", "image": {"url": "https://cdn/ok.jpg"}},
                {"id": "bad", "image": {"url": "https://cdn/error.jpg"}},
            ],
            "background_music": {"url": "https://cdn/error-bg.mp3"},
        }
        ctx.set("validated_data", vd)

        await step.run(ctx)

        segs = ctx.get("segments")
        assert segs[0]["image"]["local_path"].endswith("ok.jpg")
        # failed asset should be removed
        assert "image" not in segs[1]
        # bg music failed => None
        assert ctx.get("background_music") is None

    @pytest.mark.asyncio
    async def test_handles_missing_optional_assets(self, fake_downloader, ctx):
        step = DownloadAssetsStep(fake_downloader)
        vd = {"segments": [{"id": "s1", "text": "Hello"}]}
        ctx.set("validated_data", vd)

        await step.run(ctx)

        segs = ctx.get("segments")
        assert isinstance(segs, list) and len(segs) == 1
        # no assets added
        assert "image" not in segs[0]
        assert "video" not in segs[0]
        assert "voice_over" not in segs[0]

    @pytest.mark.asyncio
    async def test_background_music_only(self, fake_downloader, ctx):
        step = DownloadAssetsStep(fake_downloader)
        vd = {
            "segments": [{"id": "s1"}],
            "background_music": {"url": "https://cdn/bg.mp3"},
        }
        ctx.set("validated_data", vd)

        await step.run(ctx)

        bg = ctx.get("background_music")
        assert isinstance(bg, dict) and bg.get("local_path")
        assert bg["local_path"].endswith("bg.mp3")
        assert fake_downloader.download_asset.await_count == 1

    # Note: The step delegates path management to downloader; no temp_dir assumptions.

    # Tests related to custom temp_dir are not applicable; downloader owns path logic.

    @pytest.mark.asyncio
    async def test_duplicate_urls_are_downloaded_per_segment(
        self, fake_downloader, ctx
    ):
        step = DownloadAssetsStep(fake_downloader)
        duplicate_url = "https://cdn/dup.jpg"
        vd = {
            "segments": [
                {"id": "1", "image": {"url": duplicate_url}},
                {"id": "2", "image": {"url": duplicate_url}},
            ]
        }
        ctx.set("validated_data", vd)

        await step.run(ctx)

        segs = ctx.get("segments")
        assert segs[0]["image"]["local_path"].endswith("dup.jpg")
        assert segs[1]["image"]["local_path"].endswith("dup.jpg")
        # 2 calls for images
        assert fake_downloader.download_asset.await_count == 2


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

    adapter = AssetRepoDownloader(temp_dir=tmp_path / "assets")

    # Act
    out = await adapter.download_asset("http://example.com/foo.png", kind="image")

    # Assert layout {temp_dir}/{seg_id or 'common'}/{kind}.{ext}
    assert out.endswith("image.png")
    assert Path(out).exists()
    assert Path(out).parents[1].name == "assets"  # tmp/assets/common/image.png
    assert Path(out).parents[0].name == "common"
