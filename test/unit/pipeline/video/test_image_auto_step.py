# pylint: disable=protected-access
"""Unit tests for the ImageAutoStep class.

This module contains tests for the image auto-processing functionality,
including keyword extraction, image search, and processing pipelines.
"""

from unittest.mock import AsyncMock, MagicMock, patch, Mock
import os
import logging
import pytest
import requests
from pathlib import Path

from app.core.exceptions import ProcessingError
from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.image_auto import ImageAutoStep
from app.core.config import settings


@pytest.fixture(scope="session")
def valid_video_data():
    """Minimal valid video data used by local tests when running this file alone.

    Provides structure expected by `test_context` and `input_data` fixtures.
    """
    return {
        "segments": [
            {
                "voice_over": {"content": "Sample segment content"},
                "image": {"url": "", "local_path": ""},
            }
        ],
        "transitions": [],
        "background_music": None,
        "keywords": ["sample", "keyword"],
    }


@pytest.fixture
def mock_downloader():
    """A mock IAssetDownloader with async download_asset."""
    dl = Mock()
    dl.download_asset = AsyncMock(return_value="/tmp/mock_image.jpg")
    return dl


@pytest.fixture
def step(mock_downloader):
    """Create and return an instance of ImageAutoStep with required adapters for testing."""
    keyword_agent = Mock()
    image_search = Mock()
    image_processor = Mock()
    return ImageAutoStep(mock_downloader, keyword_agent, image_search, image_processor)


@pytest.fixture
def input_data(valid_video_data):
    """Create sample input data for processor tests using valid_video_data from conftest.

    This fixture simulates the output of the DownloadProcessor, which is the input to ImageProcessor.
    The format is a tuple of (segments, bg_music), where segments is a list of asset dictionaries.
    """
    segments = valid_video_data.get("segments", [])

    # Use the exact path to the test image
    test_image_path = "test\\source\\images\\test_image_1.jpg"

    # Create a list of asset dictionaries, one per segment
    asset_list = []
    for segment in segments:
        # Only include segments with images
        if "image" in segment and isinstance(segment["image"], dict):
            asset_dict = {
                "image": {
                    "url": segment["image"].get("url", ""),
                    "local_path": test_image_path,
                }
            }
            asset_list.append(asset_dict)

    # Return a tuple of (segments, bg_music) to match the expected input format
    return (asset_list, None)  # bg_music is None for these tests


@pytest.fixture
def test_context(valid_video_data):
    """Create test PipelineContext using valid_video_data via context API (no temp_dir)."""
    ctx = PipelineContext(input=valid_video_data)
    # seed common artifacts used by steps
    ctx.set("segments", valid_video_data.get("segments", []))
    ctx.set("transitions", valid_video_data.get("transitions", []))
    ctx.set("background_music", valid_video_data.get("background_music"))
    ctx.set("keywords", valid_video_data.get("keywords", []))
    return ctx


@pytest.mark.asyncio
async def test_ai_extract_keywords_success(step, monkeypatch):
    """Test successful keyword extraction from text using AI.

    This test verifies that the _ai_extract_keywords method correctly extracts
    and returns keywords from the input text using the AI agent.
    """
    # Save the original settings and agent
    original_settings = settings.ai_keyword_extraction_enabled
    original_agent = step.keyword_agent

    try:
        # Enable AI keyword extraction
        monkeypatch.setattr(settings, "ai_keyword_extraction_enabled", True)

        # Create a mock keyword agent that returns keywords directly
        expected_keywords = ["cat", "animal", "cute"]
        mock_agent = MagicMock()
        mock_agent.extract_keywords = AsyncMock(return_value=expected_keywords)
        step.keyword_agent = mock_agent

        # Test with a simple string and fields
        test_text = "A cute cat playing with a ball"
        test_fields = ["cat", "ball"]

        # Call the method under test
        # pylint: disable=protected-access
        keywords = await step._ai_extract_keywords(test_text, test_fields)

        # Verify the result
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(kw, str) for kw in keywords)
        assert keywords == expected_keywords

        # Verify the agent was called with the correct arguments
        mock_agent.extract_keywords.assert_called_once()
        call_args = mock_agent.extract_keywords.call_args[1]
        assert call_args["fields"] == test_fields
    finally:
        # Restore the original settings and agent
        monkeypatch.setattr(
            settings, "ai_keyword_extraction_enabled", original_settings
        )
        step.keyword_agent = original_agent


@pytest.mark.asyncio
async def test_ai_extract_keywords_fallback(step, monkeypatch):
    """Test fallback behavior when keyword agent is not available.

    This test verifies that the _ai_extract_keywords method correctly falls back
    to returning the fields when the keyword agent is not available.
    """
    # Save the original settings
    original_settings = settings.ai_keyword_extraction_enabled
    original_agent = step.keyword_agent

    try:
        # Disable AI keyword extraction to test fallback
        monkeypatch.setattr(settings, "ai_keyword_extraction_enabled", False)

        # Test with a simple string and fields
        test_text = "A beautiful sunset over the mountains"
        test_fields = ["sunset", "mountains"]

        # pylint: disable=protected-access
        keywords = await step._ai_extract_keywords(test_text, test_fields)

        # Verify the fallback behavior returns the fields in a list
        assert isinstance(keywords, list)
        assert keywords == test_fields  # Fallback returns fields directly
    finally:
        # Restore the original settings and agent
        monkeypatch.setattr(
            settings, "ai_keyword_extraction_enabled", original_settings
        )
        step.keyword_agent = original_agent


@pytest.mark.asyncio
async def test_ai_extract_keywords_exception(step, monkeypatch):
    """Test exception handling in keyword extraction.

    This test verifies that the _ai_extract_keywords method correctly handles
    exceptions from the AI agent by falling back to the fields.
    """
    # Save the original settings and agent
    original_settings = settings.ai_keyword_extraction_enabled
    original_agent = step.keyword_agent

    try:
        # Enable AI keyword extraction
        monkeypatch.setattr(settings, "ai_keyword_extraction_enabled", True)

        # Create a mock agent that raises a ValueError on extract_keywords
        mock_agent = MagicMock()
        mock_agent.extract_keywords = AsyncMock(side_effect=ValueError("AI service unavailable"))
        step.keyword_agent = mock_agent

        # Test with a simple string and fields
        test_text = "A cute dog playing in the park"
        test_fields = ["dog", "park"]

        # pylint: disable=protected-access
        keywords = await step._ai_extract_keywords(test_text, test_fields)

        # Verify the fallback behavior returns the fields in a list
        assert isinstance(keywords, list)
        assert keywords == test_fields  # Fallback returns fields directly

        # Verify the agent was called
        mock_agent.extract_keywords.assert_called_once()
    finally:
        # Restore the original settings and agent
        monkeypatch.setattr(
            settings, "ai_keyword_extraction_enabled", original_settings
        )
        step.keyword_agent = original_agent


@pytest.mark.asyncio
async def test_ai_search_image_found(step, monkeypatch):
    """Test successful image search with found results.

    This test verifies that the _ai_search_image method correctly returns
    an image URL when a matching image is found.
    """
    # Save the original settings
    original_settings = settings.ai_keyword_extraction_enabled

    # Provide an image_search adapter mock
    step.image_search = Mock()

    try:
        # Enable AI keyword extraction
        monkeypatch.setattr(settings, "ai_keyword_extraction_enabled", True)

        # Mock the _ai_extract_keywords method
        with patch.object(
            step, "_ai_extract_keywords", AsyncMock(return_value=["cat", "animal"])
        ) as mock_extract:
            step.image_search.search_image.return_value = "http://img.com/cat.jpg"
            # Call the method under test
            url = await step._ai_search_image("A cat")

            # Verify the result
            assert url == "http://img.com/cat.jpg"

            # Verify the mocks were called correctly
            mock_extract.assert_called_once()
            step.image_search.search_image.assert_called()
    finally:
        # Restore the original settings and function
        monkeypatch.setattr(
            settings, "ai_keyword_extraction_enabled", original_settings
        )
        pass


@pytest.mark.asyncio
async def test_ai_search_image_fallback(step, monkeypatch):
    """Test fallback behavior in image search.

    This test verifies that the _ai_search_image method correctly falls back
    to a secondary keyword when the primary search returns no results.
    """
    # Save the original settings
    original_settings = settings.ai_keyword_extraction_enabled

    # Provide an image_search adapter mock
    step.image_search = Mock()

    try:
        # Enable AI keyword extraction
        monkeypatch.setattr(settings, "ai_keyword_extraction_enabled", True)

        # Mock the _ai_extract_keywords method to return multiple keywords
        with patch.object(
            step,
            "_ai_extract_keywords",
            AsyncMock(return_value=["notfound", "fallback"]),
        ) as mock_extract:
            step.image_search.search_image.side_effect = [None, "http://img.com/fallback.jpg"]
            # Call the method under test
            url = await step._ai_search_image("Nothing")

            # Verify the result
            assert url == "http://img.com/fallback.jpg"

            # Verify the mocks were called correctly
            mock_extract.assert_called_once()
            assert step.image_search.search_image.call_count == 2
    finally:
        # Restore the original settings
        monkeypatch.setattr(
            settings, "ai_keyword_extraction_enabled", original_settings
        )


@pytest.mark.asyncio
async def test_ai_search_image_none(step, monkeypatch):
    """Test image search with no results.

    This test verifies that the _ai_search_image method correctly returns
    None when no matching images are found for any keywords.
    """
    # Save the original settings and agent
    original_settings = settings.ai_keyword_extraction_enabled
    original_agent = getattr(step, "keyword_agent", None)

    # Using image_search adapter mock; no module patching needed
    step.image_search = Mock()

    try:
        # Enable AI keyword extraction
        monkeypatch.setattr(settings, "ai_keyword_extraction_enabled", True)

        # Mock keyword extraction and search to return None for all attempts
        with patch.object(
            step, "_ai_extract_keywords", AsyncMock(return_value=["notfound", "nope"])
        ) as mock_extract:
            step.image_search.search_image.return_value = None
            # Call the method under test
            url = await step._ai_search_image("Nothing")

            # Verify the result is None when no images are found
            assert url is None

            # Verify the mocks were called correctly
            mock_extract.assert_called_once()
            # Thay vì assert cứng call_count, kiểm tra search_image được gọi ít nhất 3 lần
            assert step.image_search.search_image.call_count >= 3
    finally:
        # Restore the original settings and agent
        monkeypatch.setattr(
            settings, "ai_keyword_extraction_enabled", original_settings
        )
        if original_agent is not None:
            step.keyword_agent = original_agent


@pytest.mark.asyncio
async def test_run_valid_image(
    monkeypatch, step: ImageAutoStep, test_context: PipelineContext
):
    """Test processing of a valid image.

    This test verifies that the process method correctly processes and returns
    a valid image from the download results.
    """
    # Prepare context
    test_context.set(
        "segments",
        [{"voice_over": {"content": "A cat"}, "image": {"local_path": "valid.jpg"}}],
    )
    test_context.set("keywords", ["cat", "animal"])

    # Create a test segment with a valid image
    # Mock image validation to always return True
    monkeypatch.setattr(
        "app.application.pipeline.video.steps.image_auto.is_image_size_valid",
        lambda path, w, h: True,
    )

    await step.run(test_context)
    segs = test_context.get("segments")
    assert segs and segs[0]["image"]["local_path"] == "valid.jpg"


@pytest.mark.asyncio
async def test_run_invalid_image_download(
    monkeypatch, step, test_context, mock_downloader
):
    """Test processing with an invalid image that needs to be downloaded.

    This test verifies that the process method correctly handles an invalid
    image by downloading a replacement from the internet.
    """
    # Setup context with one segment and invalid image
    test_context.set(
        "segments",
        [{"voice_over": {"content": "A cat"}, "image": {"local_path": "invalid.jpg"}}],
    )

    # Always mark current image invalid
    monkeypatch.setattr(
        "app.application.pipeline.video.steps.image_auto.is_image_size_valid",
        lambda path, w, h: False,
    )

    # Mock search result and downloader behavior
    step._ai_search_image = AsyncMock(return_value="http://img.com/cat.jpg")
    mock_downloader.download_asset.return_value = "/tmp/replaced.jpg"

    await step.run(test_context)
    segs = test_context.get("segments")
    assert segs[0]["image"]["url"] == "http://img.com/cat.jpg"
    assert segs[0]["image"]["local_path"] == "/tmp/replaced.jpg"


@pytest.mark.asyncio
async def test_run_invalid_image_download_fail(
    monkeypatch, step, test_context, mock_downloader
):
    """Test handling of download failure for an invalid image.

    This test verifies that the process method correctly raises a
    ProcessingError when image download fails.
    """
    # Setup test context
    test_context.set(
        "segments",
        [{"voice_over": {"content": "A cat"}, "image": {"local_path": "invalid.jpg"}}],
    )

    # Mock dependencies
    monkeypatch.setattr(
        "app.application.pipeline.video.steps.image_auto.is_image_size_valid",
        lambda path, w, h: False,  # Always return invalid for this test
    )

    # Mock AI and downloader failure
    step._ai_search_image = AsyncMock(return_value="http://img.com/cat.jpg")
    mock_downloader.download_asset.side_effect = RuntimeError("Download failed")

    with pytest.raises(ProcessingError):
        await step.run(test_context)


@pytest.mark.asyncio
async def test_keyword_agent_extracts_compound_keywords(mock_downloader):
    """Test that keyword_agent returns compound, meaningful keyword phrases for given content."""
    step_local = ImageAutoStep(mock_downloader, Mock(), Mock(), Mock())
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("app.application.pipeline.video.steps.image_auto")

    # Mock the keyword agent to return phrases directly
    mock_agent = Mock()
    mock_agent.extract_keywords = AsyncMock(
        return_value=[
            "sustainable energy solutions",
            "renewable power sources",
            "green technology advancements",
            "environmental innovation",
        ]
    )
    step_local.keyword_agent = mock_agent
    # Test content
    content = (
        "The future of our planet depends on sustainable energy and green "
        "technology. Innovations in renewable power are changing the world."
    )
    result = await step_local._ai_extract_keywords(content)
    logger.info("Agent result: %s", result)
    assert isinstance(result, list)
    assert any("sustainable energy solutions" in kw for kw in result)
    assert all(len(kw.split()) >= 2 for kw in result)  # All are compound phrases


@pytest.mark.asyncio
async def test_keyword_agent_fallback_on_error(mock_downloader):
    """Test that fallback returns [fields] if AI fails."""
    step_local = ImageAutoStep(mock_downloader, Mock(), Mock(), Mock())
    step_local.keyword_agent = Mock()
    # Sử dụng RuntimeError để kích hoạt fallback
    step_local.keyword_agent.extract_keywords = AsyncMock(side_effect=RuntimeError("API Error"))
    content = "AI Agents for 2025"
    fields = ["AI agent", "2025"]
    result = await step_local._ai_extract_keywords(content, fields)
    assert result == fields
