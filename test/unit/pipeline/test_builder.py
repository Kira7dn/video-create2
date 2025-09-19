from __future__ import annotations

import logging
from types import SimpleNamespace
import pytest

from app.application.pipeline.base import BaseStep, PipelineContext, StepStatus
from app.application.pipeline.video import builder as builder_mod
from app.application.pipeline.video.builder import build_video_pipeline_via_container


class _DummyStep(BaseStep):
    def __init__(self, name: str):
        self.name = name

    async def run(self, context: PipelineContext) -> None:
        context.set(self.name, True)


class _FailStep(BaseStep):
    name = "fail_step"

    async def run(self, context: PipelineContext) -> None:
        raise RuntimeError("boom")


def _make_adapters_bundle():
    # Provide minimal placeholders; steps are monkeypatched to ignore these
    return SimpleNamespace(
        downloader=object(),
        renderer=object(),
        uploader=object(),
        keyword_agent=object(),
        image_search=object(),
        aligner=object(),
        image_processor=object(),
        splitter=object(),
        text_over_builder=object(),
    )


@pytest.fixture()
def patch_steps(monkeypatch):
    # Replace heavy steps with light dummies preserving names for order assertions
    monkeypatch.setattr(
        builder_mod, "ValidateInputStep", lambda: _DummyStep("validate_input")
    )
    monkeypatch.setattr(
        builder_mod, "DownloadAssetsStep", lambda *_: _DummyStep("download_assets")
    )
    monkeypatch.setattr(
        builder_mod, "ImageAutoStep", lambda *_: _DummyStep("image_auto")
    )
    monkeypatch.setattr(
        builder_mod,
        "TranscriptionAlignStep",
        lambda *_: _DummyStep("transcription_align"),
    )
    monkeypatch.setattr(
        builder_mod,
        "CreateSegmentClipsStep",
        lambda *_: _DummyStep("create_segment_clips"),
    )
    monkeypatch.setattr(
        builder_mod, "ConcatenateVideoStep", lambda *_: _DummyStep("concatenate_video")
    )
    monkeypatch.setattr(
        builder_mod, "UploadFinalStep", lambda *_: _DummyStep("upload_final")
    )


@pytest.mark.asyncio
async def test_builder_constructs_steps_in_expected_order_with_logging(
    patch_steps, caplog
):
    caplog.set_level(logging.DEBUG)

    pipeline = build_video_pipeline_via_container(
        _make_adapters_bundle(), enable_logging_middleware=True, fail_fast=True
    )
    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    # Order check via step names
    names = [s["name"] for s in result["steps"]]
    assert names == [
        "validate_input",
        "download_assets",
        "image_auto",
        "transcription_align",
        "create_segment_clips",
        "concatenate_video",
        "upload_final",
    ]

    # All completed
    assert all(s["status"] == StepStatus.COMPLETED.value for s in result["steps"])  # type: ignore[truthy-function]

    # Middleware emitted BEGIN/END
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "BEGIN" in logs and "END" in logs


@pytest.mark.asyncio
async def test_builder_fail_fast_false_continues_on_error(monkeypatch):
    # Patch only the first step to fail, the rest to succeed
    monkeypatch.setattr(builder_mod, "ValidateInputStep", lambda: _FailStep())
    monkeypatch.setattr(
        builder_mod, "DownloadAssetsStep", lambda *_: _DummyStep("download_assets")
    )
    monkeypatch.setattr(
        builder_mod, "ImageAutoStep", lambda *_: _DummyStep("image_auto")
    )
    monkeypatch.setattr(
        builder_mod,
        "TranscriptionAlignStep",
        lambda *_: _DummyStep("transcription_align"),
    )
    monkeypatch.setattr(
        builder_mod,
        "CreateSegmentClipsStep",
        lambda *_: _DummyStep("create_segment_clips"),
    )
    monkeypatch.setattr(
        builder_mod, "ConcatenateVideoStep", lambda *_: _DummyStep("concatenate_video")
    )
    monkeypatch.setattr(
        builder_mod, "UploadFinalStep", lambda *_: _DummyStep("upload_final")
    )

    pipeline = build_video_pipeline_via_container(
        _make_adapters_bundle(), enable_logging_middleware=False, fail_fast=False
    )
    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is False
    assert result["steps"][0]["status"] == StepStatus.FAILED.value
    # Later steps should still run
    assert all(s["status"] == StepStatus.COMPLETED.value for s in result["steps"][1:])


@pytest.mark.asyncio
async def test_builder_fail_fast_true_raises(monkeypatch):
    monkeypatch.setattr(builder_mod, "ValidateInputStep", lambda: _FailStep())
    # rest won't be reached
    pipeline = build_video_pipeline_via_container(
        _make_adapters_bundle(), enable_logging_middleware=False, fail_fast=True
    )

    ctx = PipelineContext(input={})
    with pytest.raises(RuntimeError):
        await pipeline.execute(ctx)
