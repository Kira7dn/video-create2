from __future__ import annotations

import asyncio
import logging
import pytest

from app.application.pipeline.base import (
    BaseStep,
    PipelineContext,
    StepStatus,
    make_logging_middleware,
)
from app.application.pipeline.factory import PipelineFactory


class _ReqKeysStep(BaseStep):
    name = "req_keys"
    required_keys = ["needed"]

    async def run(self, context: PipelineContext) -> None:  # pragma: no cover - not reached
        context.set("ok", True)


class _SkipStep(BaseStep):
    name = "skip_me"

    def can_skip(self, context: PipelineContext) -> bool:
        return True

    async def run(self, context: PipelineContext) -> None:  # pragma: no cover - skipped
        context.set("ran", True)


class _RetryThenSucceedStep(BaseStep):
    name = "retry_then_ok"

    def __init__(self, fail_times: int):
        self._remaining = fail_times
        # Allow retries equal to planned failures
        self.retries = fail_times
        self.retry_backoff = 0.0
        self.jitter = 0.0
        self.use_exponential_backoff = False

    async def run(self, context: PipelineContext) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("transient")
        context.set("done", True)


class _TimeoutStep(BaseStep):
    name = "timeout_step"

    def __init__(self, sleep_s: float, timeout: float):
        self._sleep = sleep_s
        self.timeout = timeout

    async def run(self, context: PipelineContext) -> None:
        await asyncio.sleep(self._sleep)


class _FailingStep(BaseStep):
    name = "always_fail"

    async def run(self, context: PipelineContext) -> None:
        raise ValueError("boom")


class _SimpleStep(BaseStep):
    name = "simple"

    async def run(self, context: PipelineContext) -> None:
        context.set("simple", True)


class _RetryByExceptionStep(BaseStep):
    name = "retry_by_exception"

    def __init__(self):
        self._failed = False
        self.retries = 0  # default retries
        # Allow retry only for KeyError once
        self.retry_exceptions = {
            KeyError: {
                "retries": 1,
                "retry_backoff": 0.0,
                "max_backoff": 0.0,
                "jitter": 0.0,
                "use_exponential_backoff": False,
            }
        }

    async def run(self, context: PipelineContext) -> None:
        if not self._failed:
            self._failed = True
            raise KeyError("transient-key-error")
        context.set("exception_retry_ok", True)


@pytest.mark.asyncio
async def test_required_keys_missing_raises():
    factory = PipelineFactory()
    factory.add(_ReqKeysStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    with pytest.raises(KeyError):
        await pipeline.execute(ctx)


@pytest.mark.asyncio
async def test_can_skip_sets_status_and_success():
    factory = PipelineFactory()
    factory.add(_SkipStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    # According to current design in Pipeline.execute(), success is True only
    # if all steps are COMPLETED. A skipped step keeps success=False.
    assert result["success"] is False
    assert result["steps"][0]["status"] == StepStatus.SKIPPED.value


@pytest.mark.asyncio
async def test_retries_then_succeeds_records_attempts():
    step = _RetryThenSucceedStep(fail_times=2)
    factory = PipelineFactory()
    factory.add(step)
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is True
    # attempts should be failures (2) + final success (1) = 3
    assert result["steps"][0]["attempts"] == 3
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_timeout_raises_and_fail_fast():
    factory = PipelineFactory()
    factory.add(_TimeoutStep(sleep_s=0.2, timeout=0.05))
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    with pytest.raises(asyncio.TimeoutError):
        await pipeline.execute(ctx)


@pytest.mark.asyncio
async def test_fail_fast_false_continues_on_error():
    factory = PipelineFactory(fail_fast=False)
    factory.add(_FailingStep()).add(_SimpleStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is False
    assert result["steps"][0]["status"] == StepStatus.FAILED.value
    assert result["steps"][1]["status"] == StepStatus.COMPLETED.value
    assert ctx.get("simple") is True


@pytest.mark.asyncio
async def test_logging_middleware_emits_begin_end(caplog):
    caplog.set_level(logging.DEBUG)
    factory = PipelineFactory(middlewares=[make_logging_middleware()])
    factory.add(_SimpleStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    await pipeline.execute(ctx)

    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "BEGIN" in logs
    assert "END" in logs


@pytest.mark.asyncio
async def test_retry_exceptions_policy_overrides_default():
    step = _RetryByExceptionStep()
    factory = PipelineFactory()
    factory.add(step)
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is True
    # one failure + one success
    assert result["steps"][0]["attempts"] == 2
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value
    assert ctx.get("exception_retry_ok") is True


@pytest.mark.asyncio
async def test_wrapped_step_preserves_name():
    factory = PipelineFactory(middlewares=[make_logging_middleware()])
    step = _SimpleStep()
    factory.add(step)
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["steps"][0]["name"] == step.name


class _RetryNotMatchingExceptionStep(BaseStep):
    name = "retry_not_match"

    def __init__(self):
        self.retries = 0
        self.retry_exceptions = {KeyError: {"retries": 3}}

    async def run(self, context: PipelineContext) -> None:
        raise ValueError("no-retry")


@pytest.mark.asyncio
async def test_retry_exceptions_not_matching_means_no_retry():
    factory = PipelineFactory(fail_fast=False)
    factory.add(_RetryNotMatchingExceptionStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is False
    assert result["steps"][0]["status"] == StepStatus.FAILED.value
    assert result["steps"][0]["attempts"] == 1


class _RequireThenSkipStep(BaseStep):
    name = "require_then_skip"
    required_keys = ["needed"]

    def can_skip(self, context: PipelineContext) -> bool:
        return True

    async def run(self, context: PipelineContext) -> None:  # pragma: no cover
        pass


@pytest.mark.asyncio
async def test_validate_happens_before_can_skip():
    factory = PipelineFactory()
    factory.add(_RequireThenSkipStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    with pytest.raises(KeyError):
        await pipeline.execute(ctx)


class _BoomStep(BaseStep):
    name = "boom"

    async def run(self, context: PipelineContext) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_fail_fast_true_propagates_exception():
    factory = PipelineFactory(fail_fast=True)
    factory.add(_BoomStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    with pytest.raises(RuntimeError):
        await pipeline.execute(ctx)


class _ReqKeysPositiveStep(BaseStep):
    name = "req_keys_ok"
    required_keys = ["needed"]

    async def run(self, context: PipelineContext) -> None:
        context.set("ran", True)


@pytest.mark.asyncio
async def test_required_keys_positive_path_success():
    factory = PipelineFactory()
    factory.add(_ReqKeysPositiveStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={})
    ctx.set("needed", 123)
    result = await pipeline.execute(ctx)

    assert result["success"] is True
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value
    assert ctx.get("ran") is True


@pytest.mark.asyncio
async def test_pipeline_result_structure_and_types():
    factory = PipelineFactory()
    factory.add(_SimpleStep()).add(_SimpleStep())
    pipeline = factory.build()

    ctx = PipelineContext(input={"x": 1})
    result = await pipeline.execute(ctx)

    # structure
    assert set(result.keys()) == {"success", "duration", "steps", "error", "context"}
    assert isinstance(result["duration"], float) and result["duration"] >= 0.0
    assert isinstance(result["steps"], list) and len(result["steps"]) == 2
    assert result["error"] is None
    assert isinstance(result["context"], PipelineContext)
    # step fields
    for s in result["steps"]:
        assert set(s.keys()) == {"name", "status", "duration", "error", "attempts"}
        assert isinstance(s["name"], str)
        assert s["status"] in {v.value for v in StepStatus}
        assert isinstance(s["duration"], float) and s["duration"] >= 0.0
        # simple steps completed, so error should be None
        assert s["error"] is None
        assert isinstance(s["attempts"], int) and s["attempts"] >= 1
