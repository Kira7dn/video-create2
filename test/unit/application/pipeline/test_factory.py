import pytest
import asyncio

from app.application.pipeline.factory import PipelineFactory
from app.application.pipeline.base import PipelineContext, make_logging_middleware, StepStatus


class DummyStep:
    def __init__(self, name="dummy", record=None):
        self.name = name
        self.record = record if record is not None else []
        self.called = 0

    async def __call__(self, context: PipelineContext):
        self.called += 1
        self.record.append(self.name)
        # optionally set some context
        context.set(f"ran_{self.name}", True)


class FailingStep:
    def __init__(self, name="fail_once", exc=RuntimeError("boom")):
        self.name = name
        self.exc = exc
        self.called = 0

    async def __call__(self, context: PipelineContext):
        self.called += 1
        raise self.exc


@pytest.mark.asyncio
async def test_pipeline_factory_build_and_run_with_middleware(monkeypatch):
    # Prepare steps and middleware
    order = []
    s1 = DummyStep("s1", order)
    s2 = DummyStep("s2", order)

    # Use provided logging middleware to wrap steps
    logging_mw = make_logging_middleware()

    factory = PipelineFactory(middlewares=[logging_mw], fail_fast=True)
    pipeline = factory.add(s1).add(s2).build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is True
    assert order == ["s1", "s2"]
    assert ctx.get("ran_s1") is True
    assert ctx.get("ran_s2") is True
    # Ensure attempts recorded on steps (middleware must not break attributes)
    assert isinstance(result["steps"], list) and len(result["steps"]) == 2
    assert result["steps"][0]["name"] == "s1"
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_pipeline_factory_fail_fast_true_raises(monkeypatch):
    s1 = DummyStep("ok")
    s2 = FailingStep("bad")
    s3 = DummyStep("never")

    factory = PipelineFactory(fail_fast=True)
    pipeline = factory.extend([s1, s2, s3]).build()

    ctx = PipelineContext(input={})
    with pytest.raises(RuntimeError):
        await pipeline.execute(ctx)
    # s3 should not run due to fail_fast
    assert s1.called == 1
    assert s2.called == 1
    assert s3.called == 0


@pytest.mark.asyncio
async def test_pipeline_factory_fail_fast_false_continues(monkeypatch):
    s1 = DummyStep("ok1")
    s2 = FailingStep("bad")
    s3 = DummyStep("ok2")

    factory = PipelineFactory(fail_fast=False)
    pipeline = factory.extend([s1, s2, s3]).build()

    ctx = PipelineContext(input={})
    result = await pipeline.execute(ctx)

    assert result["success"] is False
    assert [step["name"] for step in result["steps"]] == ["ok1", "bad", "ok2"]
    statuses = [s["status"] for s in result["steps"]]
    assert statuses == [StepStatus.COMPLETED.value, StepStatus.FAILED.value, StepStatus.COMPLETED.value]
    # All were attempted (no raise)
    assert s1.called == 1 and s2.called == 1 and s3.called == 1
