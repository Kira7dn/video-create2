from __future__ import annotations

from typing import List
import asyncio

from app.application.pipeline.base import Pipeline, Step, Middleware


class PipelineFactory:
    """Fluent builder for Pipelines, with optional middlewares per step.

    Example:
        factory = PipelineFactory()
        pipeline = factory.add(step1).add(step2).build()
    """

    def __init__(self, *, middlewares: List[Middleware] | None = None, fail_fast: bool = True):
        self._steps: List[Step] = []
        self._middlewares = list(middlewares or [])
        self._fail_fast = fail_fast

    def add(self, step: Step) -> "PipelineFactory":
        wrapped = step

        # Adapt legacy steps that implement `async run(context)` but not `__call__`
        if not hasattr(wrapped, "__call__") and hasattr(wrapped, "run"):
            run_attr = getattr(wrapped, "run")
            if asyncio.iscoroutinefunction(run_attr):
                impl = wrapped

                class _RunAdapter:
                    def __init__(self, impl_obj):
                        self.impl = impl_obj
                        self.name = getattr(impl_obj, "name", impl_obj.__class__.__name__)

                    async def __call__(self, context):
                        await self.impl.run(context)

                wrapped = _RunAdapter(impl)
        for mw in self._middlewares:
            wrapped = mw(wrapped)
        self._steps.append(wrapped)
        return self

    def extend(self, steps: List[Step]) -> "PipelineFactory":
        for s in steps:
            self.add(s)
        return self

    def build(self) -> Pipeline:
        return Pipeline(self._steps, fail_fast=self._fail_fast)
