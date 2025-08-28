# Guide to Implementing Pipelines with Clean/Onion Architecture (updated)

This document explains how to structure: Steps → Pipeline → UseCase + Adapters (DI) → Endpoint.
Design follows OOP, SOLID, Clean/Onion: Endpoint in Delivery, UseCase and Steps in Application, Adapters in Infrastructure.

## Table of contents

- Introduction & Overview
- Project structure (common)
- Core primitives (PipelineContext, Step, BaseStep, Pipeline, PipelineFactory)
- PipelineContext usage conventions (Data Flow)
- Step-by-step guide: end-to-end (Step → Builder → Container → UseCase → DI → Router → Usage)
- Best practices
- Clean/Onion/SOLID compliance checklist
- Additional implementation notes

## Overview (Clean/Onion Architecture)

- **Delivery Layer** (`app/presentation/api/v1/routers/*`, `app/presentation/api/v1/dependencies/*`): receives HTTP requests, composes UseCases via DI.
- **Application Layer** (`app/application/`):
  - UseCase: orchestrates business logic, initializes Pipelines.
  - Pipeline + Steps: executes a sequence of processing steps.
  - Interfaces (Ports): defines contracts for Infrastructure.
- **Infrastructure Layer** (`app/infrastructure/adapters/*`): implements Ports, interacts with external services.

## Project structure (condensed)

```text
app/
  presentation/
    api/
      v1/
        routers/
          my_pipeline.py             # Example pipeline endpoint (optional)
        dependencies/
          my_pipeline.py             # DI: get_run_example_use_case

  application/
    pipeline/
      base.py                        # PipelineContext, BaseStep, Pipeline, Middleware
      factory.py                     # PipelineFactory (middlewares, fail_fast, legacy adapter)
      my_pipeline/                   # Example pipeline package (from docs)
        builder.py                   # Builder
        steps/                       # Example steps

    interfaces/                      # Ports (Application layer contracts)

    use_cases/
      run_example_pipeline.py        # RunExamplePipelineUseCase

  core/
    config.py                        # Settings

  infrastructure/
    adapters/                        # Concrete adapter implementations (Infrastructure layer)

utils/                               # Utilities (resource manager, etc.)
```

## Core primitives

Primitives in `app/application/pipeline/base.py` (simplified and consistent with current code):

- `PipelineContext`:
  - `input: Mapping[str, Any]`: immutable-like run payload (do not mutate).
  - `artifacts: Dict[str, Any]`: cross-step data.
  - Reserved key: `_run_id` with helpers `ensure_run_id()`, `get_run_id()`, `set_run_id()`.
  - Does not manage `temp_dir` inside Context (handled at Presentation/Infrastructure when needed).
- `Step` (Protocol): defines `async def __call__(context) -> None`. `BaseStep` provides lifecycle and requires `async def run(context)`.
- `BaseStep`: sync hooks (`on_start/on_finish/on_skip`), input validation (`required_keys`), retry/backoff/timeout and status.
- `Pipeline`: orchestrator to run steps sequentially, gather results, supports `fail_fast`.

### PipelineContext

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, ClassVar

@dataclass(slots=True)
class PipelineContext:
    """Common pipeline context shared across all pipelines."""

    RUN_ID_KEY: ClassVar[str] = "_run_id"

    input: Mapping[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Artifacts helpers
    def set(self, key: str, value: Any) -> None: self.artifacts[key] = value
    def get(self, key: str, default: Any = None) -> Any: return self.artifacts.get(key, default)
    def has(self, key: str) -> bool: return key in self.artifacts
    def remove(self, key: str) -> None: self.artifacts.pop(key, None)
    def update(self, **items: Any) -> None: self.artifacts.update(items)

    # Validation helpers
    def require(self, keys: list[str]) -> None:
        missing = [k for k in keys if k not in self.artifacts]
        if missing:
            raise KeyError(f"Missing required context keys: {', '.join(missing)}")

    # Run ID helpers
    def get_run_id(self) -> Optional[str]:
        return self.get(self.RUN_ID_KEY)
    def set_run_id(self, run_id: str) -> None:
        self.set(self.RUN_ID_KEY, run_id)
    def ensure_run_id(self, factory: Optional[callable[[], str]] = None) -> str:
        rid = self.get_run_id()
        if not rid and factory:
            rid = factory()
        if not rid:
            import uuid as _uuid
            rid = str(_uuid.uuid4())
        self.set_run_id(rid)
        return rid
```

### Step Protocol

```python
from typing import Protocol

class Step(Protocol):
    async def __call__(self, context: PipelineContext) -> None: ...
```

Lưu ý: `BaseStep` yêu cầu implement `async def run(context)` và cung cấp `__call__` lifecycle. `PipelineFactory` tự động bọc các legacy step chỉ có `run()` thành callable step.

### BaseStep

```python
import logging, asyncio, random
from abc import ABC, abstractmethod
from time import perf_counter
from enum import Enum

logger = logging.getLogger(__name__)

class StepStatus(str, Enum):
    PENDING = "pending"; RUNNING = "running"; COMPLETED = "completed"; SKIPPED = "skipped"; FAILED = "failed"

class BaseStep(ABC):
    name: str = "base_step"
    required_keys: list[str] = []
    retries: int = 0
    retry_backoff: float = 0.5
    timeout: float | None = None
    use_exponential_backoff: bool = True
    max_backoff: float = 5.0
    jitter: float = 0.1
    retry_exceptions: dict[type[Exception], dict[str, object]] = {}

    status: StepStatus = StepStatus.PENDING
    last_error: Exception | None = None
    duration: float = 0.0
    attempts: int = 0

    async def __call__(self, context: PipelineContext) -> None:
        attempts = 0
        self.last_error = None
        if not self.validate_inputs(context):
            missing = [k for k in self.required_keys if not context.has(k)]
            raise KeyError(f"Missing required inputs for step '{self.name}': {', '.join(missing)}")
        if self.can_skip(context):
            self.status = StepStatus.SKIPPED
            self.on_skip(context)
            return
        while True:
            attempts += 1; self.attempts = attempts; self.status = StepStatus.RUNNING
            self.on_start(context); start = perf_counter()
            try:
                if self.timeout:
                    await asyncio.wait_for(self.run(context), timeout=self.timeout)
                else:
                    await self.run(context)
                self.status = StepStatus.COMPLETED; return
            except Exception as e:
                self.last_error = e; self.status = StepStatus.FAILED
                # apply retry policy (global or per-exception)
                eff = dict(retries=self.retries, retry_backoff=self.retry_backoff, max_backoff=self.max_backoff,
                           jitter=self.jitter, use_exponential_backoff=self.use_exponential_backoff)
                for exc_t, cfg in self.retry_exceptions.items():
                    if isinstance(e, exc_t): eff.update(cfg); break
                if attempts <= int(eff["retries"]):
                    base = max(0.0, float(eff["retry_backoff"]))
                    sleep_s = base * (2 ** max(0, attempts - 1)) if eff["use_exponential_backoff"] else base
                    sleep_s = min(float(eff["max_backoff"]), float(sleep_s))
                    sleep_s += random.uniform(0.0, max(0.0, float(eff["jitter"])) )
                    await asyncio.sleep(sleep_s); continue
                raise
            finally:
                self.duration = perf_counter() - start
                self.on_finish(context, self.duration)

    @abstractmethod
    async def run(self, context: PipelineContext) -> None: ...

    # Hooks & utils
    def on_start(self, context: PipelineContext) -> None: logger.debug("Step %s start", self.name)
    def on_finish(self, context: PipelineContext, duration: float) -> None:
        logger.info("Step %s finished in %.3fs status=%s attempts=%d run_id=%s", self.name, duration, self.status.value, self.attempts, context.get_run_id())
    def on_skip(self, context: PipelineContext) -> None: logger.info("Step %s skipped", self.name)
    def validate_inputs(self, context: PipelineContext) -> bool: return all(context.has(k) for k in self.required_keys) if self.required_keys else True
    def can_skip(self, context: PipelineContext) -> bool: return False
```

### Pipeline

```python
from typing import List, TypedDict, Optional, Dict, Any

class StepResult(TypedDict):
    name: str; status: str; duration: float; error: Optional[str]; attempts: int

class PipelineResult(TypedDict):
    success: bool; duration: float; steps: List[StepResult]; error: Optional[str]; context: PipelineContext

class Pipeline:
    def __init__(self, steps: List[Step], *, fail_fast: bool = True):
        self._steps = steps; self.fail_fast = fail_fast

    async def execute(self, context: PipelineContext) -> PipelineResult:
        context.ensure_run_id()
        from time import perf_counter
        start = perf_counter(); results: Dict[str, Any] = {"success": False, "duration": 0.0, "steps": [], "error": None}
        for step in self._steps:
            info = {"name": getattr(step, "name", step.__class__.__name__), "status": "pending", "duration": 0.0, "error": None, "attempts": 0}
            results["steps"].append(info)
            s = perf_counter()
            try:
                await step(context)
                info["status"] = getattr(step, "status").value
                info["attempts"] = int(getattr(step, "attempts", 1) or 1)
            except Exception as e:
                info["status"] = "failed"; info["error"] = str(e)
                info["attempts"] = int(getattr(step, "attempts", 1) or 1)
                if self.fail_fast: raise
            finally:
                info["duration"] = perf_counter() - s
        results["duration"] = perf_counter() - start
        results["success"] = all(s["status"] == "completed" for s in results["steps"])
        results["context"] = context
        return results  # PipelineResult
```

### PipelineFactory

```python
from typing import List
from app.application.pipeline.base import Middleware
import asyncio

class PipelineFactory:
    def __init__(self, *, middlewares: List[Middleware] | None = None, fail_fast: bool = True):
        self._steps: List[Step] = []; self._middlewares = list(middlewares or []); self._fail_fast = fail_fast

    def add(self, step: Step) -> "PipelineFactory":
        wrapped = step
        # Tự động bọc legacy step chỉ có async run(context)
        if not hasattr(wrapped, "__call__") and hasattr(wrapped, "run") and asyncio.iscoroutinefunction(getattr(wrapped, "run")):
            impl = wrapped
            class _RunAdapter:
                def __init__(self, impl_obj): self.impl = impl_obj; self.name = getattr(impl_obj, "name", impl_obj.__class__.__name__)
                async def __call__(self, context): await self.impl.run(context)
            wrapped = _RunAdapter(impl)
        for mw in self._middlewares: wrapped = mw(wrapped)
        self._steps.append(wrapped); return self

    def extend(self, steps: List[Step]) -> "PipelineFactory":
        for s in steps: self.add(s); return self

    def build(self) -> Pipeline: return Pipeline(self._steps, fail_fast=self._fail_fast)
```

#### Logging middleware example (Observability)

```python
# File: app/application/pipeline/middlewares/logging.py
from typing import Callable, Awaitable
from app.application.pipeline.base import PipelineContext
import logging

StepCallable = Callable[[PipelineContext], Awaitable[None]]


def make_logging_middleware(logger: logging.Logger) -> Callable[[StepCallable], StepCallable]:
    """Wrap each step to log start/finish with run_id and duration if available."""
    def middleware(step: StepCallable) -> StepCallable:
        async def wrapped(context: PipelineContext) -> None:
            name = getattr(step, "name", getattr(step, "__name__", step.__class__.__name__))
            run_id = context.get_run_id() or "-"
            logger.info("[run=%s] step %s start", run_id, name)
            try:
                await step(context)
            finally:
                # If step has duration (BaseStep), include it
                duration = getattr(step, "duration", None)
                if duration is not None:
                    logger.info("[run=%s] step %s finish in %.3fs", run_id, name, duration)
                else:
                    logger.info("[run=%s] step %s finish", run_id, name)
        return wrapped  # type: ignore[return-value]

    return middleware


# Example usage in builder
import logging
from app.application.pipeline.factory import PipelineFactory
from app.application.pipeline.middlewares.logging import make_logging_middleware


def build_example_pipeline_via_container_with_logging(adapters: ExamplePipelineAdapters) -> Pipeline:
    logger = logging.getLogger("pipeline")
    factory = PipelineFactory(middlewares=[make_logging_middleware(logger)], fail_fast=True)
    factory.add(ExampleStep(adapters.example_processor))
    return factory.build()
```

## PipelineContext usage conventions (Data Flow)

- **Input (immutable-like)**: read from `context.input`. DO NOT write to input.
- **Artifacts (mutable)**: write/read outputs via helpers `set/get/has/update`.
  - Naming: `snake_case`, with optional namespacing (e.g., `domain.segment_items`, `audio.synthesized_path`).
  - Suggested flow: `validated_data` → `processed_items` → `output_chunks` → `final_output_path` → `final_output_url`.
- **Reserved keys**: only `_run_id` in Context. Manage `temp_dir` at Presentation/Infrastructure when needed.

## Step-by-step guide: end-to-end (Step → Builder → Container → UseCase → DI → Router → Usage)

```python
# File: app/application/pipeline/my_pipeline/steps/example_step.py
from app.application.pipeline.base import BaseStep, PipelineContext
from app.application.interfaces.example import IExampleProcessor


class ExampleStep(BaseStep):
    name = "example_step"

    def __init__(self, processor: IExampleProcessor) -> None:
        super().__init__()
        self._processor = processor

    async def run(self, context: PipelineContext) -> None:
        context.ensure_run_id()
        data = context.input
        if not isinstance(data, dict):
            raise ValueError("context.input must be a dict")
        result = await self._processor.process(data)
        context.set("example_result", result)
```

```python
# File: app/application/interfaces/example.py
from typing import Protocol


class IExampleProcessor(Protocol):
    async def process(self, data: dict) -> dict: ...
```

```python
# File: app/infrastructure/adapters/example_processor.py
from app.application.interfaces.example import IExampleProcessor


class ExampleProcessor(IExampleProcessor):
    async def process(self, data: dict) -> dict:
        return {"echo_size": len(data), "ok": True}
```

```python
# File: app/application/pipeline/my_pipeline/builder.py
from dataclasses import dataclass
from app.application.pipeline.factory import PipelineFactory
from app.application.pipeline.base import Pipeline
from app.application.pipeline.my_pipeline.steps.example_step import ExampleStep
from app.application.interfaces.example import IExampleProcessor


@dataclass
class ExamplePipelineAdapters:
    example_processor: IExampleProcessor


def build_example_pipeline_via_container(adapters: ExamplePipelineAdapters) -> Pipeline:
    factory = PipelineFactory()
    factory.add(ExampleStep(adapters.example_processor))
    return factory.build()
```

```python
# File: app/application/use_cases/run_example_pipeline.py
from app.application.pipeline.base import PipelineContext, Pipeline


class RunExamplePipelineUseCase:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, input_data: dict) -> dict:
        ctx = PipelineContext(input=input_data)
        result = await self._pipeline.execute(ctx)
        return result["context"]
```

```python
# File: app/presentation/api/v1/dependencies/my_pipeline.py
from app.infrastructure.adapters.example_processor import ExampleProcessor
from app.application.pipeline.my_pipeline.builder import (
    build_example_pipeline_via_container,
    ExamplePipelineAdapters,
)
from app.application.use_cases.run_example_pipeline import RunExamplePipelineUseCase


def get_run_example_use_case() -> RunExamplePipelineUseCase:
    adapters = ExamplePipelineAdapters(example_processor=ExampleProcessor())
    pipeline = build_example_pipeline_via_container(adapters)
    return RunExamplePipelineUseCase(pipeline)
```

```python
# File: app/presentation/api/v1/routers/my_pipeline.py
from fastapi import APIRouter, Depends
from app.application.use_cases.run_example_pipeline import RunExamplePipelineUseCase
from app.presentation.api.v1.dependencies.my_pipeline import get_run_example_use_case


router = APIRouter(prefix="/my-pipeline", tags=["my-pipeline"])


@router.post("/run")
async def run_my_pipeline(payload: dict, use_case: RunExamplePipelineUseCase = Depends(get_run_example_use_case)):
    ctx = await use_case.execute(payload)
    return {"run_id": ctx.get("_run_id"), "result": ctx.get("example_result")}
```

```python
# File: app/application/pipeline/my_pipeline/example_usage.py
from app.presentation.api.v1.dependencies.my_pipeline import get_run_example_use_case
import anyio


async def main():
    use_case = get_run_example_use_case()
    ctx = {"foo": "bar", "n": 3}
    result_ctx = await use_case.execute(ctx)
    print(result_ctx.get("example_result"))


if __name__ == "__main__":
    anyio.run(main)
```

## Best practices

- **SRP for Steps**: each step does one clear thing; name outputs (artifacts) clearly.
- **Light typing**: consider TypedDict/Pydantic for important artifacts (e.g., `ValidatedData`, `SegmentClip`).
- **Idempotency and retries**: leverage `BaseStep.retries`, `retry_exceptions`, `timeout` for I/O steps. Temp dir is managed outside `PipelineContext`.

---

## Additional Implementation Notes

- Call `PipelineContext.ensure_run_id()` at the start of a run (already ensured by `Pipeline.execute()`).
- Main artifact flow:
  - `validated_data` → `processed_items` → `output_chunks` → `final_output_path` → `final_output_url`.
- Temp dir/job-scoped resources: managed at Presentation/Infrastructure (see `utils/resource_manager.py`).
- Builder uses the Adapter Container pattern to manage dependencies in a scalable, type-safe way.

---

## Clean/Onion/SOLID Compliance Checklist

**Dependency Rule (Clean Architecture)**:

- Application layer (Steps, UseCase, Adapter Container) MUST NOT import from Infrastructure.
- Infrastructure adapters implement Application interfaces (Ports).
- Dependencies flow inward: Delivery → Application → Infrastructure.

**Adapter Container Pattern**:

- Container lives in the Application Layer and contains Application interfaces (Ports).
- Infrastructure populates the container with concrete implementations.
- Builders receive the container instead of many individual parameters (avoids parameter explosion).
- Scales for complex pipelines with many adapters.

**STRICT Pipeline Assembly**:

- Container validates required adapters at type-check/definition time (dataclass).
- DI Provider always initializes required adapters before building the pipeline.
- Steps receive adapters via constructor (Dependency Injection).
- Optional adapters are handled gracefully in the builder.

**Single Responsibility (SOLID)**:

- Each Step does one clear job with descriptive naming.
- UseCase orchestrates only; it does not hold detailed business logic.
- Adapters handle only I/O, not business rules.
- Container only groups adapters, no logic inside.

**Data Flow & Context Management**:

- The first step calls `ensure_run_id()`.
- Name artifacts clearly following the flow: input → processing → output.
- Do not overwrite artifacts casually; use namespaces when needed.

## Code review checklist (practical)

- [ ] Architecture: No Dependency Rule violations (Application must not import Infrastructure).
- [ ] Steps: Clear naming, one responsibility per step, validate `required_keys` appropriately.
- [ ] Retries/Timeouts: Configure `retries`, `retry_exceptions`, `timeout` for I/O-bound steps.
- [ ] Artifacts: Clear namespacing; avoid accidental overwrites; clean up if needed.
- [ ] PipelineFactory: Use necessary middlewares (logging, metrics, tracing…).
- [ ] UseCase: No detailed business logic; orchestrates the pipeline only.
- [ ] DI Provider: Initialize all required adapters; handle optional adapters in the builder.
- [ ] Observability: Log with `run_id`; enable `make_logging_middleware` if needed.
- [ ] Tests: Unit tests for steps; integration tests for pipeline happy-path and common failures.

## Frequently asked questions (FAQs)

- Q: Why is there no `temp_dir` in `PipelineContext`?
  - A: By design, `temp_dir` belongs to Presentation/Infrastructure; the Context only carries run data and artifacts.

- Q: My step only has `run(context)` and no `__call__`. Can I still use it?
  - A: Yes. `PipelineFactory` wraps legacy steps with `async run(context)` into callable steps.

- Q: Where should I put artifacts to avoid collisions between pipelines?
  - A: Use namespaced keys (e.g., `domain.output_chunks`) and avoid sharing artifacts across different pipelines.

- Q: How do I enable detailed logging per step?
  - A: Use `make_logging_middleware` and initialize `PipelineFactory(middlewares=[...])` in your builder.

- Q: When should I use `fail_fast=False`?
  - A: When you want to continue less-dependent steps and collect errors; remember to handle failed step status in the result.
