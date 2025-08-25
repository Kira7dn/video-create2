from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Mapping,
    Callable,
    ClassVar,
)
import logging
from abc import ABC, abstractmethod
from time import perf_counter
from enum import Enum
import asyncio
import random


@dataclass(slots=True)
class PipelineContext:
    """Common pipeline context shared across all pipelines.

    - input: immutable-like request/run input payload
    - artifacts: cross-step working data and outputs (also stores run-scoped
      technical info like run_id/temp_dir via reserved keys)
    """

    # Reserved artifact keys (not dataclass fields)
    RUN_ID_KEY: ClassVar[str] = "_run_id"

    input: Mapping[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # ----- Artifacts: primary cross-step data store -----
    def set(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.artifacts

    def remove(self, key: str) -> None:
        if key in self.artifacts:
            del self.artifacts[key]

    def update(self, **items: Any) -> None:
        self.artifacts.update(items)

    # ----- Validation helpers -----
    def require(self, keys: List[str]) -> None:
        missing = [k for k in keys if k not in self.artifacts]
        if missing:
            raise KeyError(f"Missing required context keys: {', '.join(missing)}")

    # ----- Run ID (generic, stored in artifacts) -----
    def get_run_id(self) -> Optional[str]:
        return self.get(self.RUN_ID_KEY, None)

    def set_run_id(self, run_id: str) -> None:
        self.set(self.RUN_ID_KEY, run_id)

    def ensure_run_id(self, factory: Optional[Callable[[], str]] = None) -> str:
        rid = self.get_run_id()
        if not rid and factory:
            rid = factory()
        if not rid:
            import uuid as _uuid

            rid = str(_uuid.uuid4())
        self.set_run_id(rid)
        return rid

    # Note: temp_dir management intentionally not part of the generic pipeline context


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@runtime_checkable
class Step(Protocol):
    async def __call__(
        self, context: PipelineContext
    ) -> None:  # pragma: no cover - protocol
        ...


logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Base class for pipeline steps with lifecycle hooks, status, retry & timeout."""

    name: str = "base_step"

    # execution config
    required_keys: List[str] = []
    retries: int = 0
    retry_backoff: float = 0.5  # seconds
    timeout: Optional[float] = None  # seconds
    use_exponential_backoff: bool = True
    max_backoff: float = 5.0
    jitter: float = 0.1  # added random [0, jitter) seconds
    # Per-exception retry policy overrides
    # Example: {NetworkError: {"retries": 3, "retry_backoff": 0.5, "max_backoff": 3.0, "jitter": 0.2}}
    retry_exceptions: Dict[type[Exception], Dict[str, Any]] = {}

    # runtime fields
    status: StepStatus = StepStatus.PENDING
    last_error: Optional[Exception] = None
    duration: float = 0.0
    attempts: int = 0

    async def __call__(self, context: PipelineContext) -> None:
        attempts = 0
        start_total = perf_counter()
        self.last_error = None

        # validate and maybe skip
        if not self.validate_inputs(context):
            missing = [k for k in self.required_keys if not context.has(k)]
            raise KeyError(
                f"Missing required inputs for step '{self.name}': {', '.join(missing)}"
            )

        if self.can_skip(context):
            self.status = StepStatus.SKIPPED
            self.on_skip(context)
            return

        while True:
            attempts += 1
            self.attempts = attempts
            self.status = StepStatus.RUNNING
            self.on_start(context)
            start = perf_counter()
            try:
                if self.timeout:
                    await asyncio.wait_for(self.run(context), timeout=self.timeout)
                else:
                    await self.run(context)
                self.status = StepStatus.COMPLETED
                return
            except Exception as e:  # noqa: BLE001
                self.last_error = e
                self.status = StepStatus.FAILED
                # resolve effective retry policy based on exception type
                eff_retries = self.retries
                eff_backoff = self.retry_backoff
                eff_max_backoff = self.max_backoff
                eff_jitter = self.jitter
                eff_exp = self.use_exponential_backoff

                for exc_type, cfg in getattr(self, "retry_exceptions", {}).items():
                    if isinstance(e, exc_type):
                        eff_retries = int(cfg.get("retries", eff_retries))
                        eff_backoff = float(cfg.get("retry_backoff", eff_backoff))
                        eff_max_backoff = float(cfg.get("max_backoff", eff_max_backoff))
                        eff_jitter = float(cfg.get("jitter", eff_jitter))
                        eff_exp = bool(cfg.get("use_exponential_backoff", eff_exp))
                        break

                if attempts <= eff_retries:
                    # compute backoff with optional exponential and jitter
                    base = max(0.0, float(eff_backoff))
                    if eff_exp:
                        sleep_s = base * (2 ** max(0, attempts - 1))
                    else:
                        sleep_s = base
                    sleep_s = min(float(eff_max_backoff), float(sleep_s))
                    sleep_s += random.uniform(0.0, max(0.0, float(eff_jitter)))
                    await asyncio.sleep(sleep_s)
                    continue
                raise
            finally:
                self.duration = perf_counter() - start
                self.on_finish(context, self.duration)
        # end while
        self.duration = perf_counter() - start_total

    @abstractmethod
    async def run(
        self, context: PipelineContext
    ) -> None:  # pragma: no cover - abstract
        ...

    # Hooks
    def on_start(self, context: PipelineContext) -> None:
        logger.debug("Step %s start", getattr(self, "name", self.__class__.__name__))

    def on_finish(self, context: PipelineContext, duration: float) -> None:
        logger.info(
            "Step %s finished in %.3fs with status=%s attempts=%d run_id=%s",
            getattr(self, "name", self.__class__.__name__),
            duration,
            self.status.value,
            getattr(self, "attempts", 0),
            context.get_run_id(),
        )

    def on_skip(self, context: PipelineContext) -> None:
        logger.info("Step %s skipped", getattr(self, "name", self.__class__.__name__))

    # Utilities
    def validate_inputs(self, context: PipelineContext) -> bool:
        if not self.required_keys:
            return True
        return all(context.has(k) for k in self.required_keys)

    def can_skip(self, context: PipelineContext) -> bool:
        return False


from typing import TypedDict


class StepResult(TypedDict):  # pragma: no cover - typing helper
    name: str
    status: str
    duration: float
    error: Optional[str]
    attempts: int


class PipelineResult(TypedDict):  # pragma: no cover - typing helper
    success: bool
    duration: float
    steps: List[StepResult]
    error: Optional[str]
    context: PipelineContext


class Pipeline:
    def __init__(self, steps: List[Step], *, fail_fast: bool = True):
        self._steps = steps
        self.fail_fast = fail_fast

    async def execute(self, context: PipelineContext) -> PipelineResult:
        # ensure run_id exists (temp_dir handling removed)
        context.ensure_run_id()

        pipeline_start = perf_counter()
        results: Dict[str, Any] = {
            "success": False,
            "duration": 0.0,
            "steps": [],
            "error": None,
        }

        for step in self._steps:
            step_info: Dict[str, Any] = {
                "name": getattr(step, "name", step.__class__.__name__),
                "status": StepStatus.PENDING.value,
                "duration": 0.0,
                "error": None,
                "attempts": 0,
            }
            results["steps"].append(step_info)

            step_start = perf_counter()
            try:
                await step(context)  # use __call__ lifecycle
                step_info["status"] = getattr(
                    step, "status", StepStatus.COMPLETED
                ).value
                step_info["attempts"] = int(getattr(step, "attempts", 1) or 1)
            except Exception as e:  # noqa: BLE001
                step_info["status"] = StepStatus.FAILED.value
                step_info["error"] = str(e)
                step_info["attempts"] = int(getattr(step, "attempts", 1) or 1)
                if self.fail_fast:
                    # Propagate the original exception to the caller for fail-fast behavior
                    raise
            finally:
                step_info["duration"] = perf_counter() - step_start

        results["duration"] = perf_counter() - pipeline_start
        # success when all steps completed (works for fail_fast=False)
        results["success"] = all(
            s.get("status") == StepStatus.COMPLETED.value for s in results["steps"]
        )
        results["context"] = context
        return results


class Middleware(Protocol):  # pragma: no cover - optional extension point
    def __call__(self, step: Step) -> Step: ...


def make_logging_middleware(
    logger_obj: logging.Logger | None = None,
    level_before: int = logging.DEBUG,
    level_after: int = logging.INFO,
) -> Middleware:
    """Return a middleware that logs before and after each step execution.

    Logs include: step name, run_id, status, attempts, duration.
    """
    _log = logger_obj or logger

    def _middleware(step: Step) -> Step:
        class _Wrapped:
            # keep common attributes for downstream access
            def __init__(self, inner: Step):
                self._inner = inner

            def __getattr__(self, item):  # delegate attributes like 'name'
                return getattr(self._inner, item)

            async def __call__(self, context: PipelineContext) -> None:
                step_name = getattr(self._inner, "name", self._inner.__class__.__name__)
                rid = context.get_run_id()
                _log.log(level_before, "[run_id=%s] Step %s BEGIN", rid, step_name)
                from time import perf_counter as _pc

                _start = _pc()
                try:
                    await self._inner(context)
                finally:
                    duration = _pc() - _start
                    status = getattr(self._inner, "status", StepStatus.PENDING)
                    attempts = int(getattr(self._inner, "attempts", 0) or 0)
                    _log.log(
                        level_after,
                        "[run_id=%s] Step %s END status=%s attempts=%d duration=%.3fs",
                        rid,
                        step_name,
                        getattr(status, "value", str(status)),
                        attempts,
                        duration,
                    )

        return _Wrapped(step)

    return _middleware


class PipelineFactory:
    """Fluent builder for Pipelines, with optional middlewares per step.

    Example:
        factory = PipelineFactory()
        pipeline = factory.add(step1).add(step2).build()
    """

    def __init__(
        self, *, middlewares: List[Middleware] | None = None, fail_fast: bool = True
    ):
        self._steps: List[Step] = []
        self._middlewares = list(middlewares or [])
        self._fail_fast = fail_fast

    def add(self, step: Step) -> "PipelineFactory":
        wrapped = step
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
