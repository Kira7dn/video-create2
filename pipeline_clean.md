# Hướng dẫn triển khai Pipeline theo Clean/Onion Architecture (cập nhật)

Tài liệu này hướng dẫn triển khai cấu trúc: Steps → Pipeline → UseCase + Adapters (DI) → Endpoint.
Thiết kế tuân thủ OOP, SOLID, Clean/Onion: Endpoint ở Delivery, UseCase và Steps ở Application, Adapters ở Infrastructure.

## Kiến trúc tổng quan (Clean/Onion Architecture)

- **Delivery Layer** (`app/api/v1/endpoints/*`): nhận request HTTP, gọi UseCase qua DI.
- **Application Layer** (`app/application/`):
  - UseCase: điều phối business logic, khởi tạo Pipeline.
  - Pipeline + Steps: thực thi tuần tự các bước xử lý.
  - Interfaces (Ports): định nghĩa contracts cho Infrastructure.
- **Infrastructure Layer** (`app/infrastructure/adapters/*`): triển khai Ports, tương tác với external services.

## Cấu trúc thư mục (thu gọn)

````text
app/
  api/
    v1/
      endpoints/
        pipelines.py                 # Endpoint gọi UseCase (ví dụ trong tài liệu)

  application/
    pipeline/
      base.py                        # PipelineContext, BaseStep, Pipeline
      factory.py                     # PipelineFactory (tùy chọn middleware)
      video/                         # Video pipeline package
        __init__.py                  # (tùy chọn) export public API
        adapter_bundle.py            # VideoPipelineAdapters (bundle)
        builder.py                   # Builders
        steps/                       # Mỗi step tách file riêng

    interfaces/                      # Ports (Application layer contracts)

    use_cases/
      run_video_pipeline.py          # RunVideoPipelineUseCase

  core/
    dependencies.py                  # DI: khởi tạo adapters và pipeline, trả về UseCase

  infrastructure/
    adapters/                        # Adapters (Infrastructure layer)

config/
  settings.py                        # Cấu hình (ví dụ các loại asset, đường dẫn)

utils/                               # Tiện ích (nếu dùng trong steps/adapters)

## Hợp đồng cốt lõi

Các primitive được định nghĩa tại `app/application/pipeline/base.py`:

- `PipelineContext`:
  - `input: Mapping[str, Any]` (read-only usage): payload gốc của run.
  - `artifacts: Dict[str, Any]`: dữ liệu làm việc và outputs giữa các bước.
  - Reserved keys trong artifacts: `_run_id`, `_temp_dir` với helpers:
    - `ensure_run_id()`, `get_run_id()`
    - `ensure_temp_dir()`, `get_temp_dir()`
- `Step` (Protocol): interface cho mỗi step, triển khai `async def run(context: PipelineContext) -> None`.
- `BaseStep`: abstract base class với lifecycle hooks (`on_start/on_finish`), timing và error handling.
- `Pipeline`: orchestrator nhận danh sách steps và thực thi tuần tự qua `execute(context)`.

### PipelineContext
```python
from typing import Any, Dict, Mapping, Optional
from pathlib import Path
import uuid
import tempfile

class PipelineContext:
    """Context object chứa input và artifacts cho pipeline execution."""

    def __init__(self, input: Mapping[str, Any]):
        self._input = input or {}
        self.artifacts: Dict[str, Any] = {}

    @property
    def input(self) -> Mapping[str, Any]:
        """Read-only access to input payload."""
        return self._input

    def get_input(self) -> Mapping[str, Any]:
        """Get input payload (alternative method)."""
        return self._input

    def get(self, key: str, default: Any = None) -> Any:
        """Get artifact by key."""
        return self.artifacts.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set artifact value."""
        self.artifacts[key] = value

    # Reserved keys helpers
    def ensure_run_id(self) -> str:
        """Ensure run_id exists, create if not."""
        if "_run_id" not in self.artifacts:
            self.artifacts["_run_id"] = str(uuid.uuid4())
        return self.artifacts["_run_id"]

    def get_run_id(self) -> Optional[str]:
        """Get run_id if exists."""
        return self.artifacts.get("_run_id")

    def ensure_temp_dir(self) -> Path:
        """Ensure temp directory exists, create if not."""
        if "_temp_dir" not in self.artifacts:
            temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_"))
            self.artifacts["_temp_dir"] = str(temp_dir)
        return Path(self.artifacts["_temp_dir"])

    def get_temp_dir(self) -> Optional[Path]:
        """Get temp directory if exists."""
        temp_path = self.artifacts.get("_temp_dir")
        return Path(temp_path) if temp_path else None
````

### Step Protocol

```python
from typing import Protocol

class Step(Protocol):
    """Interface cho mỗi pipeline step."""

    async def run(self, context: PipelineContext) -> None:
        """Execute step logic.

        Args:
            context: Pipeline context chứa input và artifacts
        """
        ...
```

### BaseStep

```python
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseStep(ABC):
    """Abstract base class với lifecycle hooks và error handling."""

    name: str = "unnamed_step"

    async def __call__(self, context: PipelineContext) -> None:
        """Execute step với lifecycle management."""
        start_time = time.time()

        try:
            logger.info(f"Starting step: {self.name}")
            await self.on_start(context)
            await self.run(context)
            await self.on_finish(context)

            duration = time.time() - start_time
            logger.info(f"Completed step: {self.name} in {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed step: {self.name} after {duration:.2f}s - {e}")
            await self.on_error(context, e)
            raise

    @abstractmethod
    async def run(self, context: PipelineContext) -> None:
        """Main step logic - must be implemented by subclasses."""
        pass

    async def on_start(self, context: PipelineContext) -> None:
        """Hook called before step execution."""
        pass

    async def on_finish(self, context: PipelineContext) -> None:
        """Hook called after successful step execution."""
        pass

    async def on_error(self, context: PipelineContext, error: Exception) -> None:
        """Hook called when step fails."""
        pass
```

### Pipeline

```python
from typing import List

class Pipeline:
    """Orchestrator thực thi tuần tự các steps."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute all steps sequentially.

        Args:
            context: Pipeline context

        Returns:
            Dict containing final context and metadata
        """
        logger.info(f"Starting pipeline with {len(self.steps)} steps")

        try:
            for i, step in enumerate(self.steps):
                logger.info(f"Executing step {i+1}/{len(self.steps)}: {getattr(step, 'name', 'unnamed')}")

                if hasattr(step, '__call__'):
                    await step(context)  # BaseStep với lifecycle
                else:
                    await step.run(context)  # Raw Step protocol

            logger.info("Pipeline completed successfully")
            return {
                "status": "success",
                "context": context.artifacts,
                "steps_executed": len(self.steps)
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "context": context.artifacts,
                "steps_executed": i if 'i' in locals() else 0
            }
```

### PipelineFactory

```python
from typing import List

class PipelineFactory:
    """Factory để build Pipeline với middleware support."""

    def __init__(self):
        self._steps: List[Step] = []

    def add(self, step: Step) -> 'PipelineFactory':
        """Add step to pipeline."""
        self._steps.append(step)
        return self

    def build(self) -> Pipeline:
        """Build final Pipeline instance."""
        if not self._steps:
            raise ValueError("Pipeline must have at least one step")
        return Pipeline(self._steps)
```

## Quy ước sử dụng PipelineContext (Data Flow)

- **Input (immutable)**: đọc từ `context.input` hoặc `context.get_input()`. KHÔNG ghi vào input.
- **Artifacts (mutable state)**: ghi/đọc outputs qua `context.artifacts` với keys rõ ràng:
  - Naming: `snake_case`, có thể namespace (vd: `video.segment_clips`, `audio.synthesized_path`).
  - Flow: `validated_data` → `processed_items` → `final_output`.
- **Reserved keys**: `_run_id`, `_temp_dir` cho metadata kỹ thuật, không trộn với business data.

## Tạo một Step mới (ví dụ)

```python
from app.application.pipeline.base import BaseStep, PipelineContext

class ValidateInputStep(BaseStep):
    name = "validate_input"

    async def run(self, context: PipelineContext) -> None:
        # 1. Setup run metadata
        context.ensure_run_id()
        context.ensure_temp_dir()

        # 2. Validate business input
        data = context.get_input()
        if not isinstance(data, dict) or "json_data" not in data:
            raise ValueError("input.json_data is required")

        jd = data.get("json_data", {})
        if not isinstance(jd, dict) or not isinstance(jd.get("segments"), list):
            raise ValueError("'segments' list is required")

        # 3. Set validated output
        context.set("validated_data", jd)
```

## Lắp ráp Pipeline và DI (Steps → Pipeline)

```python
# Sử dụng Adapter Bundle pattern (khuyến nghị)
from app.application.pipeline.video.builder import build_video_pipeline_via_container
from app.application.pipeline.video.adapter_bundle import VideoPipelineAdapters

# Tạo adapter bundle
adapters = VideoPipelineAdapters(
    assets=assets,
    renderer=renderer,
    uploader=uploader,
)

# Build pipeline với bundle
pipeline = build_video_pipeline_via_container(adapters)
```

## Dependency Provider (Adapters → UseCase)

```python
# app/core/dependencies.py (ví dụ tham khảo)
from app.application.pipeline.video.builder import build_video_pipeline_via_container
from app.application.pipeline.video.adapter_bundle import VideoPipelineAdapters
from app.application.use_cases.run_video_pipeline import RunVideoPipelineUseCase

# Ví dụ các factory adapters tầng hạ tầng
from app.infrastructure.adapters.asset_repo import AssetRepoFactory  # ví dụ
from app.infrastructure.adapters.video_renderer import VideoRendererFactory  # ví dụ
from app.infrastructure.adapters.uploader import UploaderFactory  # ví dụ


def get_video_use_case() -> RunVideoPipelineUseCase:
    assets = AssetRepoFactory.create()
    renderer = VideoRendererFactory.create()
    uploader = UploaderFactory.create()

    # Sử dụng Adapter Bundle pattern
    adapters = VideoPipelineAdapters(
        assets=assets,
        renderer=renderer,
        uploader=uploader,
    )
    pipeline = build_video_pipeline_via_container(adapters)
    return RunVideoPipelineUseCase(pipeline=pipeline)
```

## UseCase và Endpoint (Endpoint → UseCase)

```python
# app/application/use_cases/run_video_pipeline.py
from app.application.pipeline.base import PipelineContext

class RunVideoPipelineUseCase:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    async def execute(self, context: PipelineContext | None = None) -> dict:
        ctx = context or PipelineContext(input={})
        result = await self._pipeline.execute(ctx)
        return result["context"]

# app/api/v1/endpoints/pipelines.py
from fastapi import APIRouter, Depends
from app.application.pipeline.base import PipelineContext
from app.application.use_cases.run_video_pipeline import RunVideoPipelineUseCase
from app.core.dependencies import get_video_use_case

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

@router.post("/run-video")
async def run_video_pipeline(
    payload: dict,
    use_case: RunVideoPipelineUseCase = Depends(get_video_use_case),
):
    ctx = PipelineContext(input=payload)
    context = await use_case.execute(ctx)
    return {"run_id": context.get("_run_id"), "final_video": context.get("final_video_url")}
```

Tại endpoint, sử dụng DI để cung cấp adapters và build pipeline rồi gọi UseCase.

## Best practices

- **SRP cho Step**: mỗi Step làm một việc rõ ràng, đặt tên artifact đầu ra rõ ràng.
- **Typing nhẹ**: cân nhắc TypedDict/Pydantic cho artifacts quan trọng (vd: `ValidatedData`, `SegmentClip`).
- **Idempotency và retry**: đối với IO, dùng đường dẫn xác định dưới `ensure_temp_dir()` và skip nếu tồn tại.
- **Observability**: log theo `run_id`, đo thời gian bằng `BaseStep`.
- **Không chia sẻ artifacts giữa pipelines khác nhau** trừ khi có namespace rõ ràng.

---

Ghi chú triển khai bổ sung:

- Sử dụng `PipelineContext.ensure_run_id()` và `ensure_temp_dir()` ngay ở bước đầu để ổn định I/O.
- Artifacts chính:
  - `validated_data` → `processed_segments` → `segment_clips` → `final_video_path` → `final_video_url`.
- Builder sử dụng Adapter Container pattern để quản lý dependencies một cách scalable và type-safe.

---

## Template tạo pipeline mới (chuẩn duy nhất)

Mô hình: Steps → Pipeline → Builder (public) → UseCase → Endpoint

1. Tạo Steps

```python
# app/application/pipeline/my_pipeline/steps.py
from app.application.pipeline.base import BaseStep, PipelineContext

class ValidateInputStep(BaseStep):
    name = "validate_input"
    async def run(self, context: PipelineContext) -> None:
        data = context.get_input()
        # ... validate
        context.set("validated_data", data)
        context.ensure_run_id()
        context.ensure_temp_dir()

class ProcessDataStep(BaseStep):
    def __init__(self, processor: IProcessor) -> None:
        super().__init__()
        self._processor = processor

    name = "process_data"

    async def run(self, context: PipelineContext) -> None:
        # 1. Get input from previous step
        validated_data = context.get("validated_data")
        if not validated_data:
            raise ValueError("validated_data is required from previous step")

        # 2. Process via injected adapter
        result = await self._processor.process(validated_data)

        # 3. Set output for next step
        context.set("processed_result", result)

class SaveResultStep(BaseStep):
    def __init__(self, storage: IStorage) -> None:
        super().__init__()
        self._storage = storage

    name = "save_result"

    async def run(self, context: PipelineContext) -> None:
        result = context.get("processed_result")
        run_id = context.get_run_id()

        save_path = f"results/{run_id}/output.json"
        saved_url = await self._storage.save(result, save_path)
        context.set("saved_url", saved_url)

class NotifyStep(BaseStep):
    def __init__(self, notifier: INotifier) -> None:
        super().__init__()
        self._notifier = notifier

    name = "notify_completion"

    async def run(self, context: PipelineContext) -> None:
        run_id = context.get_run_id()
        saved_url = context.get("saved_url")

        message = f"Pipeline {run_id} completed. Result: {saved_url}"
        await self._notifier.notify(message)
```

2. Định nghĩa Interfaces (Ports) nếu cần adapters

```python
# app/application/interfaces/processor.py
from typing import Protocol

class IProcessor(Protocol):
    async def process(self, payload: dict) -> str: ...

class IStorage(Protocol):
    async def save(self, data: dict, path: str) -> str: ...

class INotifier(Protocol):
    async def notify(self, message: str) -> None: ...
```

3. Tạo Adapter Container (Application Layer)

```python
# app/application/pipeline/my_pipeline/adapters.py
from dataclasses import dataclass
from typing import Optional
from app.application.interfaces.processor import IProcessor, IStorage, INotifier

@dataclass
class MyPipelineAdapters:
    """Container cho tất cả adapters của MyPipeline.

    Chứa Application interfaces (Ports), được populate bởi Infrastructure.
    """
    processor: IProcessor
    storage: IStorage
    notifier: Optional[INotifier] = None  # Optional adapter
```

4. Tạo Builder (public API, STRICT)

```python
# app/application/pipeline/my_pipeline/orchestrator.py
from app.application.pipeline.base import Pipeline
from app.application.pipeline.factory import PipelineFactory
from app.application.pipeline.my_pipeline.steps import ValidateInputStep, ProcessDataStep, SaveResultStep, NotifyStep
from app.application.pipeline.my_pipeline.adapters import MyPipelineAdapters

def build_my_pipeline_via_builder(adapters: MyPipelineAdapters) -> Pipeline:
    """Build pipeline với Adapter Container pattern.

    Args:
        adapters: Container chứa tất cả required/optional adapters

    Returns:
        Configured Pipeline ready to execute
    """
    factory = PipelineFactory()
    factory.add(ValidateInputStep())
    factory.add(ProcessDataStep(adapters.processor))
    factory.add(SaveResultStep(adapters.storage))

    # Optional steps dựa trên adapters có sẵn
    if adapters.notifier is not None:
        factory.add(NotifyStep(adapters.notifier))

    return factory.build()
```

5. UseCase

```python
# app/application/use_cases/run_my_pipeline.py
from app.application.pipeline.base import PipelineContext, Pipeline

class RunMyPipelineUseCase:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, ctx: PipelineContext) -> dict:
        result = await self._pipeline.execute(ctx)
        return result["context"]
```

6. DI Provider (Infrastructure Layer)

```python
# app/core/dependencies.py
from app.application.pipeline.my_pipeline.orchestrator import build_my_pipeline_via_builder
from app.application.pipeline.my_pipeline.adapters import MyPipelineAdapters
from app.application.use_cases.run_my_pipeline import RunMyPipelineUseCase

# Infrastructure factories
from app.infrastructure.adapters.processor import ProcessorFactory
from app.infrastructure.adapters.storage import StorageFactory
from app.infrastructure.adapters.notifier import NotifierFactory

def get_my_pipeline_use_case() -> RunMyPipelineUseCase:
    """DI Provider: Infrastructure tạo adapters và inject vào Application."""
    # Infrastructure Layer: Tạo concrete implementations
    processor = ProcessorFactory.create()
    storage = StorageFactory.create()
    notifier = NotifierFactory.create()  # Optional

    # Application Layer: Populate adapter container
    adapters = MyPipelineAdapters(
        processor=processor,
        storage=storage,
        notifier=notifier,
    )

    # Application Layer: Build pipeline với container
    pipeline = build_my_pipeline_via_builder(adapters)
    return RunMyPipelineUseCase(pipeline=pipeline)
```

7. Endpoint (tùy chọn)

```python
# app/api/v1/endpoints/my_pipeline.py
from fastapi import APIRouter, Depends
from app.application.pipeline.base import PipelineContext
from app.application.use_cases.run_my_pipeline import RunMyPipelineUseCase
from app.core.dependencies import get_my_pipeline_use_case

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

@router.post("/run-my-pipeline")
async def run_my_pipeline(payload: dict, use_case: RunMyPipelineUseCase = Depends(get_my_pipeline_use_case)):
    ctx = PipelineContext(input=payload)
    context = await use_case.execute(ctx)
    return {"run_id": context.get("_run_id"), "result": context.get("saved_url")}
```

## Checklist tuân thủ Clean/Onion/SOLID:

**Dependency Rule (Clean Architecture)**:

- Application layer (Steps, UseCase, Adapter Container) KHÔNG import từ Infrastructure.
- Infrastructure adapters implement Application interfaces (Ports).
- Dependencies flow inward: Delivery → Application → Infrastructure.

**Adapter Container Pattern**:

- Container nằm ở Application Layer, chứa Application interfaces (Ports).
- Infrastructure populate container với concrete implementations.
- Builder nhận container thay vì individual parameters (giảm parameter explosion).
- Scalable cho pipelines phức tạp với nhiều adapters.

**STRICT Pipeline Assembly**:

- Container validate required adapters tại compile-time (dataclass).
- DI Provider luôn khởi tạo đủ adapters trước khi build pipeline.
- Steps nhận adapters qua constructor (Dependency Injection).
- Optional adapters được handle gracefully trong builder.

**Single Responsibility (SOLID)**:

- Mỗi Step làm 1 việc rõ ràng, có tên mô tả chức năng.
- UseCase chỉ điều phối, không chứa business logic chi tiết.
- Adapters chỉ handle I/O, không chứa business rules.
- Container chỉ group adapters, không chứa logic.

**Data Flow & Context Management**:

- Step đầu gọi `ensure_run_id()` và `ensure_temp_dir()`.
- Artifacts đặt tên rõ ràng theo flow: input → processing → output.
- Không ghi đè artifacts tùy tiện, có namespace khi cần.
