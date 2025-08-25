# Feature Delivery Workflow (LLM-Friendly Onion/Clean Architecture)

This document is a repeatable, end-to-end guide for delivering any feature.
Optimized for **LLM-assisted development**: structured, consistent, and easy to parse.

---

## I. General Rules

- Always run inside venv:
  - `venv/bin/python -m pytest -q`
  - `venv/bin/alembic revision --autogenerate -m "<msg>" && venv/bin/alembic upgrade head`
- Dependency direction: **Presentation → Application → Domain**. Infrastructure implements Application interfaces.
- Use Cases call only **interfaces** in `application/interfaces/`, never raw SDKs.
- Config & secrets → `backend/app/core/config.py` + `.env`. Never hardcode.
- Endpoints = thin: validate → call Use Case → return schema.
- Write tests as soon as Use Case is created (mock Application interfaces).
- External API usage:
  - **Domain**: pure business logic, no I/O
  - **Application**: defines ports (interfaces)
  - **Infrastructure**: SDK/API/DB adapters implementing ports
  - **Presentation**: DI + routers
- Testing strategy:
  - `-m "not integration"` → fast unit
  - `-m integration` → infra tests
  - `httpx.AsyncClient` for API E2E
- DB sessions: short-lived, per-request (`app/core/db.py::get_db`).

### Naming & Pluralization Rules

- Resource names use snake_case; collections use plural snake_case.
  - Examples: `user -> users`, `order -> orders`, `product_category -> product_categories`.
  - Irregulars map: `person -> people`, `category -> categories`.
- Routers: set `APIRouter(prefix="/{plural}")`; then use empty path for collection create (`@router.post("")`).
- SQLAlchemy `__tablename__`: plural snake_case of model name (trim trailing `Model`). See Infra Persistence workflow.

---

## II. Directory Map

```text
backend/app/
├── domain/
│   ├── entities/
│   └── services/
├── application/
│   ├── interfaces/   # I<Name> interfaces
│   └── use_cases/
├── infrastructure/
│   ├── database/
│   ├── repositories/
│   ├── adapters/
│   ├── models/
│   └── pipelines/
├── presentation/
│   ├── api/v1/dependencies/
│   ├── api/v1/routers/
│   ├── api/v1/schemas/
│   └── main.py
└── core/
    └── config.py

backend/
└── alembic/
```

Note: Alembic lives at `backend/alembic/` (not inside `backend/app/`).

---

## III. Workflow (Step-by-Step)

### Step 0 – Clarify Requirements

**[Spec]**

- Define Acceptance Criteria: input/output, latency, security.
- Identify Entities, Use Cases, AI integration, risks.

**[Action]**

- Ask: related entities? Expected input/output? Real-time vs offline AI? Data volume/security?
- Risk: AI latency → use cache; Invalid input → Pydantic validation.
- Example feature:
  - Entities: Product (id, name, category, price_range).
  - AI logic: Recommendation service using cosine similarity over vectors derived from category and price_range.
  - Use cases: Create a new product; Recommend similar products based on user preferences (e.g., list of categories).
  - Risks: Large data may be slow (solution: vector DB like pgvector); Invalid input may crash AI (solution: Pydantic validation).

---

### Step 1 – Domain Layer (Pure Business Logic)

**[Spec]**

- Entities: Pydantic models.
- Services: pure functions/AI logic.

**[Action]**

- Define entities with Pydantic BaseModel and validators.
- Define services containing business/AI logic (import libs like numpy, sklearn as needed).
- No external deps (no DB, HTTP, file I/O).

**[Sample Code]**

```python
# backend/app/domain/entities/product.py (minimal example)
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    id: int
    name: str
    category: str  # 'electronics' | 'books' | 'clothing'
    price_range: str  # 'low' | 'medium' | 'high'

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        allowed = ['electronics', 'books', 'clothing']
        if v not in allowed:
            raise ValueError(f"Invalid category: {v}")
        return v
```

```python
# backend/app/domain/services/recommendation.py (example AI service)
from typing import List
import numpy as np
from app.domain.entities.product import Product

class ProductRecommendationService:
    """Pure business logic for product recommendations."""

    @staticmethod
    def calculate_similarity(product1: Product, product2: Product) -> float:
        """Calculate similarity score between two products."""
        # Simple categorical similarity (in real app, use embeddings)
        category_match = 1.0 if product1.category == product2.category else 0.0
        price_match = 1.0 if product1.price_range == product2.price_range else 0.5
        return (category_match + price_match) / 2.0

    @classmethod
    def recommend_similar(cls, target: Product, candidates: List[Product], limit: int = 5) -> List[Product]:
        """Return most similar products to target."""
        scored = [(p, cls.calculate_similarity(target, p)) for p in candidates if p.id != target.id]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, score in scored[:limit]]
```

---

### Step 2 – Application Layer (Use Cases + Interfaces)

**[Spec]**

- Define ports (interfaces) with prefix `I<Name>`.
- Use Cases orchestrate logic via interfaces to connect domain to infrastructure without tight coupling.

**[Action]**

- Define interfaces (ABC) for repositories.
- Define use cases that inject repos and call services.
- Risks: Complex use cases create tight coupling (solution: keep orchestration simple and mockable); Unvalidated AI inputs (solution: Pydantic in entities).

Tip: Tests should use realistic fakes over heavy mocks. See Appendix in `generate-application-class.md` for fake repo patterns.

**[Sample Code]**

- **Interfaces**

```python
# backend/app/application/interfaces/product.py
from abc import ABC, abstractmethod
from app.domain.entities.product import Product

class IProductRepository(ABC):
    @abstractmethod
    def create(self, product: Product) -> Product: ...
    @abstractmethod
    def get_by_id(self, product_id: int) -> Product: ...
    @abstractmethod
    def get_all(self) -> list[Product]: ...
```

- **Use Cases**

```python
# backend/app/application/use_cases/create_product_use_case.py
from app.application.interfaces.product import IProductRepository
from app.domain.entities.product import Product

class CreateProductUseCase:
    def __init__(self, repo: IProductRepository):
        self.repo = repo
    def execute(self, name: str, category: str, price_range: str) -> Product:
        return self.repo.create(
            Product(id=0, name=name, category=category, price_range=price_range)
        )
```

---

### Step 3 – Infrastructure Layer

**[Spec]**

- Implements Application interfaces (repos/adapters).
- Handles DB, external APIs, pipelines.
- Covers persistence (models, repositories, Alembic migrations), concrete external adapters (SDKs/HTTP/queues), and pipeline storage/observability.

**[Action]**

- **Persistence (DB)**:

  - Create SQLAlchemy models in `backend/app/infrastructure/models/` with proper table names, columns, and constraints.
  - Implement repository classes in `backend/app/infrastructure/repositories/` that inherit from Application interfaces (e.g., `IProductRepository`).
  - Map between domain entities and DB models in repository methods (create, get_by_id, get_all, etc.).
  - Generate and run Alembic migrations: `venv/bin/alembic revision --autogenerate -m "add <table_name> table"` then `venv/bin/alembic upgrade head`.

- **External Adapters**:

  - Create adapter classes in `backend/app/infrastructure/adapters/` (one file per service) that implement Application interfaces.
  - Add retry logic with exponential backoff, request timeouts, and error handling for SDK calls.
  - Read configuration from `app/core/config.py` (never hardcode secrets).
  - Implement idempotency for create operations using keys or input hashing.
  - Return plain dicts/DTOs, not raw SDK objects.
  - Create DI factories in `presentation/api/v1/dependencies/` for injection into endpoints.
  - Type DI providers to Application interfaces (Protocols/ABCs), not concrete classes.

- **Pipeline**:
  - When durability/observability needed: create models for `pipeline_runs`, `pipeline_steps`, `pipeline_artifacts`.
  - Implement step classes with `run(context)` method and compose via `SimplePipeline` orchestrator.
  - Persist pipeline progress and results for restart capability.

**[Sample Code]**

- **Persistence (DB):**

```python
# backend/app/infrastructure/models/product.py
from sqlalchemy import Column, Integer, String
from app.infrastructure.database.base import Base

class ProductModel(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    price_range = Column(String, nullable=False)
```

```python
# backend/app/infrastructure/repositories/product.py
from sqlalchemy.orm import Session
from app.application.interfaces.product import IProductRepository
from app.domain.entities.product import Product
from app.infrastructure.models.product import ProductModel

class ProductRepository(IProductRepository):
    def __init__(self, db: Session):
        self.db = db
    def create(self, product: Product) -> Product:
        db_product = ProductModel(
            name=product.name, category=product.category, price_range=product.price_range
        )
        self.db.add(db_product); self.db.commit(); self.db.refresh(db_product)
        return Product(id=db_product.id, name=db_product.name,
                       category=db_product.category, price_range=db_product.price_range)
```

- **External Adapters:**

```python
# backend/app/infrastructure/adapters/payment.py
from app.application.interfaces.payment import IPaymentGateway
from app.core.config import settings

class StripeClient(IPaymentGateway):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_payment_intent(self, amount_cents: int, currency: str, metadata: dict) -> dict:
        # Implementation with retry/timeout logic
        return {"id": "pi_123", "status": "requires_payment_method"}
```

- **Pipeline:**

```python
# backend/app/application/interfaces/pipeline.py
from typing import Protocol, Any, Dict, List

class IPipelineStep(Protocol):
    name: str
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]: ...

class IPipeline(Protocol):
    steps: List[IPipelineStep]
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]: ...
```

```python
# backend/app/infrastructure/pipelines/steps/transcribe_step.py
from typing import Dict, Any
from app.application.interfaces.media import ITranscriber

class TranscribeStep:
    name = "transcribe"
    def __init__(self, transcriber: ITranscriber):
        self.transcriber = transcriber
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["text"] = self.transcriber.transcribe(context["audio_path"])
        return context
```

```python
# backend/app/application/use_cases/run_pipeline.py (orchestrator example)
from typing import Dict, Any, List
from app.application.interfaces.pipeline import IPipelineStep

class SimplePipeline:
    def __init__(self, steps: List[IPipelineStep]):
        self.steps = steps
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.steps:
            context = step.run(context)
        return context
```

```python
# Compose and run (example)
from app.application.use_cases.run_pipeline import SimplePipeline
pipeline = SimplePipeline([TranscribeStep(transcriber)])
result = pipeline.execute({"audio_path": "/tmp/audio.wav"})
```

#### Video Pipeline (Clean Architecture-aligned)

```python
# Application-layer builder + bundle
from app.application.pipeline.video.builder import build_video_pipeline_via_container
from app.application.pipeline.video.adapter_bundle import VideoPipelineAdapters

# Infrastructure provides concrete adapters (examples)
from app.infrastructure.adapters.asset_repo import AssetRepo  # example
from app.infrastructure.adapters.video_renderer import VideoRenderer  # example
from app.infrastructure.adapters.uploader import S3Uploader  # example

adapters = VideoPipelineAdapters(
    assets=AssetRepo(),
    renderer=VideoRenderer(),
    uploader=S3Uploader(),
)

pipeline = build_video_pipeline_via_container(adapters)

# Execute via Use Case (typical)
from app.application.pipeline.base import PipelineContext
ctx = PipelineContext(input={"json_data": {"segments": []}})
result = await pipeline.execute(ctx)  # if async pipeline
```

```python
# Presentation DI provider example
from app.application.pipeline.video.builder import build_video_pipeline_via_container
from app.application.pipeline.video.adapter_bundle import VideoPipelineAdapters

from app.infrastructure.adapters.asset_repo import AssetRepo
from app.infrastructure.adapters.video_renderer import VideoRenderer
from app.infrastructure.adapters.uploader import S3Uploader

def get_video_pipeline():
    adapters = VideoPipelineAdapters(
        assets=AssetRepo(),
        renderer=VideoRenderer(),
        uploader=S3Uploader(),
    )
    return build_video_pipeline_via_container(adapters)
```

Note: `transcriber` should be provided via DI. Example factory:

```python
# backend/app/presentation/api/v1/dependencies/media.py
from app.infrastructure.adapters.transcriber import WhisperTranscriber

def get_transcriber():
    return WhisperTranscriber()
```

---

### Step 4 – Presentation Layer

**[Spec]**

- FastAPI endpoints.
- Inject dependencies.
- Use Pydantic for request/response schemas.

Validation & Error Handling

- Validate inputs with Pydantic schemas; prefer constrained types/validators.
- Map domain/application errors to HTTP:
  - `ValueError`/validation → 422/400
  - `LookupError` → 404
  - Permission errors → 403
  - Unexpected → 500 (avoid leaking details)

API Error Schema (suggested)

```json
{
  "type": "string",
  "title": "string",
  "detail": "string",
  "status": 400,
  "instance": "string",
  "errors": { "field": ["message"] }
}
```

**[Action]**

- Define schemas.
- Define dependencies and routers.

**[Sample Code]**

- **Schema:**

```python
# backend/app/presentation/api/v1/schemas/product.py
from pydantic import BaseModel

class CreateProductRequest(BaseModel):
    name: str
    category: str
    price_range: str

class ProductResponse(BaseModel):
    id: int
    name: str
    category: str
    price_range: str
```

- **Dependency:**

```python
# backend/app/presentation/api/v1/dependencies/product.py
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.infrastructure.repositories.product import ProductRepository

def get_product_repo(db: Session = Depends(get_db)):
    return ProductRepository(db)
```

- **Router:**

```python
# backend/app/presentation/api/v1/routers/product.py
from fastapi import APIRouter, Depends
from app.application.use_cases.create_product_use_case import CreateProductUseCase
from app.presentation.api.v1.dependencies.product import get_product_repo
from app.presentation.api.v1.schemas.product import CreateProductRequest, ProductResponse

router = APIRouter(prefix="/products")  # prefix set once at router

@router.post("", response_model=ProductResponse, status_code=201)  # empty path to avoid duplicating prefix
def create_product(request: CreateProductRequest, repo=Depends(get_product_repo)):
    product = CreateProductUseCase(repo).execute(
        request.name, request.category, request.price_range
    )
    return ProductResponse.model_validate(product)
```

---

### Step 5 – Tests (Unit, Integration, E2E)

**[Spec]**

- Structure: `tests/unit/`, `tests/integration/`, `tests/e2e/`.
- Markers: `unit`, `integration`, `e2e`.

**[Test Case Schema]**

```yaml
test_case:
  name: "Create Product"
  type: "unit"
  precondition: "Repo empty"
  steps:
    - action: "call use case with name=Phone"
  expected:
    - "id > 0"
    - "category == electronics"
```

**[Sample Code]**

```python
# backend/tests/unit/application/test_use_case_create_product.py
import pytest
from app.application.use_cases.create_product_use_case import CreateProductUseCase
from app.domain.entities.product import Product

class InMemoryRepo:
    def __init__(self):
        self.items = []
    def create(self, product: Product) -> Product:
        product.id = len(self.items) + 1
        self.items.append(product)
        return product

@pytest.mark.unit
def test_create_product():
    repo = InMemoryRepo()
    uc = CreateProductUseCase(repo)
    p = uc.execute(name="Phone", category="electronics", price_range="high")
    assert p.id == 1
    assert p.category == "electronics"
```

```python
# backend/tests/integration/test_payment_integration.py
import pytest
from httpx import AsyncClient
from app.presentation.main import app

@pytest.mark.asyncio
@pytest.mark.integration
async def test_create_payment_intent():
    class FakeGateway:
        def create_payment_intent(self, amount_cents, currency, metadata):
            return {"id": "pi_test", "status": "requires_payment_method",
                   "amount": amount_cents, "currency": currency, "client_secret": "pi_test_client_secret"}
        def construct_webhook_event(self, payload, sig_header):
            return {"type": "payment_intent.succeeded"}

    # Override dependency for testing
    from app.presentation.api.v1.dependencies.payment import get_stripe_client
    app.dependency_overrides[get_stripe_client] = lambda: FakeGateway()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        res = await ac.post("/payments/intents", json={"amount_cents": 1000, "currency": "usd", "metadata": {}})
        assert res.status_code == 200
        assert res.json()["id"] == "pi_test"

    # Clean up
    app.dependency_overrides.clear()
```

```python
# backend/tests/e2e/test_product_api.py
import pytest
from httpx import AsyncClient
from app.presentation.main import app

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_product_crud_flow():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Create product
        create_data = {"name": "Laptop", "category": "electronics", "price_range": "high"}
        res = await ac.post("/products", json=create_data)
        assert res.status_code == 201
        product = res.json()
        assert product["name"] == "Laptop"
        assert product["id"] > 0
```

#### Testing Commands

```bash
# Run unit tests only (fast)
venv/bin/python -m pytest -q -m "not integration"

# Run integration tests
venv/bin/python -m pytest -q -m integration

# Run all tests with coverage
venv/bin/python -m pytest -q --cov=backend/app --cov-report=html --cov-fail-under=85

# Run specific test file
venv/bin/python -m pytest -q backend/tests/unit/application/test_use_case_create_product.py
```

---

## IV. Config & Env Schema

```yaml
ENV_SCHEMA:
  STRIPE_SECRET_KEY: string
  STRIPE_WEBHOOK_SECRET: string
  STRIPE_DEFAULT_CURRENCY: string
  DATABASE_URL: string
```

---

## V. Logging & Error Convention

- **Info**: `logger.info("UC:CreateProduct started")`
- **Error**: `logger.error("UC:CreateProduct failed", exc_info=True)`
- Never log secrets/PII.

---

## VI. Observability & Resilience

- Logs, metrics, traces → OpenTelemetry/Prometheus.
- Retries + backoff for flaky APIs.
- Idempotency for external calls.
- Circuit breaker optional.

---

## VII. Supplementary (Background Jobs, Observability, Security)

#### Background Jobs (long tasks)

- Use a queue (Celery/RQ/Arq) under `infrastructure/jobs/` for heavy video/LLM tasks.
- Use Case enqueues a job and returns `run_id`; add endpoints `POST /pipeline/run`, `GET /pipeline/status/{run_id}`, `GET /pipeline/result/{run_id}`.

#### Testing Tips (standard)

- Structure `tests/`: `unit/`, `integration/`, `e2e/`, and `test_output/` (logs, junit, coverage). Markers: `integration`, `slow`, `ai` (declare in `pytest.ini`).
- Run in venv: `venv/bin/python -m pytest -q`. Filter with `-m "not integration"`/`-m integration`. Add `--maxfail=3` if needed.
- API/E2E: use `httpx.AsyncClient` + `ASGITransport`. Override dependencies to mock clients/pipelines.

#### Observability & Reliability

- Step-level logs with latency; add metrics (Prometheus/OpenTelemetry).
- Retry/backoff/circuit breaker around SDKs in Infrastructure; ensure idempotency for expensive operations (hash inputs, cache).

#### Security

- Secrets via env (`core/config.py`), never log PII/keys.
- Strict validation in Presentation with Pydantic; sanitize file paths/URLs; scan uploads if needed.
