from __future__ import annotations

from abc import ABC, abstractmethod

from app.application.pipeline.base import PipelineContext


class IPipelineStep(ABC):
    """Interface cho một bước trong pipeline (phù hợp Clean Architecture).

    Đây là lớp bridge tương thích với Step của pipeline hiện có.
    Tất cả triển khai nên dùng async để phù hợp hệ thống hiện tại.
    """

    @abstractmethod
    async def run(
        self, context: PipelineContext
    ) -> None:  # pragma: no cover - abstract
        ...
