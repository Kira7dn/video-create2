from app.application.use_cases.video_create import CreateVideoUseCase
from app.infrastructure.adapters.bundles.video import get_video_adapter_bundle
from typing import Callable


def get_create_video_use_case(*, temp_dir: str | None = None) -> CreateVideoUseCase:
    """Compose the CreateVideoUseCase at Presentation layer using adapter providers."""
    adapters = get_video_adapter_bundle(temp_dir=temp_dir)
    return CreateVideoUseCase(adapters)


def get_create_video_use_case_factory() -> Callable[[str | None], CreateVideoUseCase]:
    """Provide a factory that can build CreateVideoUseCase per job with a temp_dir.

    This aligns with Clean/Onion: Presentation composes per-job use cases with
    job-scoped infrastructure concerns (like temp_dir) while Application stays pure.
    """
    def factory(temp_dir: str | None = None) -> CreateVideoUseCase:
        adapters = get_video_adapter_bundle(temp_dir=temp_dir)
        return CreateVideoUseCase(adapters)

    return factory
