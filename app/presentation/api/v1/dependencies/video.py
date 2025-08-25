from app.application.use_cases.video_create import CreateVideoUseCase
from app.infrastructure.adapters.bundles.video import get_video_adapter_bundle


def get_create_video_use_case(*, temp_dir: str | None = None) -> CreateVideoUseCase:
    """Compose the CreateVideoUseCase at Presentation layer using adapter providers."""
    adapters = get_video_adapter_bundle(temp_dir=temp_dir)
    return CreateVideoUseCase(adapters)
