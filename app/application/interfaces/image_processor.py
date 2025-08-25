from __future__ import annotations

from typing import Protocol, Sequence


class IImageProcessor(Protocol):
    async def process(
        self,
        image_path: str,
        *,
        target_width: int,
        target_height: int,
        output_dir: str,
        smart_pad_color: bool = True,
        pad_color_method: str = "average_edge",
        auto_enhance: bool = True,
        enhance_brightness: bool = True,
        enhance_contrast: bool = True,
        enhance_saturation: bool = True,
    ) -> Sequence[str]:
        """Return processed image file path(s) ready for rendering canvas.
        Implementations may perform scaling, padding and enhancement.
        """
        ...
