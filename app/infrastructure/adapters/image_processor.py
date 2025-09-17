from __future__ import annotations

import asyncio
from typing import Sequence
from pathlib import Path

from app.application.interfaces.image_processor import IImageProcessor
from utils.image_utils import process_image


class ImageProcessor(IImageProcessor):
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir

    async def process(
        self,
        image_path: str,
        *,
        target_width: int,
        target_height: int,
        seg_id: str | None = None,
        smart_pad_color: bool = True,
        pad_color_method: str = "average_edge",
        mode: str = "contain",
        auto_enhance: bool = True,
        enhance_brightness: bool = True,
        enhance_contrast: bool = True,
        enhance_saturation: bool = True,
    ) -> Sequence[str]:
        def _run():
            # Compute output directory; per-segment folder if seg_id provided
            if seg_id:
                out_dir = Path(self.temp_dir) / seg_id
            else:
                out_dir = Path(self.temp_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            return process_image(
                image_paths=image_path,
                target_size=(target_width, target_height),
                smart_pad_color=smart_pad_color,
                pad_color_method=pad_color_method,
                mode=mode,
                auto_enhance=auto_enhance,
                enhance_brightness=enhance_brightness,
                enhance_contrast=enhance_contrast,
                enhance_saturation=enhance_saturation,
                output_dir=str(out_dir),
            )

        return await asyncio.to_thread(_run)
