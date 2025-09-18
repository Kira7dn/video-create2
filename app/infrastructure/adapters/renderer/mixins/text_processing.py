from typing import List, Dict, Any
import logging

from app.infrastructure.adapters.renderer.utils import (
    normalize_text,
    parse_pos,
)

logger = logging.getLogger(__name__)


class TextProcessingMixin:
    """Mixin for handling text processing operations."""

    def _process_text_operations(
        self, ops: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """Process text drawing operations."""

        ass_events = []

        for op in ops:
            if op.get("op") != "draw_text":
                continue

            try:
                # Process text
                raw_text = str(op.get("text", ""))
                text = normalize_text(raw_text)

                # Get styling and positioning
                style = op.get("style", {})
                position = op.get("position", {})
                box = op.get("box", {})

                # Process timing
                start = float(op.get("start", 0))
                duration = float(op.get("duration", 5))
                end = start + duration

                # Process font and colors
                font_size = int(style.get("size", 36))
                font_color = style.get("color", "white")

                # Process position
                x = position.get("x", "(w-text_w)/2")
                y = position.get("y", "h-text_h-36")

                # Process text box
                box_enabled = box.get("enabled", True)
                boxcolor = box.get("color", "black@0.4")

                # Compute position and alignment using centralized utils
                try:
                    px, py, an = parse_pos(x, y, width, height)
                except Exception as e:
                    logger.warning(f"Error parsing position: {e}")
                    px, py, an = width // 2, height - 36, 2

                ass_events.append(
                    {
                        "text": text,
                        "start": start,
                        "end": end,
                        "fontsize": font_size,
                        "color": font_color,
                        "box_enabled": box_enabled,
                        "boxcolor": boxcolor,
                        "pos_x": px,
                        "pos_y": py,
                        "align": an,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing text operation: {e}", exc_info=True)

        return ass_events
