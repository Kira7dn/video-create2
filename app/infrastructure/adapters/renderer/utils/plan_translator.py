from __future__ import annotations
from typing import Any, Dict, List


def translate_specification_to_plan(
    specification: Dict[str, Any], canvas_width: int, canvas_height: int
) -> Dict[str, Any]:
    """Translate abstract specification to concrete renderer plan.

    Mirrors the former `_translate_specification_to_plan` method but as a standalone
    pure function so it can be reused and unit-tested independently.
    """
    source_type = specification.get("source_type", "static")
    primary_source = specification.get("primary_source")
    audio_source = specification.get("audio_source")
    transformations = specification.get("transformations", [])
    spec_duration = specification.get("duration", 4.0)

    # Map source type
    input_type = "video" if source_type == "dynamic" else "image"

    # Build renderer ops from abstract transformations
    ops: List[Dict] = []

    # Always ensure proper pixel format for video output
    ops.append({"op": "pixel_format", "format": "yuv420p"})

    # Helper: wrap long text into multiple lines using a simple width heuristic
    def _wrap_text_to_width(
        text: str, fontsize: int, max_width_pct: float = 0.9
    ) -> str:
        """Wrap text into multiple lines so the longest line width ~= canvas_width * max_width_pct.

        Heuristic: avg_char_width ≈ fontsize * 0.6 (works reasonably for Roboto Black).
        We return lines joined with '\n' so drawtext renders multi-line without changing fontsize.
        """
        try:
            t = (text or "").strip()
            if not t:
                return ""
            # Avoid pathological small/large values
            fs = max(8, int(fontsize or 42))
            avg_char_w = max(4.0, fs * 0.6)
            target_w = max(50.0, canvas_width * float(max_width_pct))
            max_chars = max(8, int(target_w / avg_char_w))

            words = t.split()
            if not words:
                return t
            lines: list[str] = []
            cur: list[str] = []
            cur_len = 0
            for w in words:
                add_len = len(w) + (1 if cur else 0)
                if cur_len + add_len <= max_chars:
                    cur.append(w)
                    cur_len += add_len
                else:
                    if cur:
                        lines.append(" ".join(cur))
                    cur = [w]
                    cur_len = len(w)
            if cur:
                lines.append(" ".join(cur))
            # Join with \n for ffmpeg drawtext (we pass a literal backslash-n)
            return "\\n".join(lines)
        except Exception:
            # On any error, return original text
            return text

    for transform in transformations:
        if not isinstance(transform, dict):
            continue

        transform_type = transform.get("type")

        # Dispatch table simplifies transform -> ops mapping
        def _build_canvas_fit(t: dict) -> None:
            fit_mode = t.get("fit_mode", "contain")
            fill_color = t.get("fill_color", "black")
            ops.append(
                {
                    "op": "fit_canvas",
                    "width": canvas_width,
                    "height": canvas_height,
                    "mode": fit_mode,
                    "background": fill_color,
                }
            )

        def _build_audio_normalize(_: dict) -> None:
            ops.append({"op": "audio_normalize"})

        def _build_audio_delay(_: dict) -> None:
            ms = int(_.get("milliseconds", 0) or 0)
            if ms > 0:
                # Ensure audio delay is applied BEFORE fade/transition by giving higher priority (lower value)
                ops.append({"op": "audio_delay", "milliseconds": ms, "priority": -100})

        def _build_transition(t: dict) -> None:
            target = "video" if t.get("target") == "visual" else "audio"
            direction = t.get("direction", "in")
            start = float(t.get("start", 0))
            dur = float(t.get("duration", 0))
            fade_spec = {direction: {"start": start, "duration": dur}}
            # Keep default priority (0) so fades come AFTER delays
            ops.append({"op": "fade", "target": target, **fade_spec, "priority": 0})

        def _build_text_overlay(t: dict) -> None:
            content = t.get("content", "")
            timing = t.get("timing", {})
            appearance = t.get("appearance", {})
            layout = t.get("layout", {})
            background = t.get("background", {})
            fontsize_for_wrap = int(appearance.get("size", 42))
            max_width_pct = float(layout.get("max_width_pct", 0.9))
            wrapped = _wrap_text_to_width(
                str(content), fontsize_for_wrap, max_width_pct
            )
            ops.append(
                {
                    "op": "draw_text",
                    "text": wrapped,
                    "start": float(timing.get("start", 0)),
                    "duration": float(timing.get("duration", 0)),
                    "style": {
                        "size": int(appearance.get("size", 42)),
                        "color": appearance.get("color", "white"),
                        "family": appearance.get("typeface", "default"),
                    },
                    "position": {
                        "x": layout.get("x", "(w-text_w)/2"),
                        "y": layout.get("y", "h-text_h-20"),
                    },
                    "box": {
                        "enabled": background.get("enabled", True),
                        "color": background.get("color", "black@0.4"),
                    },
                }
            )

        transform_builders = {
            "canvas_fit": _build_canvas_fit,
            "audio_normalize": _build_audio_normalize,
            "audio_delay": _build_audio_delay,
            "transition": _build_transition,
            "text_overlay": _build_text_overlay,
        }

        handler = transform_builders.get(transform_type)
        if handler:
            handler(transform)

    return {
        "input_type": input_type,
        "video_input": primary_source,
        "audio_input": audio_source,
        "loop_image": input_type == "image",
        "ops": ops,
        "duration": float(spec_duration),
    }
