from __future__ import annotations

from typing import Optional

from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces.renderer import IVideoRenderer
from app.core.config import settings


class CreateSegmentClipsStep(BaseStep):
    name = "create_segment_clips"
    required_keys = ["segments"]
    def __init__(self, renderer: IVideoRenderer):
        self.renderer = renderer

    async def _build_specification(
        self,
        segment: dict,
        *,
        canvas_width: int,
        canvas_height: int,
    ) -> dict:
        seg = segment or {}
        video_path = (
            (seg.get("video") or {}).get("local_path")
            if isinstance(seg.get("video"), dict)
            else None
        )
        # Use final image path provided by previous steps (must be in image["local_path"]).
        image_obj = seg.get("image") if isinstance(seg.get("image"), dict) else None
        image_path = None
        if isinstance(image_obj, dict):
            image_path = image_obj.get("local_path")
        voice_over = seg.get("voice_over") or {}
        voice_path = (
            (voice_over or {}).get("local_path")
            if isinstance(voice_over, dict)
            else None
        )

        base_duration = float(
            seg.get(
                "duration", getattr(settings, "video_default_segment_duration", 4.0)
            )
        )
        original_duration = base_duration

        input_type = "image"
        input_media: Optional[str] = None

        if video_path:
            input_type = "video"
            input_media = video_path
            try:
                original_duration = await self.renderer.get_media_duration(video_path)
            except Exception:
                original_duration = base_duration
        else:
            if not image_path:
                raise RuntimeError("Background image not found for segment")
            # No image processing here; use the image path as provided by previous steps
            input_media = image_path
            if voice_path:
                try:
                    voice_len = await self.renderer.get_media_duration(voice_path)
                    original_duration = max(base_duration, float(voice_len))
                except Exception:
                    original_duration = base_duration
            else:
                original_duration = base_duration

        transition_in = seg.get("transition_in") or {}
        transition_out = seg.get("transition_out") or {}
        fade_in_duration = float(
            (transition_in.get("duration") or 0)
            if isinstance(transition_in, dict)
            else 0
        )
        fade_out_duration = float(
            (transition_out.get("duration") or 0)
            if isinstance(transition_out, dict)
            else 0
        )

        # Build abstract transformations
        transformations: list[dict] = []
        # For dynamic content, fit into target canvas (static content was preprocessed earlier)
        if input_type == "video":
            transformations.append(
                {
                    "type": "canvas_fit",
                    "canvas_width": int(canvas_width),
                    "canvas_height": int(canvas_height),
                    "fit_mode": "contain",
                    "fill_color": "black",
                }
            )
        # If we have audio, ensure proper handling
        if voice_path:
            transformations.append({"type": "audio_normalize"})

        if fade_in_duration > 0:
            transformations.append(
                {
                    "type": "transition",
                    "target": "visual",
                    "direction": "in",
                    "start": 0.0,
                    "duration": float(fade_in_duration),
                }
            )
            transformations.append(
                {
                    "type": "transition",
                    "target": "audio",
                    "direction": "in",
                    "start": 0.0,
                    "duration": float(fade_in_duration),
                }
            )

        if fade_out_duration > 0:
            fade_out_start = (
                max(0.0, original_duration - fade_out_duration)
                if input_type == "video"
                else fade_in_duration + original_duration
            )
            transformations.append(
                {
                    "type": "transition",
                    "target": "visual",
                    "direction": "out",
                    "start": float(fade_out_start),
                    "duration": float(fade_out_duration),
                }
            )
            transformations.append(
                {
                    "type": "transition",
                    "target": "audio",
                    "direction": "out",
                    "start": float(fade_out_start),
                    "duration": float(fade_out_duration),
                }
            )

        total_duration = (
            original_duration
            if input_type == "video"
            else fade_in_duration + original_duration + fade_out_duration
        )
        delay = fade_in_duration + (voice_over.get("start_delay", 0) or 0)

        # Text overlays (processor-agnostic)
        text_overlays: list[dict] = []
        text_overs = seg.get("text_over")
        if text_overs:
            if not isinstance(text_overs, list):
                raise RuntimeError("text_over must be an array of objects")
            for item in text_overs:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", ""))
                if not text:
                    continue
                start = float(item.get("start", 0) or 0) + delay
                dur = float(item.get("duration", total_duration))
                end = min(total_duration, start + dur)
                fontsize = int(item.get("font_size", 42))
                color = item.get("color", "white")
                x = item.get("x", "(w-text_w)/2")
                y = item.get("y", "(h-text_h)/2")
                box = item.get("box", True)
                boxcolor = item.get("boxcolor", "black@0.4")
                text_overlays.append(
                    {
                        "type": "text_overlay",
                        "content": text,
                        "timing": {
                            "start": float(start),
                            "duration": float(end - start),
                        },
                        "appearance": {
                            "size": int(fontsize),
                            "color": color,
                            "typeface": item.get("font_family", "default"),
                        },
                        "layout": {"x": x, "y": y},
                        "background": {"enabled": bool(box), "color": boxcolor},
                    }
                )

        if text_overlays:
            transformations.extend(text_overlays)

        specification = {
            "source_type": "dynamic" if input_type == "video" else "static",
            "primary_source": input_media,
            "audio_source": voice_path,
            "repeat_static": input_type == "image",
            "transformations": transformations,
            "duration": float(total_duration),
        }
        return specification

    async def run(self, context: PipelineContext) -> None:  # type: ignore[override]
        segments = context.get("segments") or []
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        clips = []
        for idx, seg in enumerate(segments):
            # Let adapter handle temp_dir and file extensions
            out_path = f"segment_{idx:03d}"
            # Resolve defaults from settings
            canvas_width, canvas_height = settings.video_resolution_tuple
            frame_rate = getattr(settings, "video_default_fps", 30)
            specification = await self._build_specification(
                seg, canvas_width=canvas_width, canvas_height=canvas_height
            )
            clip_out = await self.renderer.process_with_specification(
                specification,
                target_path=out_path,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                frame_rate=frame_rate,
            )

            clip_id = seg.get("id", f"seg_{idx}")
            clips.append({"id": clip_id, "path": clip_out})
        context.set("segment_clips", clips)
        return
