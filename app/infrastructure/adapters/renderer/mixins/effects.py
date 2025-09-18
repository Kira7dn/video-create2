from typing import List, Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class EffectsMixin:
    """Mixin for handling video/audio effects."""

    def _process_effects(
        self, ops: List[Dict[str, Any]]
    ) -> tuple[List[str], List[str]]:
        """Process video and audio effects."""
        video_filters = []
        audio_filters = []

        for op in ops:
            op_type = op.get("op")

            if op_type == "scale":
                self._handle_scale(op, video_filters)
            elif op_type == "fade":
                self._handle_fade(op, video_filters, audio_filters)
            elif op_type == "pixel_format":
                self._handle_pixel_format(op, video_filters)
            elif op_type == "fit_canvas":
                self._handle_fit_canvas(op, video_filters)
            elif op_type == "audio_pad":
                audio_filters.append("apad")
            elif op_type == "audio_delay":
                self._handle_audio_delay(op, audio_filters)
            elif op_type == "audio_normalize":
                self._handle_audio_normalize(op, audio_filters)

        return video_filters, audio_filters

    def _handle_scale(self, op: Dict[str, Any], video_filters: List[str]) -> None:
        """Handle scale operation."""
        w = op.get("width")
        h = op.get("height")
        if w and h:
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
            video_filters.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black")

    def _handle_fade(
        self, op: Dict[str, Any], video_filters: List[str], audio_filters: List[str]
    ) -> None:
        """Handle fade effects."""
        target = op.get("target", "video")
        # Support structured {in:{start,duration}, out:{start,duration}} like origin
        if "in" in op and isinstance(op["in"], dict):
            st = float(op["in"].get("start", 0))
            d = float(op["in"].get("duration", 0))
            if target == "video":
                video_filters.append(f"fade=t=in:st={st}:d={d}")
            else:
                audio_filters.append(f"afade=t=in:st={st}:d={d}")
        if "out" in op and isinstance(op["out"], dict):
            st = float(op["out"].get("start", 0))
            d = float(op["out"].get("duration", 0))
            if target == "video":
                video_filters.append(f"fade=t=out:st={st}:d={d}")
            else:
                audio_filters.append(f"afade=t=out:st={st}:d={d}")

    def _handle_pixel_format(
        self, op: Dict[str, Any], video_filters: List[str]
    ) -> None:
        fmt = op.get("format") or "yuv420p"
        video_filters.append(f"format={fmt}")

    def _handle_fit_canvas(self, op: Dict[str, Any], video_filters: List[str]) -> None:
        w = int(op.get("width", 1920))
        h = int(op.get("height", 1080))
        mode = op.get("mode", "contain")
        bg = op.get("background", "black")
        if mode == "contain":
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
            video_filters.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg}")
        elif mode == "cover":
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=increase")
            video_filters.append(f"crop={w}:{h}:(in_w-out_w)/2:(in_h-out_h)/2")

    def _handle_audio_delay(self, op: Dict[str, Any], audio_filters: List[str]) -> None:
        ms = int(op.get("milliseconds", 0) or 0)
        if ms > 0:
            audio_filters.append(f"adelay={ms}|{ms}")

    def _handle_audio_normalize(
        self, _op: Dict[str, Any], audio_filters: List[str]
    ) -> None:
        if getattr(settings, "segment_audio_normalize_enabled", True):
            target_i = float(getattr(settings, "segment_audio_target_i", -16.0))
            target_tp = float(getattr(settings, "segment_audio_target_tp", -1.5))
            target_lra = float(getattr(settings, "segment_audio_target_lra", 11.0))
            loudnorm = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=summary"
            audio_filters.append(loudnorm)
            if getattr(settings, "segment_audio_limiter_enabled", True):
                audio_filters.append("alimiter=limit=0.95")
