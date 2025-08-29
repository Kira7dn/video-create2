from __future__ import annotations
from typing import Sequence, Any
import asyncio
import os
from pathlib import Path

from app.core.config import settings
from app.application.interfaces.renderer import IVideoRenderer
from utils.video_utils import ffmpeg_concat_videos
from utils.subprocess_utils import safe_subprocess_run


class FFMpegVideoRenderer(IVideoRenderer):
    # Compatibility flag so pipeline steps can detect and pass background_music safely
    SUPPORTS_BACKGROUND_MUSIC: bool = True
    """Renderer backed by existing VideoProcessor and ffmpeg utilities."""

    def __init__(self, *, temp_dir: str | None = None) -> None:
        self._temp_dir = temp_dir

    async def duration(self, input_path: str) -> float:
        """Probe media duration using ffprobe."""

        def _probe() -> float:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                input_path,
            ]
            result = safe_subprocess_run(cmd, f"Probe duration for {input_path}")
            duration_str = (result.stdout or "").strip()
            if not duration_str:
                raise ValueError("Empty duration output from ffprobe")
            return float(duration_str)

        return await asyncio.to_thread(_probe)

    async def concat_clips(
        self,
        inputs: Sequence[str],
        *,
        output_path: str,
        transition: str | None = None,
        background_music: dict | None = None,
    ) -> str:
        # Build segment dicts for our helper
        segments = []
        for idx, p in enumerate(inputs):
            segments.append({"id": f"seg_{idx}", "path": p})
        temp_dir = self._temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        # Transition param currently ignored; utils handles plain concat + optional BGM
        await asyncio.to_thread(
            ffmpeg_concat_videos,
            segments,
            output_path,
            temp_dir=temp_dir,
            background_music=background_music,
        )
        return output_path

    async def render_segment(
        self,
        segment: dict,
        *,
        output_path: str,
        temp_dir: str,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        raise NotImplementedError(
            "Infra renderer must not know about segment; use render_with_plan() from Application layer"
        )

    # Primitive operations (used by Application layer)
    async def transcode_video(
        self,
        input_path: str,
        *,
        output_path: str,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass

        threads = str(settings.ffmpeg_threads or 1)
        vcodec = getattr(settings, "video_default_codec", "libx264")
        acodec = getattr(settings, "video_default_audio_codec", "aac")
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        cmd = [
            settings.ffmpeg_binary_path,
            "-y",
            "-threads",
            threads,
            "-i",
            input_path,
            "-r",
            str(fps),
            "-vf",
            vf,
            "-c:v",
            vcodec,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            acodec,
            output_path,
        ]
        await asyncio.to_thread(safe_subprocess_run, cmd, "Transcode video")
        return output_path

    async def still_from_image(
        self,
        image_path: str,
        *,
        output_path: str,
        duration: float,
        width: int,
        height: int,
        fps: int,
        voice_path: str | None = None,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass

        threads = str(settings.ffmpeg_threads or 1)
        vcodec = getattr(settings, "video_default_codec", "libx264")
        acodec = getattr(settings, "video_default_audio_codec", "aac")
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        if voice_path and os.path.exists(voice_path):
            cmd = [
                settings.ffmpeg_binary_path,
                "-y",
                "-threads",
                threads,
                "-loop",
                "1",
                "-r",
                str(fps),
                "-i",
                image_path,
                "-i",
                voice_path,
                "-t",
                str(duration),
                "-vf",
                vf,
                "-shortest",
                "-c:v",
                vcodec,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                acodec,
                output_path,
            ]
        else:
            cmd = [
                settings.ffmpeg_binary_path,
                "-y",
                "-threads",
                threads,
                "-loop",
                "1",
                "-r",
                str(fps),
                "-i",
                image_path,
                "-t",
                str(duration),
                "-vf",
                vf,
                "-c:v",
                vcodec,
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        await asyncio.to_thread(safe_subprocess_run, cmd, "Create still video")
        return output_path

    async def placeholder(
        self,
        *,
        output_path: str,
        duration: float,
        width: int,
        height: int,
        fps: int,
        color: str = "black",
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass

        threads = str(settings.ffmpeg_threads or 1)
        cmd = [
            settings.ffmpeg_binary_path,
            "-y",
            "-threads",
            threads,
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={width}x{height}:d={duration}",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        await asyncio.to_thread(safe_subprocess_run, cmd, "Create placeholder video")
        return output_path

    async def render_with_plan(
        self,
        plan: dict,
        *,
        output_path: str,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        threads = str(getattr(settings, "ffmpeg_threads", 1))
        vcodec = getattr(settings, "video_default_codec", "libx264")
        acodec = getattr(settings, "video_default_audio_codec", "aac")

        input_type = plan.get("input_type", "image")
        input_media = plan.get("video_input")
        voice_path = plan.get("audio_input")
        ops = plan.get("ops") or []

        # Translate ops -> ffmpeg filters
        video_filters: list[str] = []
        audio_filters: list[str] = []

        def add_fade(target: str, spec: dict):
            if not isinstance(spec, dict):
                return
            if "in" in spec and isinstance(spec["in"], dict):
                st = float(spec["in"].get("start", 0))
                d = float(spec["in"].get("duration", 0))
                if target == "video":
                    video_filters.append(f"fade=t=in:st={st}:d={d}")
                else:
                    audio_filters.append(f"afade=t=in:st={st}:d={d}")
            if "out" in spec and isinstance(spec["out"], dict):
                st = float(spec["out"].get("start", 0))
                d = float(spec["out"].get("duration", 0))
                if target == "video":
                    video_filters.append(f"fade=t=out:st={st}:d={d}")
                else:
                    audio_filters.append(f"afade=t=out:st={st}:d={d}")

        # Pass 1: apply audio_delay first (before any afade)
        for op in ops:
            if not isinstance(op, dict):
                continue
            if op.get("op") == "audio_delay":
                ms = int(op.get("milliseconds", 0))
                if ms > 0:
                    # duplicate for stereo channels
                    audio_filters.append(f"adelay={ms}|{ms}")

        # Pass 2: process remaining ops in order (excluding audio_delay)
        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = op.get("op")
            if kind == "audio_delay":
                continue
            if kind == "pixel_format":
                fmt = op.get("format") or "yuv420p"
                video_filters.append(f"format={fmt}")
            elif kind == "scale":
                w = int(op.get("width", width))
                h = int(op.get("height", height))
                video_filters.append(f"scale={w}:{h}")
            elif kind == "fit_canvas":
                # Abstract contain fit with optional background pad
                w = int(op.get("width", width))
                h = int(op.get("height", height))
                mode = op.get("mode", "contain")
                bg = op.get("background", "black")
                if mode == "contain":
                    video_filters.append(
                        f"scale={w}:{h}:force_original_aspect_ratio=decrease"
                    )
                    video_filters.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg}")
                elif mode == "cover":
                    video_filters.append(
                        f"scale={w}:{h}:force_original_aspect_ratio=increase"
                    )
                    video_filters.append(f"crop={w}:{h}:(in_w-out_w)/2:(in_h-out_h)/2")
            elif kind == "audio_pad":
                audio_filters.append("apad")
            elif kind == "fade":
                target = op.get("target", "video")
                add_fade(target, op)
            elif kind == "draw_text":
                text = str(op.get("text", "")).replace(":", "\\:").replace("'", "\\'")
                start = float(op.get("start", 0))
                dur = float(op.get("duration", 0))
                end = start + dur if dur > 0 else start
                pos = op.get("position", {}) or {}
                x = pos.get("x", "(w-text_w)/2")
                y = pos.get("y", "(h-text_h)/2")
                # Back-compat: accept either 'font' or abstract 'style'
                style = op.get("style") or {}
                font = op.get("font") or {}
                fontsize = int(
                    (style.get("size") if isinstance(style, dict) else None)
                    or font.get("size", 42)
                )
                color = (
                    style.get("color") if isinstance(style, dict) else None
                ) or font.get("color", "white")
                family = (
                    style.get("family") if isinstance(style, dict) else None
                ) or "default"
                # Map family to a concrete font file (infra responsibility)
                family_map = {
                    "default": "fonts/Roboto-Black.ttf",
                    "sans": "fonts/Roboto-Black.ttf",
                    "serif": "fonts/Roboto-Black.ttf",
                }
                fontfile = font.get("file") or family_map.get(
                    family, family_map["default"]
                )
                box = op.get("box", {}) or {}
                box_enabled = 1 if box.get("enabled", True) else 0
                boxcolor = box.get("color", "black@0.4")
                enable = f"between(t,{start},{end})"
                video_filters.append(
                    "drawtext="
                    f"fontfile='{fontfile}':text='{text}':"
                    f"fontcolor={color}:fontsize={fontsize}:x={x}:y={y}:"
                    f"box={box_enabled}:boxcolor={boxcolor}:"
                    f"enable='{enable}'"
                )

        total_duration = float(plan.get("duration", 4.0))
        if input_type == "video":
            cmd = [
                settings.ffmpeg_binary_path,
                "-y",
                "-threads",
                threads,
            ]
            if voice_path and os.path.exists(voice_path):
                cmd += ["-i", voice_path]
            else:
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                ]
            cmd += [
                "-i",
                input_media,
                "-vf",
                ",".join(video_filters) if video_filters else "null",
                "-af",
                ",".join(audio_filters) if audio_filters else "anull",
                "-t",
                str(total_duration),
                "-map",
                "1:v",
                "-map",
                "0:a",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-c:v",
                vcodec,
                "-c:a",
                acodec,
                "-b:a",
                "192k",
                output_path,
            ]
        else:
            cmd = [
                settings.ffmpeg_binary_path,
                "-y",
                "-threads",
                threads,
                "-loop",
                "1",
                "-i",
                input_media,
            ]
            if voice_path and os.path.exists(voice_path):
                cmd += ["-i", voice_path]
            else:
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                ]
            cmd += [
                "-vf",
                ",".join(video_filters) if video_filters else "null",
                "-af",
                ",".join(audio_filters) if audio_filters else "anull",
                "-t",
                str(total_duration),
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-c:v",
                vcodec,
                "-c:a",
                acodec,
                "-b:a",
                "192k",
                output_path,
            ]

        await asyncio.to_thread(safe_subprocess_run, cmd, "Render with plan")
        return output_path

    # Compatibility helpers for legacy media-processing method names
    async def get_media_duration(self, source_path: str) -> float:
        """Map to renderer duration method."""
        return await self.duration(source_path)

    async def combine_media_sequence(
        self, sources: Sequence[str], *, target_path: str, blend_mode: str | None = None
    ) -> str:
        """Map to renderer concat method."""
        if not target_path.endswith((".mp4", ".mov", ".avi")):
            target_path = f"{target_path}.mp4"
        return await self.concat_clips(
            sources, output_path=target_path, transition=blend_mode
        )

    async def transform_media(
        self,
        source_path: str,
        *,
        target_path: str,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        """Map to renderer transcode method."""
        if not target_path.endswith((".mp4", ".mov", ".avi")):
            target_path = f"{target_path}.mp4"
        return await self.transcode_video(
            source_path,
            output_path=target_path,
            width=canvas_width,
            height=canvas_height,
            fps=frame_rate,
        )

    async def create_from_static(
        self,
        source_path: str,
        *,
        target_path: str,
        duration: float,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
        audio_source: str | None = None,
    ) -> str:
        """Map to renderer still_from_image method."""
        if not target_path.endswith((".mp4", ".mov", ".avi")):
            target_path = f"{target_path}.mp4"
        return await self.still_from_image(
            source_path,
            output_path=target_path,
            duration=duration,
            width=canvas_width,
            height=canvas_height,
            fps=frame_rate,
            voice_path=audio_source,
        )

    async def create_placeholder(
        self,
        *,
        target_path: str,
        duration: float,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
        fill_color: str = "black",
    ) -> str:
        """Map to renderer placeholder method."""
        if not target_path.endswith((".mp4", ".mov", ".avi")):
            target_path = f"{target_path}.mp4"
        return await self.placeholder(
            output_path=target_path,
            duration=duration,
            width=canvas_width,
            height=canvas_height,
            fps=frame_rate,
            color=fill_color,
        )

    async def process_with_specification(
        self,
        specification: dict[str, Any],
        seg_id: str,
        *,
        canvas_width: int,
        canvas_height: int,
        frame_rate: int,
    ) -> str:
        """Process media based on abstract specification."""
        # Handle temp_dir and file path construction internally
        work_dir = self._temp_dir
        full_target_path = str(Path(work_dir) / seg_id / "segment_video.mp4")
        # Translate abstract specification to renderer plan
        plan = self._translate_specification_to_plan(
            specification, canvas_width, canvas_height
        )

        return await self.render_with_plan(
            plan,
            output_path=full_target_path,
            width=canvas_width,
            height=canvas_height,
            fps=frame_rate,
        )

    def _translate_specification_to_plan(
        self, specification: dict[str, Any], canvas_width: int, canvas_height: int
    ) -> dict:
        """Translate abstract specification to concrete renderer plan."""
        source_type = specification.get("source_type", "static")
        primary_source = specification.get("primary_source")
        audio_source = specification.get("audio_source")
        transformations = specification.get("transformations", [])
        spec_duration = specification.get("duration", 4.0)

        # Map source type
        input_type = "video" if source_type == "dynamic" else "image"

        # Build renderer ops from abstract transformations
        ops: list[dict] = []

        # Always ensure proper pixel format for video output
        ops.append({"op": "pixel_format", "format": "yuv420p"})

        for transform in transformations:
            if not isinstance(transform, dict):
                continue

            transform_type = transform.get("type")

            if transform_type == "canvas_fit":
                # Map abstract canvas_fit to concrete scale operation
                fit_mode = transform.get("fit_mode", "contain")
                fill_color = transform.get("fill_color", "black")
                ops.append(
                    {
                        "op": "fit_canvas",
                        "width": canvas_width,
                        "height": canvas_height,
                        "mode": fit_mode,
                        "background": fill_color,
                    }
                )
            elif transform_type == "audio_normalize":
                # Map to audio padding
                ops.append({"op": "audio_pad"})
            elif transform_type == "audio_delay":
                # Pass through to renderer as an op with milliseconds
                ms = int(transform.get("milliseconds", 0) or 0)
                if ms > 0:
                    ops.append({"op": "audio_delay", "milliseconds": ms})
            elif transform_type == "transition":
                # Map abstract transition to concrete fade
                target = "video" if transform.get("target") == "visual" else "audio"
                direction = transform.get("direction", "in")
                start = float(transform.get("start", 0))
                dur = float(transform.get("duration", 0))

                fade_spec = {direction: {"start": start, "duration": dur}}
                ops.append({"op": "fade", "target": target, **fade_spec})
            elif transform_type == "text_overlay":
                # Map abstract text overlay to concrete draw_text
                content = transform.get("content", "")
                timing = transform.get("timing", {})
                appearance = transform.get("appearance", {})
                layout = transform.get("layout", {})
                background = transform.get("background", {})

                ops.append(
                    {
                        "op": "draw_text",
                        "text": content,
                        "start": float(timing.get("start", 0)),
                        "duration": float(timing.get("duration", 0)),
                        "style": {
                            "size": int(appearance.get("size", 42)),
                            "color": appearance.get("color", "white"),
                            "family": appearance.get("typeface", "default"),
                        },
                        "position": {
                            "x": layout.get("x", "(w-text_w)/2"),
                            # Default to bottom placement with small margin
                            "y": layout.get("y", "h-text_h-20"),
                        },
                        "box": {
                            "enabled": background.get("enabled", True),
                            "color": background.get("color", "black@0.4"),
                        },
                    }
                )

        return {
            "input_type": input_type,
            "video_input": primary_source,
            "audio_input": audio_source,
            "loop_image": input_type == "image",
            "ops": ops,
            "duration": float(spec_duration),
        }
