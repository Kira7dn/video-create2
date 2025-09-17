from __future__ import annotations
from typing import Sequence, Any
import asyncio
import os
from pathlib import Path
import logging
import re

from app.core.config import settings
from app.application.interfaces.renderer import IVideoRenderer
from utils.video_utils import ffmpeg_concat_videos
from utils.subprocess_utils import safe_subprocess_run


logger = logging.getLogger(__name__)


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
        used_loudnorm: bool = False
        ass_events: list[dict] = []

        def _normalize_text(text: str) -> str:
            """Normalize quotes/whitespace artifacts in overlay text."""
            try:
                return (
                    text.replace("\\n", "\n")
                    .replace("’", "'")
                    .replace("‘", "'")
                    .replace("“", '"')
                    .replace("”", '"')
                    .replace("\u00a0", " ")
                    .replace("\u200b", " ")
                )
            except Exception:
                return text

        def _parse_color(
            col: str, default_rgb=(255, 255, 255)
        ) -> tuple[int, int, int, float]:
            """Return (R,G,B,opacity) from a color string like 'black@0.4' or '#RRGGBB' or 'white'.
            opacity in [0..1], where 1=opaque. If not specified, assume 1.0.
            """
            if not isinstance(col, str) or not col:
                r, g, b = default_rgb
                return r, g, b, 1.0
            name_alpha = col.split("@", 1)
            name = name_alpha[0].strip().lower()
            try:
                opacity = float(name_alpha[1]) if len(name_alpha) > 1 else 1.0
                if opacity < 0:
                    opacity = 0.0
                if opacity > 1:
                    opacity = 1.0
            except Exception:
                opacity = 1.0
            named = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "yellow": (255, 255, 0),
                "cyan": (0, 255, 255),
                "magenta": (255, 0, 255),
            }
            if name.startswith("#") and len(name) == 7:
                try:
                    r = int(name[1:3], 16)
                    g = int(name[3:5], 16)
                    b = int(name[5:7], 16)
                    return r, g, b, opacity
                except Exception:
                    pass
            if name in named:
                r, g, b = named[name]
            else:
                r, g, b = default_rgb
            return r, g, b, opacity

        def _parse_pos(
            x_val: Any, y_val: Any, vw: int, vh: int
        ) -> tuple[int, int, int]:
            """Map common drawtext-like x/y to ASS coordinates and alignment.
            Returns (px, py, an) where an is ASS alignment (1..9).
            Heuristics:
            - Center X: if contains 'w-text_w' or equals '(w-text_w)/2' -> px = vw//2
            - Bottom Y offset: pattern like 'h-text_h-<k>*h' -> py = vh - int(k*vh)
            - Middle Y: '(h-text_h)/2' -> py = vh//2
            - If numeric, use directly; else defaults to bottom area.
            Default alignment = 2 (bottom-center).
            """
            an = 2  # bottom-center by default
            # X parsing
            px: int
            if isinstance(x_val, (int, float)):
                px = int(x_val)
            elif isinstance(x_val, str):
                xs = x_val.replace(" ", "")
                if "w-text_w" in xs or xs == "(w-text_w)/2":
                    px = vw // 2
                    # keep center alignment horizontally
                else:
                    # Fallback: try numeric literal
                    try:
                        px = int(float(xs))
                    except Exception:
                        px = vw // 2
            else:
                px = vw // 2

            # Y parsing
            py: int
            if isinstance(y_val, (int, float)):
                py = int(y_val)
                # decide vertical alignment based on y
                if py <= vh // 3:
                    an = 8  # top-center
                elif py >= (2 * vh) // 3:
                    an = 2  # bottom-center
                else:
                    an = 5  # middle-center
            elif isinstance(y_val, str):
                ys = y_val.replace(" ", "")
                # h-text_h-<k>*h
                m = re.match(r"h-?text_h-([0-9]*\.?[0-9]+)\*h", ys)
                if m:
                    k = float(m.group(1))
                    py = int(vh - k * vh)
                    an = 2
                elif ys == "(h-text_h)/2":
                    py = vh // 2
                    an = 5
                else:
                    # try numeric
                    try:
                        py = int(float(ys))
                        if py <= vh // 3:
                            an = 8
                        elif py >= (2 * vh) // 3:
                            an = 2
                        else:
                            an = 5
                    except Exception:
                        # default near bottom
                        py = int(vh * 0.92)
                        an = 2
            else:
                py = int(vh * 0.92)
                an = 2

            return px, py, an

        def _fmt_time(val: float, *, ndigits: int = 3) -> str:
            try:
                d = round(float(val), ndigits)
            except Exception:
                d = float(val)
            # Keep at least one decimal place to satisfy tests expecting '1.0'
            s = f"{d:.{ndigits}f}".rstrip("0")
            if s.endswith("."):
                s = s + "0"
            return s if s else "0.0"

        def add_fade(target: str, spec: dict):
            if not isinstance(spec, dict):
                return
            if "in" in spec and isinstance(spec["in"], dict):
                st = float(spec["in"].get("start", 0))
                d = float(spec["in"].get("duration", 0))
                st_s = _fmt_time(st)
                d_s = _fmt_time(d)
                if target == "video":
                    video_filters.append(f"fade=t=in:st={st_s}:d={d_s}")
                else:
                    audio_filters.append(f"afade=t=in:st={st_s}:d={d_s}")
            if "out" in spec and isinstance(spec["out"], dict):
                st = float(spec["out"].get("start", 0))
                d = float(spec["out"].get("duration", 0))
                st_s = _fmt_time(st)
                d_s = _fmt_time(d)
                if target == "video":
                    video_filters.append(f"fade=t=out:st={st_s}:d={d_s}")
                else:
                    audio_filters.append(f"afade=t=out:st={st_s}:d={d_s}")

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
            if kind == "audio_normalize":
                # Normalize segment audio loudness to common target and apply a safety limiter
                # Use EBU R128 targets similar to concat stage, but single-pass for speed.
                # This ensures each segment has consistent perceived loudness before concat.
                if getattr(settings, "segment_audio_normalize_enabled", True):
                    target_i = float(getattr(settings, "segment_audio_target_i", -16.0))
                    target_tp = float(
                        getattr(settings, "segment_audio_target_tp", -1.5)
                    )
                    target_lra = float(
                        getattr(settings, "segment_audio_target_lra", 11.0)
                    )
                    loudnorm = (
                        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:"
                        "print_format=summary"
                    )
                    audio_filters.append(loudnorm)
                    used_loudnorm = True
                    # Add a soft limiter to avoid inter-sample peaks after normalization
                    if getattr(settings, "segment_audio_limiter_enabled", True):
                        audio_filters.append("alimiter=limit=0.95")
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
                # Always collect text as ASS subtitle events (drop drawtext path)
                raw_text = str(op.get("text", ""))
                file_text = _normalize_text(raw_text)
                start = float(op.get("start", 0))
                dur = float(op.get("duration", 0))
                end = start + dur if dur > 0 else start
                start_s = _fmt_time(start)
                end_s = _fmt_time(end + 1e-6)
                pos = op.get("position", {}) or {}
                x = pos.get("x", "(w-text_w)/2")
                y = pos.get("y", "(h-text_h)/2")
                style = op.get("style") or {}
                font = op.get("font") or {}
                fontsize = int(
                    (style.get("size") if isinstance(style, dict) else None)
                    or font.get("size", 42)
                )
                color = (
                    style.get("color") if isinstance(style, dict) else None
                ) or font.get("color", "white")
                box = op.get("box", {}) or {}
                box_enabled = 1 if box.get("enabled", True) else 0
                boxcolor = box.get("color", "black@0.4")

                if "\n" in file_text:
                    lines = file_text.splitlines()
                    for i, line in enumerate(lines):
                        px, py, an = _parse_pos(x, y, width, height)
                        py_i = min(height - 1, py + i * (fontsize + 8))
                        ass_events.append(
                            {
                                "text": line,
                                "start": start_s,
                                "end": end_s,
                                "fontsize": fontsize,
                                "color": color,
                                "box_enabled": box_enabled,
                                "boxcolor": boxcolor,
                                "pos_x": px,
                                "pos_y": py_i,
                                "align": an,
                            }
                        )
                else:
                    px, py, an = _parse_pos(x, y, width, height)
                    ass_events.append(
                        {
                            "text": file_text,
                            "start": start_s,
                            "end": end_s,
                            "fontsize": fontsize,
                            "color": color,
                            "box_enabled": box_enabled,
                            "boxcolor": boxcolor,
                            "pos_x": px,
                            "pos_y": py,
                            "align": an,
                        }
                    )

        # If we have subtitle events, generate an ASS file and use subtitles filter
        if ass_events:
            try:

                def _ass_time(s: str) -> str:
                    # s is already a short decimal string; convert to H:MM:SS.cs
                    try:
                        t = float(s)
                    except Exception:
                        t = 0.0
                    hh = int(t // 3600)
                    mm = int((t % 3600) // 60)
                    ss = int(t % 60)
                    cs = int(round((t - int(t)) * 100))
                    return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"

                ass_lines: list[str] = []
                ass_lines.append("[Script Info]")
                ass_lines.append("ScriptType: v4.00+")
                ass_lines.append(f"PlayResX: {width}")
                ass_lines.append(f"PlayResY: {height}")
                ass_lines.append("ScaledBorderAndShadow: yes")
                ass_lines.append("")
                ass_lines.append("[V4+ Styles]")
                ass_lines.append(
                    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
                )
                # Two base styles:
                # - DefaultOutline: crisp white text with black outline & slight shadow (no box)
                # - DefaultBox: text over semi-transparent black box (no outline/shadow)
                ass_lines.append(
                    "Style: DefaultOutline,Roboto,36,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,80,0"
                )
                ass_lines.append(
                    "Style: DefaultBox,Roboto,36,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,3,0,0,2,40,40,80,0"
                )
                ass_lines.append("")
                ass_lines.append("[Events]")
                ass_lines.append(
                    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
                )
                for ev in ass_events:
                    txt = ev.get("text", "")
                    # Escape ASS special braces and newlines
                    txt = (
                        txt.replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
                    )
                    st = _ass_time(ev.get("start", "0"))
                    en = _ass_time(ev.get("end", "0"))
                    # Build override tags for font size, primary color, and background/outline
                    fs = int(ev.get("fontsize", 32))
                    # Primary text color (ASS expects BBGGRR in &H00BBGGRR)
                    pr = _parse_color(str(ev.get("color", "white")), (255, 255, 255))
                    pr_r, pr_g, pr_b, _ = pr
                    primary = f"&H00{pr_b:02X}{pr_g:02X}{pr_r:02X}"
                    # Back colour from box setting; if box disabled, use fully transparent
                    be = int(ev.get("box_enabled", 0))
                    br, bg, bb, bop = _parse_color(
                        str(ev.get("boxcolor", "black@0.4")), (0, 0, 0)
                    )
                    # ASS alpha: 00 opaque, FF fully transparent. 'opacity' ~ how opaque we want.
                    back_alpha = (
                        max(0, min(255, int(round((1.0 - bop) * 255)))) if be else 255
                    )
                    back = f"&H{back_alpha:02X}{bb:02X}{bg:02X}{br:02X}"
                    # BorderStyle=3 uses BackColour; also ensure no outline/shadow
                    px = int(ev.get("pos_x", width // 2))
                    py = int(ev.get("pos_y", int(height * 0.92)))
                    an = int(ev.get("align", 2))
                    if be:
                        # Box style: use DefaultBox (BorderStyle=3), no outline/shadow; control box via BackColour
                        style_name = "DefaultBox"
                        override = f"{{\\an{an}\\pos({px},{py})\\fs{fs}\\1c{primary}\\4c{back}\\bord0\\shad0}}"
                    else:
                        # Outline style: use DefaultOutline with subtle shadow; ensure outline color black
                        style_name = "DefaultOutline"
                        outline_color = "&H00000000"  # &H00BBGGRR for black
                        override = f"{{\\an{an}\\pos({px},{py})\\fs{fs}\\1c{primary}\\3c{outline_color}\\bord2\\shad1}}"
                    ass_lines.append(
                        f"Dialogue: 0,{st},{en},{style_name},,0,0,80,,{override}{txt}"
                    )

                ass_path = Path(output_path).parent / "overlay.ass"
                ass_path.write_text("\n".join(ass_lines), encoding="utf-8")
                # Prepend subtitles filter; keep other non-text filters intact
                video_filters.append(f"subtitles='{str(ass_path)}'")
            except Exception:
                # If ASS generation fails, proceed without subtitles (may still work for simple cases)
                pass

        # Assemble video/audio filter specs. Prefer filter_script for long graphs.
        vf_spec = ",".join(video_filters) if video_filters else "null"
        af_spec = ",".join(audio_filters) if audio_filters else "anull"
        use_filter_script = False
        vf_script_path: Path | None = None
        try:
            # Heuristic threshold: if filtergraph is long/complex, use -filter_script:v
            if len(vf_spec) > 800 or "subtitles=" in vf_spec:
                vf_script_path = Path(output_path).parent / "vf.txt"
                vf_script_path.write_text(vf_spec, encoding="utf-8")
                use_filter_script = True
                logger.debug(
                    "Using filter_script for video filters at %s (len=%d)",
                    str(vf_script_path),
                    len(vf_spec),
                )
        except Exception as _e:
            # Fallback to inline -vf
            use_filter_script = False
            vf_script_path = None

        total_duration = float(plan.get("duration", 4.0))
        # Common args for filters
        vf_args = (
            ["-filter_script:v", str(vf_script_path)]
            if use_filter_script and vf_script_path
            else ["-vf", vf_spec]
        )
        af_args = ["-af", af_spec]

        def _audio_input_args(vpath: str | None) -> list[str]:
            if vpath and os.path.exists(vpath):
                return ["-i", vpath]
            return [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]

        if input_type == "video":
            cmd = [
                settings.ffmpeg_binary_path,
                "-y",
                "-threads",
                threads,
            ]
            cmd += _audio_input_args(voice_path)
            cmd += [
                "-i",
                input_media,
                *vf_args,
                *af_args,
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
            cmd += _audio_input_args(voice_path)
            cmd += [
                *vf_args,
                *af_args,
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

        result = await asyncio.to_thread(safe_subprocess_run, cmd, "Render with plan")
        # Optionally log loudnorm summary for easier analysis in tests/integration
        if used_loudnorm and getattr(settings, "segment_audio_log_loudnorm", False):
            try:
                stderr = (getattr(result, "stderr", None) or "").strip()
                if stderr:
                    # Keep log size reasonable; ffmpeg prints lots of lines
                    preview = (
                        stderr
                        if len(stderr) < 8000
                        else stderr[:8000] + "... [truncated]"
                    )
                    logger.info("[loudnorm-summary]\n%s", preview)
            except Exception:
                # Do not fail rendering due to logging issues
                pass
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

            def _build_audio_normalize(t: dict) -> None:
                ops.append({"op": "audio_normalize"})

            def _build_audio_delay(t: dict) -> None:
                ms = int(t.get("milliseconds", 0) or 0)
                if ms > 0:
                    ops.append({"op": "audio_delay", "milliseconds": ms})

            def _build_transition(t: dict) -> None:
                target = "video" if t.get("target") == "visual" else "audio"
                direction = t.get("direction", "in")
                start = float(t.get("start", 0))
                dur = float(t.get("duration", 0))
                fade_spec = {direction: {"start": start, "duration": dur}}
                ops.append({"op": "fade", "target": target, **fade_spec})

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

            TRANSFORM_BUILDERS = {
                "canvas_fit": _build_canvas_fit,
                "audio_normalize": _build_audio_normalize,
                "audio_delay": _build_audio_delay,
                "transition": _build_transition,
                "text_overlay": _build_text_overlay,
            }

            handler = TRANSFORM_BUILDERS.get(transform_type)
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
