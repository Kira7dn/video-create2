from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from .text_utils import parse_color


def _ass_time_str(seconds_str: str) -> str:
    try:
        t = float(seconds_str)
    except Exception:
        t = 0.0
    hh = int(t // 3600)
    mm = int((t % 3600) // 60)
    ss = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"


def build_ass_script(
    ass_events: List[Dict], width: int, height: int, ass_path: Path
) -> Path:
    """Build and write an ASS subtitle script for overlay text.

    - ass_events: list items like {text, start, end, fontsize, color, box_enabled, boxcolor, pos_x, pos_y, align}
    - width/height: video canvas size
    - ass_path: output file path to write
    """
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
        txt = ev.get("text") or ""
        # Escape ASS special braces and newlines
        txt = txt.replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        st = _ass_time_str(str(ev.get("start", "0")))
        en = _ass_time_str(str(ev.get("end", "0")))
        fs = int(ev.get("fontsize", 32))
        pr_r, pr_g, pr_b, _ = parse_color(
            str(ev.get("color", "white")), (255, 255, 255)
        )
        primary = f"&H00{pr_b:02X}{pr_g:02X}{pr_r:02X}"
        be = int(ev.get("box_enabled", 0))
        br, bg, bb, bop = parse_color(str(ev.get("boxcolor", "black@0.4")), (0, 0, 0))
        back_alpha = max(0, min(255, int(round((1.0 - bop) * 255)))) if be else 255
        back = f"&H{back_alpha:02X}{bb:02X}{bg:02X}{br:02X}"
        px = int(ev.get("pos_x", width // 2))
        py = int(ev.get("pos_y", int(height * 0.92)))
        an = int(ev.get("align", 2))
        if be:
            style_name = "DefaultBox"
            override = f"{{\\an{an}\\pos({px},{py})\\fs{fs}\\1c{primary}\\4c{back}\\bord0\\shad0}}"
        else:
            style_name = "DefaultOutline"
            outline_color = "&H00000000"
            override = f"{{\\an{an}\\pos({px},{py})\\fs{fs}\\1c{primary}\\3c{outline_color}\\bord2\\shad1}}"
        ass_lines.append(f"Dialogue: 0,{st},{en},{style_name},,0,0,80,,{override}{txt}")

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text("\n".join(ass_lines), encoding="utf-8")
    return ass_path
