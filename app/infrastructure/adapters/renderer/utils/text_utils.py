from __future__ import annotations
from typing import Any, Tuple
import re


def normalize_text(text: str) -> str:
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


def parse_color(col: str, default_rgb: Tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int, float]:
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


def parse_pos(x_val: Any, y_val: Any, vw: int, vh: int) -> tuple[int, int, int]:
    """Map common drawtext-like x/y to ASS coordinates and alignment.
    Returns (px, py, an) where an is ASS alignment (1..9).
    """
    an = 2  # bottom-center by default
    # X parsing
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


def fmt_time(val: float, *, ndigits: int = 3) -> str:
    try:
        d = round(float(val), ndigits)
    except Exception:
        d = float(val)
    # Keep at least one decimal place to satisfy tests expecting '1.0'
    s = f"{d:.{ndigits}f}".rstrip("0")
    if s.endswith("."):
        s = s + "0"
    return s if s else "0.0"
