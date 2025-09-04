#!/usr/bin/env python3
"""
Analyze loudness (LUFS) of media files using ffmpeg loudnorm.

Usage:
  .venv/bin/python scripts/analyze_loudness.py <root_dir> [--I -16] [--TP -1.5] [--LRA 11]

Notes:
- Requires ffmpeg in PATH.
- Parses the final JSON emitted by loudnorm and prints: input_i, input_tp, input_lra, path.
- Exit code 0 even if some files fail to parse (they will be marked accordingly).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Iterable


def find_segment_videos(root: Path) -> Iterable[Path]:
    # Search recursively for files named exactly 'segment_video.mp4'
    for p in root.rglob("segment_video.mp4"):
        if p.is_file():
            yield p


def analyze_loudness(path: Path, I: float, TP: float, LRA: float) -> Optional[dict]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-af",
        f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        text = (res.stderr or "") + (res.stdout or "")
        matches = list(re.finditer(r"\{[\s\S]*?\}", text))
        if not matches:
            return None
        data = json.loads(matches[-1].group(0))
        return {
            "input_i": data.get("input_i"),
            "input_tp": data.get("input_tp"),
            "input_lra": data.get("input_lra"),
            "path": str(path),
        }
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze loudness (LUFS) using ffmpeg loudnorm")
    parser.add_argument("root_dir", type=str, help="Root directory to search")
    parser.add_argument("--I", type=float, default=-16.0, help="Target integrated loudness (LUFS)")
    parser.add_argument("--TP", type=float, default=-1.5, help="Target true peak (dBTP)")
    parser.add_argument("--LRA", type=float, default=11.0, help="Target loudness range (LU)")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.exists():
        print(f"Error: root_dir not found: {root}")
        return 1

    files = list(find_segment_videos(root))
    if not files:
        print("No files matched.")
        return 0

    print("input_i(LUFS),input_tp(dBTP),input_lra(LU),path")
    for f in files:
        m = analyze_loudness(f, args.I, args.TP, args.LRA)
        if m is None:
            print(f",,,{f}")
        else:
            print(f"{m['input_i']},{m['input_tp']},{m['input_lra']},{m['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
