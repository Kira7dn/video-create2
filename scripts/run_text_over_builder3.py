#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
"""CLI runner for TextOverBuilder3.

Ensures project root is on sys.path before importing from the "app" package.
"""

# Ensure project root is on sys.path so 'app' is importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.adapters.text_over_builder3 import TextOverBuilder3


def load_words(words_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(words_path.read_text(encoding="utf-8"))
    words = data.get("words")
    if not isinstance(words, list):
        raise ValueError("Invalid words.json: missing 'words' list")
    return words


def load_lines(lines_path: Path) -> List[str]:
    return [ln.strip() for ln in lines_path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TextOverBuilder3 to align transcript lines with words.json and print JSON to stdout",
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing words.json and transcript_lines.txt",
    )
    parser.add_argument(
        "--words",
        required=False,
        help="Path to words.json produced by Gentle",
    )
    parser.add_argument(
        "--lines",
        required=False,
        help="Path to transcript_lines.txt (one line per caption)",
    )
    # No output writing options: script will only print to stdout

    args = parser.parse_args()

    # Resolve inputs
    input_dir = Path(args.input_dir) if args.input_dir else None
    if input_dir is not None:
        words_path = input_dir / "words.json"
        lines_path = input_dir / "transcript_lines.txt"
    else:
        if not args.words or not args.lines:
            raise SystemExit(
                "Either --input-dir or both --words and --lines must be provided"
            )
        words_path = Path(args.words)
        lines_path = Path(args.lines)

    if not words_path.exists():
        raise FileNotFoundError(f"words.json not found: {words_path}")
    if not lines_path.exists():
        raise FileNotFoundError(f"transcript_lines.txt not found: {lines_path}")

    words = load_words(words_path)
    chunks = load_lines(lines_path)

    # Build and print only (no file writes)
    # temp_dir is still required by builder's constructor but won't be used since text_over_id=None
    builder = TextOverBuilder3(temp_dir=".cache")
    results = builder.build(word_items=words, chunks=chunks, text_over_id=None)

    # Print summary then full JSON to stdout
    print(f"Total items: {len(results)}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
