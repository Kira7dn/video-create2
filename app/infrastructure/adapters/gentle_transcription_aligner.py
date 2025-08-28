from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from app.application.interfaces import ITranscriptionAligner
from app.core.config import settings
from utils.gentle_utils import align_audio_with_transcript


class GentleTranscriptionAligner(ITranscriptionAligner):
    """Adapter wrapping Gentle alignment service.

    It writes the transcript text to a temp file and calls `align_audio_with_transcript`.
    """

    def __init__(
        self,
        *,
        temp_dir: str,
        url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.url = url or getattr(
            settings, "gentle_url", "http://localhost:8765/transcriptions"
        )
        self.timeout = int(timeout or getattr(settings, "gentle_timeout", 30))
        # Resolve default temp_dir under base temp directory to avoid project root writes
        self.temp_dir = temp_dir

    def align(
        self,
        audio_path: str,
        words_id: str,
        transcript_text: str,
        *,
        min_success_ratio: float = 0.8,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Ensure audio exists
        if not audio_path or not Path(audio_path).exists():
            return [], {
                "is_verified": False,
                "success_ratio": 0.0,
                "success_count": 0,
                "total_words": 0,
            }

        # Persist transcript under per-segment directory and keep it
        target_dir = Path(self.temp_dir) / words_id
        target_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = target_dir / "transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as tf:
            tf.write(transcript_text or "")

        result, verify = align_audio_with_transcript(
            audio_path=str(audio_path),
            transcript_path=str(transcript_path),
            gentle_url=self.url,
            timeout=self.timeout,
            min_success_ratio=min_success_ratio,
        )
        # Persist words JSON under per-segment directory
        out_path = target_dir / "words.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return result.get("words", []), verify
