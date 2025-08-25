from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        temp_dir: str | None = None,
        url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.url = url or getattr(
            settings, "gentle_url", "http://localhost:8765/transcriptions"
        )
        self.timeout = int(timeout or getattr(settings, "gentle_timeout", 30))
        # Resolve default temp_dir under base temp directory to avoid project root writes
        if temp_dir:
            self.temp_dir = temp_dir
        else:
            import os
            base_dir = os.getenv("TEMP_BASE_DIR", "data/tmp")
            try:
                os.makedirs(base_dir, exist_ok=True)
            except Exception:
                pass
            self.temp_dir = os.path.join(base_dir, settings.temp_batch_dir)

    def align(
        self,
        audio_path: str,
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

        # Ensure temp dir exists
        if self.temp_dir:
            try:
                Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8", dir=self.temp_dir or None
        ) as tf:
            tf.write(transcript_text or "")
            transcript_path = tf.name

        try:
            result, verify = align_audio_with_transcript(
                audio_path=str(audio_path),
                transcript_path=str(transcript_path),
                gentle_url=self.url,
                timeout=self.timeout,
                min_success_ratio=min_success_ratio,
            )
            return result.get("words", []), verify
        finally:
            try:
                Path(transcript_path).unlink(missing_ok=True)
            except Exception:
                pass
