from __future__ import annotations

from typing import Protocol, Dict, Any, List


class ITranscriptionAligner(Protocol):
    """Aligns an audio file with its transcript text and returns word items and verification stats."""

    def align(
        self,
        audio_path: str,
        words_id: str,
        transcript_text: str,
        *,
        min_success_ratio: float = 0.8,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return (word_items, verify)

        - word_items: list of Gentle-like word dicts
        - verify: dict with keys like is_verified, success_ratio, success_count, total_words
        """
        ...
