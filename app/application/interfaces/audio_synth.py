from __future__ import annotations
from typing import Protocol, Optional


class IAudioSynthesizer(Protocol):
    """Text-to-speech / audio synthesis interface."""

    async def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Synthesize speech and return the path to the generated audio file."""
        ...
