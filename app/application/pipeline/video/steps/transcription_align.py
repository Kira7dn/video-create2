from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces import (
    ITranscriptionAligner,
    ITranscriptSplitter,
    ITextOverBuilder,
)


class TranscriptionAlignStep(BaseStep):
    name = "transcription_align"
    required_keys = ["segments"]
    """
    Aligns voice_over audio with its transcript via Gentle and produces text_over.

    Expectations:
    - Each segment has voice_over.content (transcript text)
    - Prefer voice_over.local_path for audio; skip segment if missing
    - Gentle server available at http://localhost:8765/transcriptions
    """

    def __init__(
        self,
        aligner: ITranscriptionAligner,
        splitter: ITranscriptSplitter,
        builder: ITextOverBuilder,
    ) -> None:
        # Adapter for alignment (Gentle or alternative)
        self.aligner = aligner
        # Adapter for transcript splitting (LLM or custom)
        self.splitter = splitter
        # Adapter for building text_over items
        self.builder = builder
        self.logger = logging.getLogger(__name__)

    async def run(self, context: PipelineContext) -> None:  # type: ignore[override]
        # Prefer segments (after downloads); fallback to validated_data.segments
        segments: List[Dict[str, Any]] = (
            context.get("segments")
            or (context.get("validated_data") or {}).get("segments", [])
            or []
        )
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")

        updated_segments: List[Dict[str, Any]] = []
        alignment_stats: List[Dict[str, Any]] = []

        for idx, seg in enumerate(segments):
            seg_out = dict(seg)
            voice = seg.get("voice_over") or {}
            content: str = (voice or {}).get("content") or ""
            audio_path: Optional[str] = (voice or {}).get("local_path")
            seg_id = seg.get("id")

            # Default: no text_over
            seg_out["text_over"] = []

            # Split transcript into readable chunks via injected splitter with robust fallback
            if content:
                try:
                    chunks: List[str] = await self.splitter.split(content, seg_id)
                except Exception:
                    chunks = []
            else:
                chunks = []

            # If no content at all, move on
            if not content:
                updated_segments.append(seg_out)
                continue

            word_items: List[Dict[str, Any]] = []
            verify: Dict[str, Any] = {}

            # Try alignment if audio available
            if audio_path and Path(audio_path).exists():
                try:
                    word_items, verify = self.aligner.align(
                        audio_path=str(audio_path),
                        words_id=seg_id,
                        transcript_text=content,
                        min_success_ratio=getattr(
                            settings, "alignment_min_success_ratio", 0.8
                        ),
                    )
                except Exception:
                    # Alignment failed → fallback only
                    word_items = []
                    verify = {
                        "is_verified": False,
                        "success_ratio": 0.0,
                        "success_count": 0,
                        "total_words": 0,
                    }

            # Build text_over items via adapter
            text_over_items: List[Dict[str, Any]] = self.builder.build(
                word_items=word_items,
                chunks=chunks,
                text_over_id=seg_id,
            )

            # Safety: if builder produced no items but we have content, synthesize fallback
            if not text_over_items and content:
                fallback_chunks = chunks if chunks else [content]
                try:
                    text_over_items = self.builder.build(
                        word_items=[],
                        chunks=fallback_chunks,
                        text_over_id=seg_id,
                    )
                except Exception:
                    text_over_items = []

            seg_out["text_over"] = text_over_items
            try:
                self.logger.info(
                    "TranscriptionAlignStep: segment %s created %d text_over items",
                    seg_out.get("id") or idx,
                    len(text_over_items),
                )
            except Exception:
                pass
            updated_segments.append(seg_out)

            if verify:
                alignment_stats.append(
                    {
                        "segment_index": idx,
                        "is_verified": bool(verify.get("is_verified")),
                        "success_ratio": verify.get("success_ratio"),
                        "success_count": verify.get("success_count"),
                        "total_words": verify.get("total_words"),
                        "alignment_issues": verify.get("alignment_issues", []),
                    }
                )

        context.set("segments", updated_segments)
        context.set("alignment_stats", alignment_stats)
