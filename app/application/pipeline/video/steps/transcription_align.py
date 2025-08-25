from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.application.pipeline.base import PipelineContext, BaseStep
from app.application.interfaces import ITranscriptionAligner

from utils.gentle_utils import filter_successful_words
from utils.alignment_utils import find_greedy_shift_match
from utils.text_utils import (
    _fallback_split,  # type: ignore
    create_text_over_item,
    normalize_text,
    split_transcript,
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

    def __init__(self, aligner: Optional[ITranscriptionAligner] = None) -> None:
        # Adapter for alignment (Gentle or alternative)
        self.aligner = aligner
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

            # Default: no text_over
            seg_out["text_over"] = []

            # Split transcript into readable chunks; prefer LLM splitter like legacy
            if content:
                try:
                    # Use async split_transcript for better quality
                    chunks: List[str] = await split_transcript(content)
                    if not chunks:
                        # fallback if LLM returns empty
                        chunks = _fallback_split(content)
                except Exception:
                    # Robust fallback
                    chunks = _fallback_split(content)
                    if not chunks:
                        chunks = [content]
            else:
                chunks = []

            # If no content at all, move on
            if not content:
                updated_segments.append(seg_out)
                continue

            word_items: List[Dict[str, Any]] = []
            verify: Dict[str, Any] = {}

            # Try alignment if audio available
            if audio_path and Path(audio_path).exists() and self.aligner is not None:
                try:
                    word_items, verify = self.aligner.align(
                        audio_path=str(audio_path),
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

            text_over_items: List[Dict[str, Any]] = []
            if word_items:
                # Legacy-like grouping over normalized lines with greedy shift matching
                success_words = filter_successful_words(word_items)
                word_index = 0
                cur_t = 0.0  # for synthesized items when matching fails
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    line_norm = normalize_text(chunk)
                    if not line_norm:
                        continue
                    group = find_greedy_shift_match(
                        line_norm,
                        success_words,
                        start_idx=word_index,
                        max_skips_per_side=2,
                        similarity_threshold=1.0,
                    )
                    item = create_text_over_item(chunk, group)
                    if item:
                        text_over_items.append(item)
                        # advance word_index to after last matched item
                        for it in reversed(group):
                            if it in success_words:
                                word_index = success_words.index(it) + 1
                                break
                        # update cur_t to the end of this item
                        try:
                            cur_t = float(item["start_time"]) + float(item["duration"])  # type: ignore
                        except Exception:
                            pass
                    else:
                        # Synthesize a fallback item to preserve content
                        wc = len(chunk.split())
                        dur = max(
                            getattr(settings, "text_over_min_duration", 1.0),
                            min(
                                getattr(settings, "text_over_max_duration", 6.0),
                                getattr(settings, "text_over_word_seconds", 0.4) * wc,
                            ),
                        )
                        text_over_items.append(
                            {
                                "text": chunk,
                                "start_time": cur_t,
                                "duration": dur,
                                "word_count": wc,
                            }
                        )
                        cur_t += dur
            else:
                # Fallback-only: synthesize simple timings per chunk
                cur_t = 0.0
                for chunk in chunks:
                    wc = len(chunk.split())
                    dur = max(
                        getattr(settings, "text_over_min_duration", 1.0),
                        min(
                            getattr(settings, "text_over_max_duration", 6.0),
                            getattr(settings, "text_over_word_seconds", 0.4) * wc,
                        ),
                    )
                    text_over_items.append(
                        {
                            "text": chunk,
                            "start_time": cur_t,
                            "duration": dur,
                            "word_count": wc,
                        }
                    )
                    cur_t += dur

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
