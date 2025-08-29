from __future__ import annotations

from typing import List
import logging
import time
import re
import os
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel  # type: ignore

from app.application.interfaces import ITranscriptSplitter
from app.core.config import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _TranscriptSegments(BaseModel):
    segments: List[str]

    @field_validator("segments", mode="after")
    @classmethod
    def _validate_segments(cls, v: List) -> List[str]:
        for item in v:
            if not isinstance(item, str):
                raise ValueError("string_type: segment must be a string")
        return v


class LLMTranscriptSplitter(ITranscriptSplitter):

    def __init__(self, *, temp_dir: str, model_name: str | None = None) -> None:
        # Where intermediate files (e.g., transcript_lines.txt) are written.
        # Must be provided by the caller.
        self._temp_dir = temp_dir
        self._model_name = model_name or settings.ai_pydantic_model

    async def split(self, content: str, content_id: str) -> List[str]:
        """Split transcript using an LLM, with strict preservation and robust fallback."""
        if not content:
            return []

        start_time = time.time()

        model = OpenAIModel(model_name=self._model_name)
        logger.info(
            "LLMTranscriptSplitter: initializing OPENAI agent with model=%s (content_id=%s)",
            self._model_name,
            content_id,
        )
        agent = Agent(
            model,
            output_type=_TranscriptSegments,
            system_prompt="""You are a natural language processing expert.
            Your task is to segment the transcript into short, natural sentences.
            Each sentence must be a complete semantic unit, readable and natural.

            CRITICAL RULES (MUST FOLLOW EXACTLY):
            - Preserve the original text EXACTLY. Do not add, remove, or replace ANY characters.
            - Do NOT expand or contract words. Keep contractions and apostrophes exactly as written (e.g., "you're" must stay "you're").
            - Do NOT normalize punctuation, whitespace, or casing. Keep original punctuation and case. Only insert segment breaks.
            - Do NOT paraphrase, summarize, or reorder words.
            - Your output must be a list of segments that, when concatenated with a single space between segments, contains EXACTLY the same characters as the original input (ignoring only the inserted segment boundaries).
            """,
            model_settings={"temperature": 0.1},
        )

        prompt = f"""
            CRITICAL: Preserve the original text EXACTLY.
            - Do NOT change words, punctuation, apostrophes, contractions, spacing, or casing.
            - Do NOT expand contractions ("you're" must remain "you're").
            - Your job is ONLY to insert segment boundaries; do not modify the content itself.

            Original transcript (PRESERVE EXACTLY):
            "{content}"

            REQUIREMENTS:
            1. Preserve ALL content exactly as written (characters and order unchanged).
            2. Each segment: 4-12 words (readable chunks).
            3. Maximum 80 characters per segment (screen readability).
            4. Break at natural phrase boundaries; keep related concepts together.
            5. Output segments such that, when joined (with single spaces between segments), they reconstruct the original text exactly aside from the inserted breaks.
            6. Perfect for video text overlay (3-6 seconds per segment)

            Example of good segmentation:
            {{
            "segments": [
                "Hello everyone and welcome back",
                "to our channel about technology",
                "Today we're going to explore",
                "machine learning and its applications",
                "in the modern world"
            ]
            }}

            VALIDATION: After creating segments, verify that when joined together,
            they contain exactly the same words as the original transcript.
            """

        # Run LLM and collect segments (fallback on errors)
        try:
            result = await agent.run(user_prompt=prompt)
            segs = result.output
            segments_list = list(segs.segments)
            logger.info(
                "LLMTranscriptSplitter: using OPENAI for segmentation (content_id=%s)",
                content_id,
            )
        except Exception as e:
            logger.warning(
                "LLMTranscriptSplitter: LLM exception -> %s (content_id=%s)",
                repr(e),
                content_id,
            )
            logger.info(
                "LLMTranscriptSplitter: using FALLBACK due to LLM exception (content_id=%s)",
                content_id,
            )
            segments_list = self._fallback_split(content)

        duration = time.time() - start_time

        # Validate preservation; if fails, fallback segments
        if not self._validate_content_preservation(content, segments_list):
            original_tokens = re.findall(r"\b\w+\b", content.lower())
            segmented_tokens: List[str] = []
            for segment in segments_list or []:
                segmented_tokens.extend(re.findall(r"\b\w+\b", segment.lower()))

            if not original_tokens:
                reason = "Original tokens are empty"
            else:
                missing = set(original_tokens) - set(segmented_tokens)
                extra = set(segmented_tokens) - set(original_tokens)
                reason = f"Tokens mismatch. Missing: {missing} | Extra: {extra}"
            logger.warning(
                "LLMTranscriptSplitter: LLM output failed content preservation validation (content_id=%s). Reason: %s. Switching to fallback.",
                content_id,
                reason,
            )
            logger.info(
                "LLMTranscriptSplitter: using FALLBACK due to validation failure (content_id=%s)",
                content_id,
            )
            segments_list = self._fallback_split(content)

        # Save segments to file in the same folder structure as TextOverBuilder
        # i.e., {temp_dir}/{content_id}/transcript_lines.txt
        out_path = None
        if self._temp_dir and content_id:
            target_dir = os.path.join(self._temp_dir, str(content_id))
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, "transcript_lines.txt")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    for segment in segments_list:
                        f.write(segment + "\n")
            except Exception:
                # Do not crash pipeline on IO errors
                out_path = None

        logger.debug(
            "Completed transcript segmentation in %.2f seconds, created %d segments%s",
            duration,
            len(segments_list),
            f" at {out_path}" if out_path else "",
        )
        return segments_list

    @staticmethod
    def _validate_content_preservation(original: str, segments: List[str]) -> bool:
        """Ensure segmented tokens equal original tokens (case/punct-insensitive)."""
        original_tokens = re.findall(r"\b\w+\b", original.lower())
        segmented_tokens: List[str] = []
        for segment in segments or []:
            segmented_tokens.extend(re.findall(r"\b\w+\b", segment.lower()))

        if not original_tokens:
            return len(segmented_tokens) == 0
        if not segments:
            return False
        if original_tokens != segmented_tokens:
            missing = set(original_tokens) - set(segmented_tokens)
            extra = set(segmented_tokens) - set(original_tokens)
            logger.warning(
                "Content preservation failed: tokens mismatch. Missing: %s | Extra: %s",
                missing,
                extra,
            )
            return False
        return True

    @staticmethod
    def _fallback_split(content: str) -> List[str]:
        """Heuristic splitter when LLM unavailable or invalid result."""
        if not content.strip():
            return []

        sentence_enders = r"(?<=[.!?])\s+"
        comma_separators = r"(?<=,)\s+(?=\w)"
        conjunctions = r"\s+(?=(?:and|or|but|so|because|when|if|while|although|however|therefore|moreover)\s+)"
        natural_pauses = (
            r"\s+(?=(?:now|then|next|first|second|finally|meanwhile|additionally)\s+)"
        )

        pattern = (
            f"{sentence_enders}|{comma_separators}|{conjunctions}|{natural_pauses}"
        )
        segments = [
            s.strip()
            for s in re.split(pattern, content, flags=re.IGNORECASE)
            if s.strip()
        ]

        result: List[str] = []
        for segment in segments:
            words = segment.split()
            if 4 <= len(words) <= 12 and len(segment) <= 80:
                result.append(segment)
            elif len(words) <= 3:
                if result and len(result[-1].split()) + len(words) <= 12:
                    result[-1] = result[-1] + " " + segment
                else:
                    result.append(segment)
            else:
                for i in range(0, len(words), 8):
                    chunk_words = words[i : i + 8]
                    if chunk_words:
                        result.append(" ".join(chunk_words))

        result = [s for s in result if s.strip()]
        if not result and content.strip():
            words = content.split()
            for i in range(0, len(words), 8):
                chunk_words = words[i : i + 8]
                if chunk_words:
                    result.append(" ".join(chunk_words))

        logger.debug(
            "Fallback split created %d segments from original content", len(result)
        )
        return result
