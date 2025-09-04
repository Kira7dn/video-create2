from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

from app.application.interfaces import ITextOverBuilder
from utils.gentle_utils import filter_successful_words
from utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


class TextOverBuilder2(ITextOverBuilder):
    """A simplified text_over builder (ver02).

    - Input: aligned words (words.json structure) and optional transcript via `chunks`
    - Logic priority:
      1) If transcript text is provided (via `chunks`), split by punctuation in transcript ('.', ',', '?', '!', ';', ':', '—', '–', '…')
         and additionally split inside segments on long pauses between words (> PAUSE_GAP_S)
      2) Fallback: split when a word token itself contains one of the punctuation above
         or when a long pause between consecutive words is detected
    - Output items: text, start_time, duration, word_count
    - Ignores advanced alignment/expansion logic
    """

    def __init__(self, *, temp_dir: str, pause_gap_s: float = 0.4) -> None:
        self._temp_dir = temp_dir
        # configurable heuristics
        # Sentence/phrase delimiters (ASCII + Unicode): comma, period, question, exclamation, semicolon, colon,
        # em dash, en dash, ellipsis
        self.PUNCTUATION_CHARS = {",", ".", "?", "!", ";", ":", "—", "–", "…"}
        self.PAUSE_GAP_S = float(pause_gap_s)

    def build(
        self,
        *,
        word_items: List[Dict[str, Any]],
        chunks: List[str],
        text_over_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not word_items:
            raise ValueError("alignment_required: word_items are required")

        success_words = filter_successful_words(word_items)
        if not success_words:
            raise ValueError("no successful words available for alignment")

        transcript_text = " ".join(chunks).strip() if chunks else ""
        if transcript_text:
            items = self._segment_by_transcript_punctuation(success_words, transcript_text)
            # If transcript-based produced nothing (edge cases), fallback to word punctuation
            if not items:
                items = self._segment_by_punctuation(success_words)
        else:
            items = self._segment_by_punctuation(success_words)

        if text_over_id:
            self._persist_output(items, text_over_id)

        return items

    def _segment_by_transcript_punctuation(self, words: List[Dict[str, Any]], transcript: str) -> List[Dict[str, Any]]:
        """Segment words into groups based on punctuation positions in transcript.

        Minimal heuristic approach:
        - Split transcript into tokens on whitespace; whenever a token contains ',' or '.', end a segment.
        - Compute counts of transcript tokens per segment (including the token carrying punctuation).
        - Distribute timed words into segments proportionally to those counts, preserving order.
        This avoids complex alignment while honoring transcript punctuation as the primary signal.
        """
        # Build transcript token segment sizes
        raw_tokens = [t for t in transcript.split() if t]
        seg_sizes: List[int] = []
        cur = 0
        for t in raw_tokens:
            cur += 1
            if any(ch in t for ch in self.PUNCTUATION_CHARS):
                seg_sizes.append(cur)
                cur = 0
        if cur > 0:
            seg_sizes.append(cur)

        if not seg_sizes:
            return []

        # Distribute words proportionally by token share
        total_tokens = sum(seg_sizes)
        total_words = len(words)
        if total_tokens <= 0 or total_words <= 0:
            return []

        # Greedy left-to-right allocation that prioritizes punctuation boundaries.
        # Ensures: at least 1 word per transcript segment and reserves at least 1 word
        # for each remaining segment, so the first word after a punctuation stays in
        # the next segment.
        ideal_sizes = [total_words * (sz / total_tokens) for sz in seg_sizes]
        sizes: List[int] = []
        remaining_words = total_words
        remaining_segments = len(seg_sizes)
        for i, ideal in enumerate(ideal_sizes):
            remaining_segments = len(seg_sizes) - i
            # base target: floor of ideal
            target = int(ideal)
            # bounds
            min_i = 1
            max_i = max(min_i, remaining_words - (remaining_segments - 1))
            # clamp
            size_i = max(min_i, min(target, max_i))
            # if still under-allocating due to rounding, try to add until max_i
            while size_i < max_i and (sum(sizes) + size_i) < int(sum(ideal_sizes[: i + 1])):
                size_i += 1
            sizes.append(size_i)
            remaining_words -= size_i
        # If any words remain (due to rounding), distribute 1-by-1 from left to right
        idx = 0
        while remaining_words > 0 and sizes:
            sizes[idx] += 1
            remaining_words -= 1
            idx = (idx + 1) % len(sizes)

        # Slice words accordingly and further split by pauses
        segments: List[Dict[str, Any]] = []
        idx = 0
        for sz in sizes:
            if sz <= 0:
                continue
            chunk_words = words[idx: idx + sz]
            idx += sz
            if not chunk_words:
                continue
            # Further split by long pauses within this transcript-based chunk
            buf: List[Dict[str, Any]] = []
            prev_end: Optional[float] = None
            for w in chunk_words:
                if not (isinstance(w, dict) and "start" in w and "end" in w):
                    continue
                st = float(w.get("start", 0.0) or 0.0)
                if prev_end is not None and st - prev_end > self.PAUSE_GAP_S:
                    self._append_segment_from_words(buf, segments)
                    buf = []
                buf.append(w)
                prev_end = float(w.get("end", st) or st)
            self._append_segment_from_words(buf, segments)

        # Enforce sequential non-overlap and min duration
        prev_end: Optional[float] = None
        for it in segments:
            st = float(it.get("start_time", 0.0))
            dur = float(it.get("duration", 0.0))
            if prev_end is not None and st < prev_end:
                st = prev_end
            dur = max(0.1, dur)
            it["start_time"] = st
            it["duration"] = dur
            prev_end = st + dur

        return segments

    def _append_segment_from_words(self, buf_words: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> None:
        """Build a segment dict from buffered words and append to segments.
        Ensures text, timing, duration>=0.1, and word_count via normalize_text.
        """
        if not buf_words:
            return
        text_tokens = [str(w.get("word", "")).strip() for w in buf_words]
        text = " ".join(t for t in text_tokens if t)
        try:
            start = float(buf_words[0]["start"])  # type: ignore[index]
            end = float(buf_words[-1]["end"])     # type: ignore[index]
        except Exception:
            start = float(buf_words[0].get("start", 0.0) or 0.0)
            end = float(buf_words[-1].get("end", start) or start)
        duration = max(0.1, max(0.0, end - start))

        try:
            wc = len(normalize_text(text))
        except Exception:
            wc = len([t for t in text_tokens if t])

        segments.append({
            "text": text,
            "start_time": max(0.0, start),
            "duration": duration,
            "word_count": wc,
        })

    def _segment_by_punctuation(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        buf_words: List[Dict[str, Any]] = []

        def flush() -> None:
            if not buf_words:
                return
            self._append_segment_from_words(buf_words, segments)
            buf_words.clear()

        prev_end: Optional[float] = None
        for w in words:
            # Only accept well-formed timed words
            if not (isinstance(w, dict) and "start" in w and "end" in w):
                continue
            st = float(w.get("start", 0.0) or 0.0)
            if prev_end is not None and st - prev_end > self.PAUSE_GAP_S:
                flush()
            buf_words.append(w)
            token = str(w.get("word", ""))
            # Split on a word that contains configured punctuation
            if any(ch in token for ch in self.PUNCTUATION_CHARS):
                flush()
            prev_end = float(w.get("end", st) or st)

        # Flush any remainder
        flush()

        # Fallback: if no segments could be formed but words exist, create one covering all
        if not segments and words:
            try:
                start = float(words[0].get("start", 0.0) or 0.0)
                end = float(words[-1].get("end", start) or start)
            except Exception:
                start, end = 0.0, 0.0
            text = " ".join(str(w.get("word", "")).strip() for w in words)
            segments = [
                {
                    "text": text,
                    "start_time": max(0.0, start),
                    "duration": max(0.1, end - start),
                    "word_count": len(normalize_text(text)),
                }
            ]

        # Enforce non-overlap and minimum duration sequentially
        prev_end: Optional[float] = None
        for it in segments:
            st = float(it.get("start_time", 0.0))
            dur = float(it.get("duration", 0.0))
            if prev_end is not None and st < prev_end:
                st = prev_end
            dur = max(0.1, dur)
            it["start_time"] = st
            it["duration"] = dur
            prev_end = st + dur

        return segments

    def _persist_output(self, text_over_items: List[Dict[str, Any]], text_over_id: str) -> None:
        try:
            target_dir = Path(self._temp_dir) / text_over_id
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / "text_over.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(text_over_items, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Persisted {len(text_over_items)} text overlay items to {out_path}"
            )
        except Exception as e:
            logger.error(f"Failed to persist text overlay items: {e}")
            # Do not raise
