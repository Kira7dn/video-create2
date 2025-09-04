from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import math

from app.application.interfaces import ITextOverBuilder
from utils.gentle_utils import filter_successful_words
from utils.alignment_utils import find_greedy_shift_match, find_flexible_match
from utils.text_utils import create_text_over_item, normalize_text

logger = logging.getLogger(__name__)


@dataclass
class ExpansionConfig:
    """Configuration for text overlay expansion behavior."""

    max_expand_s: float = (
        0.250  # Maximum expansion per side (seconds) when no neighbor; with pad_s=0.08 → 0.33 (matches tests)
    )
    max_gap_s: float = 0.400  # Do not bridge gaps larger than this
    pad_s: float = 0.080  # Visual padding when no neighbor exists
    min_duration_s: float = 0.100  # Minimum overlay duration
    lookahead_window: int = (
        80  # Flexible match lookahead window (increased for robustness)
    )
    max_skips_per_side: int = 4  # Greedy match tolerance
    similarity_threshold: float = 0.85  # Slightly relaxed to reduce overshoot misses
    max_internal_gap_s: float = (
        0.950  # Bridge sub-second internal gaps, but avoid crossing inter-line pauses (~>1s)
    )
    # Backoff search config: when initial flexible match fails, widen the window and allow a small look-back
    backoff_lookback: int = 15
    backoff_window: int = 140
    # Optional: enforce a minimum duration for multi-word lines (>=4 tokens). Default disabled.
    min_multiword_duration_s: float = 0.0


class TextOverBuilder(ITextOverBuilder):
    """Builds text_over items in alignment-only mode.

    - Maps transcript chunks to aligned words using greedy shift matching.
    - Falls back to a flexible partial match within a lookahead window if needed.
    - Requires aligned word items; no text-only fallback is supported.
    - Applies hybrid expansion to cover adjacent not-found words with conservative caps.
    """

    def __init__(
        self, *, temp_dir: str, config: Optional[ExpansionConfig] = None
    ) -> None:
        self._temp_dir = temp_dir
        self._config = config or ExpansionConfig()
        # Caches populated per build() call
        self._success_index_map: Dict[int, int] = {}
        self._success_token_sets: List[set] = []

    def build(
        self,
        *,
        word_items: List[Dict[str, Any]],
        chunks: List[str],
        text_over_id: Optional[str] = None,
        media_duration_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Build text overlay items from transcript chunks and aligned words.

        Args:
            word_items: List of word alignment data from speech recognition
            chunks: List of transcript text chunks to overlay
            text_over_id: Optional ID for persistence
            media_duration_s: Optional media duration for end-of-media clamping

        Returns:
            List of text overlay items with timing and content

        Raises:
            ValueError: If word_items is empty or no successful words available
        """
        if not word_items:
            raise ValueError("alignment_required: word_items are required")

        success_words = filter_successful_words(word_items)
        if not success_words:
            raise ValueError("no successful words available for alignment")

        # Build caches for this run
        try:
            self._success_index_map = {
                id(w): idx for idx, w in enumerate(success_words)
            }
        except Exception:
            self._success_index_map = {}
        try:
            self._success_token_sets = [
                (
                    set(normalize_text(str(w.get("word", ""))))
                    if isinstance(w, dict)
                    else set()
                )
                for w in success_words
            ]
        except Exception:
            self._success_token_sets = [set() for _ in success_words]

        text_over_items: List[Dict[str, Any]] = []
        word_index = 0

        for chunk in chunks:
            if not chunk.strip():
                continue

            item = self._process_chunk(
                chunk, success_words, word_index, media_duration_s
            )
            if item:
                text_over_items.append(item)
                word_index = self._advance_word_index(
                    item.get("_matched_group", []), success_words, word_index
                )

        # Post-pass: normalize total durations against success words + allowance
        try:
            self._normalize_total_duration(text_over_items, success_words)
        except Exception as e:
            logger.debug(f"normalize_total_duration skipped due to error: {e}")

        # Post-pass: enforce ordering, non-overlap, and correct word_count
        try:
            self._postprocess_items(text_over_items)
        except Exception as e:
            logger.debug(f"postprocess_items skipped due to error: {e}")

        # Final safety clamp: enforce global window [first_success_start - cap, last_success_end + cap]
        try:
            self._final_global_clamp(text_over_items, success_words)
        except Exception as e:
            logger.debug(f"final_global_clamp skipped due to error: {e}")

        if text_over_id:
            self._persist_output(text_over_items, text_over_id)

        return text_over_items

    def _final_global_clamp(
        self, items: List[Dict[str, Any]], success_words: List[Dict[str, Any]]
    ) -> None:
        if not items or not success_words:
            return
        first = None
        last = None
        for w in success_words:
            try:
                if isinstance(w, dict) and "start" in w and "end" in w:
                    s = float(w["start"])
                    e = float(w["end"])
                    if math.isfinite(s) and math.isfinite(e) and e > s:
                        first = s if first is None else min(first, s)
                        last = e if last is None else max(last, e)
            except Exception:
                continue
        if first is None or last is None:
            return
        cap = self._cap_each_side()
        min_start = max(0.0, first - cap)
        # Epsilon-safe right bound to prevent tiny FP overshoot
        eps = 1e-6
        max_end = (last + cap) - eps
        for it in items:
            try:
                start = float(it.get("start_time", 0.0))
                dur = float(it.get("duration", 0.0))
                end = start + dur
            except (ValueError, TypeError):
                continue
            if start < min_start:
                start = min_start
            if end > max_end:
                end = max_end
            if end <= start:
                end = start + self._config.min_duration_s
            it["start_time"] = start
            it["duration"] = max(0.0, end - start)

    def _process_chunk(
        self,
        chunk: str,
        success_words: List[Dict[str, Any]],
        word_index: int,
        media_duration_s: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process a single transcript chunk into a text overlay item."""
        line_norm = normalize_text(chunk)
        if not line_norm:
            return None

        # Find alignment using greedy then flexible matching
        group = self._find_alignment(line_norm, success_words, word_index)
        if not group:
            snippet = (chunk[:80] + "...") if len(chunk) > 80 else chunk
            raise ValueError(
                f"alignment_required: failed to create text_over item; "
                f"chunk='{snippet}' tokens={len(line_norm)}"
            )

        # Bridge small internal gaps to widen the matched group
        # IMPORTANT: only bridge with words that belong to this chunk (by normalized token),
        # to avoid swallowing true neighbors outside the chunk.
        group = self._extend_group_internal(group, success_words, line_norm)

        # Create base item from matched words
        item = create_text_over_item(chunk, group)
        if not item:
            return None

        # Apply hybrid expansion
        self._apply_expansion(item, group, success_words, media_duration_s)

        # Store matched group for word index advancement
        item["_matched_group"] = group

        return item

    def _find_alignment(
        self,
        normalized_text: List[str],
        success_words: List[Dict[str, Any]],
        word_index: int,
    ) -> List[Dict[str, Any]]:
        """Find word alignment using greedy then flexible matching."""

        # Small helpers to avoid duplication
        def _clamp(a: int, lo: int, hi: int) -> int:
            return max(lo, min(a, hi))

        def _run_flexible_window(start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
            """Run flexible match on a clamped [start_idx:end_idx) window."""
            s = _clamp(int(start_idx), 0, len(success_words))
            e = _clamp(int(end_idx), s, len(success_words))
            if e <= s:
                return []
            window_items = success_words[s:e]
            return (
                find_flexible_match(
                    normalized_text, window_items, max_lookahead=len(window_items)
                )
                or []
            )

        # Try greedy shift match first
        group = find_greedy_shift_match(
            normalized_text,
            success_words,
            start_idx=word_index,
            max_skips_per_side=self._config.max_skips_per_side,
            similarity_threshold=self._config.similarity_threshold,
        )

        if group:
            return group

        # Fallback to flexible partial match on primary window
        flex_group = _run_flexible_window(
            word_index, word_index + self._config.lookahead_window
        )

        if flex_group:
            return flex_group

        # Backoff: expand window and allow slight look-back in case cursor overshot the true region
        try:
            start = int(word_index) - int(self._config.backoff_lookback)
            end = int(word_index) + int(self._config.backoff_window)
            backoff_group = _run_flexible_window(start, end)
            if backoff_group:
                logger.info(
                    "[text_over_builder:backoff] used backoff search start=%s end=%s size=%s",
                    max(0, start),
                    min(len(success_words), end),
                    min(len(success_words), max(0, end)) - max(0, start),
                )
                return backoff_group
        except Exception as e:
            logger.debug(f"backoff search failed: {e}")

        return []

    def _extend_group_internal(
        self,
        group: List[Dict[str, Any]],
        success_words: List[Dict[str, Any]],
        normalized_chunk_tokens: List[str],
    ) -> List[Dict[str, Any]]:
        """Extend a matched group by bridging small internal gaps to adjacent words.

        Heuristic: If the next success word starts within max_internal_gap_s of the
        current group's end, include it. Repeat until the gap exceeds the threshold
        or we reach the end. This helps cover short regions where alignment tokens
        were dropped but audio continues seamlessly.
        """
        if not group:
            return group

        try:
            token_set = set(normalized_chunk_tokens or [])
            # Determine current group indices and boundary times
            left_idx, right_idx = self._find_group_indices(group, success_words)
            if left_idx is None or right_idx is None:
                return group

            current_start = min(
                (w["start"] for w in group if isinstance(w, dict) and "start" in w),
                default=None,
            )
            current_end = max(
                (w["end"] for w in group if isinstance(w, dict) and "end" in w),
                default=None,
            )
            if current_start is None or current_end is None:
                return group

            # Directional bridging helper to eliminate duplicate code
            def _bridge(direction: int) -> None:
                nonlocal left_idx, right_idx, current_start, current_end
                if direction > 0:
                    idx = right_idx + 1
                    while idx < len(success_words):
                        w = success_words[idx]
                        # Skip untimed entries (e.g., not-found-in-audio) instead of stopping
                        if not (isinstance(w, dict) and "start" in w and "end" in w):
                            idx += direction
                            continue
                        try:
                            w_start = float(w["start"])
                            w_end = float(w["end"])
                        except (ValueError, TypeError):
                            idx += direction
                            continue
                        if not (math.isfinite(w_start) and math.isfinite(w_end)):
                            idx += direction
                            continue

                        gap = w_start - float(current_end)
                        w_tokens_set = (
                            self._success_token_sets[idx]
                            if 0 <= idx < len(self._success_token_sets)
                            else set()
                        )
                        belongs = any(t in token_set for t in w_tokens_set)
                        if gap <= self._config.max_internal_gap_s and belongs:
                            group.append(w)
                            current_end = max(current_end, w_end)
                            right_idx = idx
                            idx += direction
                            continue
                        break
                else:
                    idx = left_idx - 1
                    while idx >= 0:
                        w = success_words[idx]
                        # Skip untimed entries (e.g., not-found-in-audio) instead of stopping
                        if not (isinstance(w, dict) and "start" in w and "end" in w):
                            idx += direction
                            continue
                        try:
                            w_start = float(w["start"])
                            w_end = float(w["end"])
                        except (ValueError, TypeError):
                            idx += direction
                            continue
                        if not (math.isfinite(w_start) and math.isfinite(w_end)):
                            idx += direction
                            continue

                        gap = float(current_start) - w_end
                        w_tokens_set = (
                            self._success_token_sets[idx]
                            if 0 <= idx < len(self._success_token_sets)
                            else set()
                        )
                        belongs = any(t in token_set for t in w_tokens_set)
                        if gap <= self._config.max_internal_gap_s and belongs:
                            group.insert(0, w)
                            current_start = min(current_start, w_start)
                            left_idx = idx
                            idx += direction
                            continue
                        break

            # Bridge to the right and then to the left
            _bridge(+1)
            _bridge(-1)

        except Exception as e:
            logger.debug(f"_extend_group_internal failed: {e}")

        return group

    def _apply_expansion(
        self,
        item: Dict[str, Any],
        group: List[Dict[str, Any]],
        success_words: List[Dict[str, Any]],
        media_duration_s: Optional[float] = None,
    ) -> None:
        """Apply hybrid expansion to extend overlay timing."""
        if not group:
            return

        # Get base timing from matched words
        base_start, base_end = self._get_base_timing(item, group)

        # Find neighbor boundaries
        left_idx, right_idx = self._find_group_indices(group, success_words)
        prev_end, next_start = self._get_neighbor_times(
            left_idx, right_idx, success_words
        )
        # Determine if this group sits at the true extremes of the success_words timeline
        is_left_extreme = bool(left_idx is not None and left_idx == 0)
        is_right_extreme = bool(
            right_idx is not None and right_idx == len(success_words) - 1
        )

        # Calculate extensions directly using side calculator
        left_gap = (base_start - prev_end) if prev_end is not None else None
        right_gap = (next_start - base_end) if next_start is not None else None
        extend_left = self._calculate_side_extension(
            left_gap, is_left=True, allow_no_neighbor=is_left_extreme
        )
        extend_right = self._calculate_side_extension(
            right_gap, is_left=False, allow_no_neighbor=is_right_extreme
        )

        # POLICY: Only bridge to neighbor boundaries for sufficiently long chunks (>=9 words).
        # For shorter chunks, avoid adding neighbor-gap time (helps keep total durations near word durations).
        try:
            word_count = int(item.get("word_count", 0))
        except Exception:
            word_count = 0

        if word_count < 9:
            # If a neighbor exists on a side, disable bridging on that side (keep internal/base timing on that side).
            if prev_end is not None:
                extend_left = 0.0
            if next_start is not None:
                extend_right = 0.0

        # Apply timing with constraints
        new_start = max(0.0, base_start - extend_left)
        new_end = base_end + extend_right

        # Apply media duration clamping if available
        if media_duration_s is not None:
            new_end = min(new_end, media_duration_s)

        # Ensure minimum duration when no extensions
        if extend_left == 0.0 and extend_right == 0.0:
            new_end = max(new_end, new_start + self._config.min_duration_s)
        elif new_end <= new_start:
            new_end = new_start + 0.01

        item["start_time"] = new_start
        item["duration"] = new_end - new_start

        # Store metadata for normalization
        item["_base_start"] = float(base_start)
        item["_base_end"] = float(base_end)
        item["_ext_left"] = float(max(0.0, base_start - new_start))
        item["_ext_right"] = float(max(0.0, new_end - base_end))
        # Sum of actual word durations inside this item
        try:
            words_sum = 0.0
            for w in group:
                if isinstance(w, dict) and "start" in w and "end" in w:
                    s = float(w["start"])
                    e = float(w["end"])
                    if math.isfinite(s) and math.isfinite(e) and e > s:
                        words_sum += e - s
            item["_words_dur_sum"] = float(max(0.0, words_sum))
        except Exception:
            item["_words_dur_sum"] = float(max(0.0, item.get("duration", 0.0)))

    def _get_base_timing(
        self, item: Dict[str, Any], group: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Extract base start and end times from matched word group."""
        try:
            group_starts = [w["start"] for w in group if "start" in w]
            group_ends = [w["end"] for w in group if "end" in w]

            if group_starts and group_ends:
                return min(group_starts), max(group_ends)
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to extract group timing: {e}")

        # Fallback to item timing
        return item["start_time"], item["start_time"] + item["duration"]

    def _find_group_indices(
        self, group: List[Dict[str, Any]], success_words: List[Dict[str, Any]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find the indices of the matched group within success_words."""
        if not group:
            return None, None

        group_indices: List[int] = []
        # Use precomputed identity map if available
        word_to_index = self._success_index_map or {
            id(word): idx for idx, word in enumerate(success_words)
        }

        try:
            for word in group:
                # Try identity lookup first (faster)
                idx_id = word_to_index.get(id(word))
                if idx_id is not None:
                    group_indices.append(idx_id)
                else:
                    # Fallback to equality comparison
                    for idx, sw in enumerate(success_words):
                        if sw == word:
                            group_indices.append(idx)
                            break
                    else:
                        logger.debug(
                            f"Word not found in success_words: {word.get('word', 'unknown')}"
                        )

            if group_indices:
                return min(group_indices), max(group_indices)

        except (KeyError, AttributeError) as e:
            logger.warning(f"Failed to find group indices: {e}")

        return None, None

    def _get_neighbor_times(
        self,
        left_idx: Optional[int],
        right_idx: Optional[int],
        success_words: List[Dict[str, Any]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get timing of neighboring words."""
        prev_end = None
        next_start = None

        try:
            # Validate indices before accessing
            if (
                left_idx is not None
                and 0 <= left_idx < len(success_words)
                and left_idx > 0
            ):
                prev_w = success_words[left_idx - 1]
                if isinstance(prev_w, dict) and "end" in prev_w:
                    val = float(prev_w["end"])
                    prev_end = val if math.isfinite(val) else None

            if right_idx is not None and 0 <= right_idx < len(success_words) - 1:
                next_w = success_words[right_idx + 1]
                if isinstance(next_w, dict) and "start" in next_w:
                    val = float(next_w["start"])
                    next_start = val if math.isfinite(val) else None

        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to get neighbor times: {e}")

        return prev_end, next_start

    def _calculate_side_extension(
        self, gap: Optional[float], is_left: bool, allow_no_neighbor: bool
    ) -> float:
        """Calculate extension for one side based on gap and configuration."""
        # Marking as used to satisfy linters; current logic is symmetric and does not need these flags.
        _ = is_left
        _ = allow_no_neighbor
        if gap is None:
            # No neighbor timing available (corrupted/missing) → treat as absent neighbor and apply no-neighbor padding
            return max(0.0, self._config.max_expand_s + self._config.pad_s)

        # Validate gap is a valid number
        try:
            gap = float(gap)
        except (ValueError, TypeError):
            logger.warning(f"Invalid gap value: {gap}, treating as no extension")
            return 0.0

        if gap <= 0:
            # Overlapping or touching: no extension
            return 0.0

        if gap > self._config.max_gap_s:
            # Gap too large: no bridging
            return 0.0

        # Small gap: extend up to available space, capped by standard max_expand
        return min(self._config.max_expand_s, gap)

    def _advance_word_index(
        self,
        group: List[Dict[str, Any]],
        success_words: List[Dict[str, Any]],
        current_index: int,
    ) -> int:
        """Advance word index based on last matched item position."""
        if not group:
            return current_index

        try:
            # Prefer identity lookups via cache; fallback to equality scan
            indices: List[int] = []
            for item in group:
                idx_id = (
                    self._success_index_map.get(id(item))
                    if self._success_index_map
                    else None
                )
                if idx_id is not None:
                    indices.append(idx_id)
                    continue
                # Fallback path (should be rare)
                try:
                    idx_fallback = success_words.index(item)
                    indices.append(idx_fallback)
                except ValueError:
                    continue
            last_global_idx = max(indices, default=current_index - 1)
            return max(current_index, last_global_idx + 1)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to advance word index: {e}")
            return current_index

    def _normalize_total_duration(
        self, items: List[Dict[str, Any]], success_words: List[Dict[str, Any]]
    ) -> None:
        """Trim added extensions if total overlay duration exceeds budget.

        Budget = sum(success word durations) + 2 * (max_expand_s + pad_s)
        Only trims the extension portion (keeps base word coverage intact),
        proportionally across items based on how much extension each has.
        """
        if not items:
            return

        # Compute success duration total and global bounds
        success_total = 0.0
        global_first_start = None
        global_last_end = None
        for w in success_words:
            try:
                if isinstance(w, dict) and "start" in w and "end" in w:
                    s = float(w["start"])
                    e = float(w["end"])
                    if math.isfinite(s) and math.isfinite(e) and e > s:
                        success_total += e - s
                        global_first_start = (
                            s
                            if global_first_start is None
                            else min(global_first_start, s)
                        )
                        global_last_end = (
                            e if global_last_end is None else max(global_last_end, e)
                        )
            except Exception:
                continue

        cap_each_side = self._cap_each_side()
        budget = success_total + 2 * cap_each_side

        def snap_to_base(it: Dict[str, Any]) -> None:
            start = float(it.get("start_time", 0.0))
            end = start + float(it.get("duration", 0.0))
            base_start = float(it.get("_base_start", start))
            base_end = float(it.get("_base_end", end))
            new_start = max(start, base_start)
            new_end = min(end, base_end)
            if new_end <= new_start:
                new_end = new_start + self._config.min_duration_s
            it["start_time"] = new_start
            it["duration"] = max(0.0, new_end - new_start)

        def cap_extreme_left(it: Dict[str, Any], cap: float) -> None:
            start = float(it.get("start_time", 0.0))
            end = start + float(it.get("duration", 0.0))
            # Enforce start_time >= global_first_start - cap
            base_start_global = (
                float(global_first_start) if global_first_start is not None else start
            )
            min_allowed_start = max(0.0, base_start_global - cap)
            new_start = max(start, min_allowed_start)
            if new_start > start:
                it["start_time"] = new_start
                it["duration"] = max(0.0, end - new_start)

        def cap_extreme_right(it: Dict[str, Any], cap: float) -> None:
            start = float(it.get("start_time", 0.0))
            end = start + float(it.get("duration", 0.0))
            # Enforce end_time <= global_last_end + cap
            base_end_global = (
                float(global_last_end) if global_last_end is not None else end
            )
            # Epsilon-safe right bound to avoid minor FP overshoot
            eps = 1e-6
            max_allowed_end = (base_end_global + cap) - eps
            new_end = min(end, max_allowed_end)
            if new_end < end:
                it["duration"] = max(0.0, new_end - start)

        # Always cap each item to global timeline window first, then also cap extremes
        if items:
            # Per-item clamp to [global_first_start - cap, global_last_end + cap]
            min_window_start = (
                max(0.0, float(global_first_start) - cap_each_side)
                if global_first_start is not None
                else 0.0
            )
            # Epsilon-safe right bound
            eps = 1e-6
            max_window_end = (
                (float(global_last_end) + cap_each_side - eps)
                if global_last_end is not None
                else None
            )
            for it in items:
                try:
                    start = float(it.get("start_time", 0.0))
                    dur = float(it.get("duration", 0.0))
                    end = start + dur
                except (ValueError, TypeError):
                    continue
                # Left clamp
                if start < min_window_start:
                    start = min_window_start
                # Right clamp
                if max_window_end is not None and end > max_window_end:
                    end = max_window_end
                if end <= start:
                    end = start + self._config.min_duration_s
                it["start_time"] = start
                it["duration"] = max(0.0, end - start)

            # Additionally, cap extremes for idempotency (no-ops if already within window)
            cap_extreme_left(items[0], cap_each_side)
            cap_extreme_right(items[-1], cap_each_side)

        # 1) If over budget after extreme capping, remove all extensions on interior items
        current_total = sum(float(it.get("duration", 0.0)) for it in items)
        excess = current_total - budget
        if excess <= 1e-6:
            return

        if len(items) > 2:
            for it in items[1:-1]:
                snap_to_base(it)

        # 2) Re-apply extreme capping after snapping interiors (idempotent safeguard)
        if items:
            cap_extreme_left(items[0], cap_each_side)
            cap_extreme_right(items[-1], cap_each_side)

        # 3) If still over budget, trim from extremes evenly while respecting min durations
        def total_duration() -> float:
            return sum(float(it.get("duration", 0.0)) for it in items)

        remaining_excess = total_duration() - budget
        if remaining_excess <= 1e-6:
            return

        # Trim loop (bounded iterations)
        for _ in range(10):
            if remaining_excess <= 1e-6:
                break
            # Trim half from left, half from right
            take = remaining_excess / 2.0

            # Left
            if items:
                it = items[0]
                start = float(it.get("start_time", 0.0))
                dur = float(it.get("duration", 0.0))
                max_move = dur - self._config.min_duration_s
                move = min(take, max(0.0, max_move))
                it["start_time"] = start + move
                it["duration"] = max(self._config.min_duration_s, dur - move)

            # Right
            if items:
                it = items[-1]
                start = float(it.get("start_time", 0.0))
                dur = float(it.get("duration", 0.0))
                max_move = dur - self._config.min_duration_s
                move = min(take, max(0.0, max_move))
                it["duration"] = max(self._config.min_duration_s, dur - move)

            remaining_excess = total_duration() - budget

    def _postprocess_items(self, items: List[Dict[str, Any]]) -> None:
        """Preserve order, remove overlaps, and normalize word_count.

        - Preserve the existing list order (transcript order)
        - Enforce non-overlap by clamping each start_time >= previous end
        - Ensure minimum duration for readability
        - Recompute word_count as number of normalized tokens from the text
        """
        if not items:
            return

        # Enforce non-overlap and min duration
        prev_end = None
        for it in items:
            try:
                start = float(it.get("start_time", 0.0))
                dur = float(it.get("duration", 0.0))
            except (ValueError, TypeError):
                start = (
                    float(it.get("start_time", 0.0))
                    if isinstance(it.get("start_time"), (int, float))
                    else 0.0
                )
                dur = (
                    float(it.get("duration", 0.0))
                    if isinstance(it.get("duration"), (int, float))
                    else 0.0
                )

            if prev_end is not None and start < prev_end:
                # Shift start to prev_end to remove overlap
                start = prev_end

            # Ensure minimum duration
            dur = max(dur, self._config.min_duration_s)

            it["start_time"] = max(0.0, start)
            it["duration"] = max(self._config.min_duration_s, dur)

            prev_end = it["start_time"] + it["duration"]

        # Recompute word_count from normalized text tokens for consistency
        for it in items:
            try:
                text = str(it.get("text", ""))
                it["word_count"] = len(normalize_text(text))
            except Exception:
                # Leave existing word_count if normalization fails
                pass

        # Enforce reasonable duration for multi-word lines (>=4 tokens) if configured
        min_req = float(getattr(self._config, "min_multiword_duration_s", 0.0) or 0.0)
        if min_req > 0:
            for it in items:
                try:
                    tokens = int(it.get("word_count", 0))
                    if tokens >= 4:
                        dur = float(it.get("duration", 0.0))
                        if dur < min_req:
                            it["duration"] = min_req
                except Exception:
                    continue

    def _cap_each_side(self) -> float:
        """Helper: maximum allowed combined extension on a single extreme side."""
        return max(0.0, self._config.max_expand_s + self._config.pad_s)

    def _persist_output(
        self, text_over_items: List[Dict[str, Any]], text_over_id: str
    ) -> None:
        """Persist text overlay items to JSON file."""
        try:
            # Remove internal metadata before persistence
            clean_items = [
                {k: v for k, v in item.items() if not k.startswith("_")}
                for item in text_over_items
            ]

            target_dir = Path(self._temp_dir) / text_over_id
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / "text_over.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(clean_items, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Persisted {len(clean_items)} text overlay items to {out_path}"
            )
        except Exception as e:
            logger.error(f"Failed to persist text overlay items: {e}")
            # Do not fail pipeline on persistence errors
