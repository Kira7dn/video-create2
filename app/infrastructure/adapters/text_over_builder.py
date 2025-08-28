from __future__ import annotations

from typing import Any, Dict, List
import json
from pathlib import Path

from app.application.interfaces import ITextOverBuilder
from app.core.config import settings
from utils.gentle_utils import filter_successful_words
from utils.alignment_utils import find_greedy_shift_match
from utils.text_utils import create_text_over_item, normalize_text


class TextOverBuilder(ITextOverBuilder):
    """Default implementation to build text_over items.

    - If alignment words available, tries to map chunks to aligned words using greedy shift matching.
    - If mapping fails or when no alignment, synthesizes fallback timing based on word count.
    """

    def __init__(self, *, temp_dir: str) -> None:
        self._temp_dir = temp_dir

    def build(
        self,
        *,
        word_items: List[Dict[str, Any]],
        chunks: List[str],
        text_over_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        text_over_items: List[Dict[str, Any]] = []

        cur_t = 0.0
        if word_items:
            success_words = filter_successful_words(word_items)
            word_index = 0
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
                    # Synthesize fallback for this chunk
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
            # No alignment → simple synthesized items
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

        # Optionally persist after building
        if text_over_id:
            try:
                target_dir = Path(self._temp_dir) / text_over_id
                target_dir.mkdir(parents=True, exist_ok=True)
                out_path = target_dir / "text_over.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(text_over_items, f, ensure_ascii=False)
            except Exception:
                # Do not fail pipeline on persistence errors
                pass

        return text_over_items
