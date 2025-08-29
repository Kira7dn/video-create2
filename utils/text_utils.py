"""
Các hàm tiện ích để tạo và quản lý text overlay.

Module này chứa các hàm hỗ trợ tạo text overlay với timing chính xác,
xử lý duration và các thao tác liên quan đến hiển thị văn bản.

"""

from typing import Dict, List, Optional
import logging
import re
import time
from pydantic_ai import Agent
from pydantic import BaseModel, field_validator


logger = logging.getLogger(__name__)


def create_text_overlay(
    text: str, start_time: float, duration: float, **kwargs
) -> Dict:
    """
    Tạo một text overlay item với các thông số cơ bản.

    Args:
        text: Nội dung văn bản
        start_time: Thời gian bắt đầu (giây)
        duration: Thời lượng hiển thị (giây)
        **kwargs: Các thông số bổ sung

    Returns:
        Dict: Đối tượng text overlay
    """
    return {
        "text": text,
        "start_time": max(0, start_time),  # Đảm bảo không âm
        "duration": max(0.1, duration),  # Đảm bảo duration tối thiểu 0.1s
        **kwargs,
    }


def merge_consecutive_overlays(
    overlays: List[Dict], max_gap: float = 0.5
) -> List[Dict]:
    """
    Hợp nhất các text overlay liên tiếp gần nhau.

    Args:
        overlays: Danh sách các overlay
        max_gap: Khoảng cách tối đa (giây) để coi là liên tiếp

    Returns:
        List[Dict]: Danh sách các overlay đã được hợp nhất
    """
    if not overlays:
        return []

    # Sắp xếp theo thời gian bắt đầu
    sorted_overlays = sorted(overlays, key=lambda x: x["start_time"])
    merged = [sorted_overlays[0]]

    for current in sorted_overlays[1:]:
        last = merged[-1]

        # Tính khoảng cách giữa overlay cuối và hiện tại
        gap = current["start_time"] - (last["start_time"] + last["duration"])

        # Nếu khoảng cách nhỏ hơn ngưỡng, hợp nhất
        if gap <= max_gap and (
            last["text"].endswith((".", "!", "?"))
            or current["text"].startswith((" ", "\n"))
        ):
            last["text"] += " " + current["text"].lstrip()
            last["duration"] = (current["start_time"] + current["duration"]) - last[
                "start_time"
            ]
        else:
            merged.append(current)

    return merged


def validate_segments(v: List[str]) -> List[str]:
    """Pass-through validation to satisfy tests.

    - Preserve original segments and whitespace
    - Do not auto-split or normalize
    - Raise ValueError("string_type") if any item is not a string
    """
    if v is None:
        return []

    if not isinstance(v, list):
        raise ValueError("segments must be a list")

    for item in v:
        if not isinstance(item, str):
            # Tests expect ValueError containing 'string_type'
            raise ValueError("string_type: segment must be a string")

    return v


class TranscriptSegments(BaseModel):
    """Pydantic model cho validated transcript segments"""

    segments: List

    @field_validator("segments", mode="after")
    @classmethod
    def _validate_segments(cls, v: List) -> List[str]:
        return validate_segments(v)


async def split_transcript(content: str) -> List[str]:
    """Split transcript into natural segments using LLM.

    Args:
        content: The transcript content to split.

    Returns:
        List of segmented transcript parts.

    Raises:
        ValidationError: If there's an error processing the LLM response.
    """
    start_time = time.time()

    agent = Agent(
        "openai:gpt-4o",
        output_type=TranscriptSegments,
        system_prompt="""You are a natural language processing expert.
        Your task is to segment the transcript into short, natural sentences.
        Each sentence must be a complete semantic unit, readable and natural.
        CRITICAL: You MUST preserve EVERY SINGLE WORD from the original transcript.
        DO NOT paraphrase, summarize, or change ANY words.
        """,
        model_settings={"temperature": 0.1},
    )
    prompt = f"""
        CRITICAL: You MUST preserve EVERY SINGLE WORD from the original transcript.
        DO NOT paraphrase, summarize, or change ANY words. Your job is ONLY to split the text.

        Original transcript (PRESERVE EXACTLY):
        "{content}"

        REQUIREMENTS:
        1. PRESERVE ALL CONTENT - Every word must be included exactly as written
        2. Each segment: 4-12 words (readable chunks)
        3. Maximum 80 characters per segment (screen readability)
        4. Break at natural phrase boundaries
        5. Keep related concepts together
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
    result = await agent.run(user_prompt=prompt)

    # Process the result
    transcript_segments = result.output
    duration = time.time() - start_time

    # Validate content preservation
    if not _validate_content_preservation(content, transcript_segments.segments):
        logger.warning(
            "LLM output failed content preservation check. Using fallback method."
        )
        return _fallback_split(content)

    logger.debug(
        "Completed transcript segmentation in %.2f seconds, created %d segments",
        duration,
        len(transcript_segments.segments),
    )

    return transcript_segments.segments


def _validate_content_preservation(original: str, segments: List[str]) -> bool:
    """Validate that LLM output preserves all content from original transcript.

    Enforces strict preservation: same sequence of word tokens (case-insensitive, punctuation-insensitive).

    Args:
        original: Original transcript content
        segments: List of segmented transcript parts

    Returns:
        bool: True if content is strictly preserved, False otherwise
    """
    # Tokenize to word sequences
    original_tokens = re.findall(r"\b\w+\b", original.lower())
    segmented_tokens: List[str] = []
    for segment in segments:
        segmented_tokens.extend(re.findall(r"\b\w+\b", segment.lower()))

    # Allow empty original
    if not original_tokens:
        # If original has no tokens, consider preserved when segments also have no tokens
        return len(segmented_tokens) == 0

    if not segments:
        return False

    # Strict sequence equality
    if original_tokens != segmented_tokens:
        # Compute simple diagnostics for logging
        missing = set(original_tokens) - set(segmented_tokens)
        extra = set(segmented_tokens) - set(original_tokens)
        logger.warning(
            "Content preservation failed: tokens mismatch. Missing: %s | Extra: %s",
            missing,
            extra,
        )
        return False

    return True


def _fallback_split(content: str) -> List[str]:
    """Improved fallback method for splitting transcript when LLM fails.

    Args:
        content: The transcript content to split.

    Returns:
        List of split segments with better content preservation.
    """
    if not content.strip():
        return []

    # More comprehensive splitting patterns
    sentence_enders = r"(?<=[.!?])\s+"
    comma_separators = r"(?<=,)\s+(?=\w)"
    conjunctions = r"\s+(?=(?:and|or|but|so|because|when|if|while|although|however|therefore|moreover)\s+)"
    natural_pauses = (
        r"\s+(?=(?:now|then|next|first|second|finally|meanwhile|additionally)\s+)"
    )

    # Combine all patterns
    pattern = f"{sentence_enders}|{comma_separators}|{conjunctions}|{natural_pauses}"

    # Split and clean up whitespace
    segments = [
        s.strip() for s in re.split(pattern, content, flags=re.IGNORECASE) if s.strip()
    ]

    # Process segments to ensure good readability
    result = []
    for segment in segments:
        words = segment.split()

        # If segment is good size (4-12 words, ≤80 chars), keep it
        if 4 <= len(words) <= 12 and len(segment) <= 80:
            result.append(segment)
        elif len(words) <= 3:
            # Very short segments - try to combine with previous
            if result and len(result[-1].split()) + len(words) <= 12:
                result[-1] = result[-1] + " " + segment
            else:
                result.append(segment)
        else:
            # Long segments - split more carefully
            for i in range(0, len(words), 8):  # Larger chunks than before
                chunk_words = words[i : i + 8]
                if chunk_words:
                    chunk = " ".join(chunk_words)
                    result.append(chunk)

    # Final validation - ensure no empty segments
    result = [s for s in result if s.strip()]

    # If we still have no segments, create one from the original content
    if not result and content.strip():
        # Split the entire content into reasonable chunks
        words = content.split()
        for i in range(0, len(words), 8):
            chunk_words = words[i : i + 8]
            if chunk_words:
                result.append(" ".join(chunk_words))

    logger.debug(f"Fallback split created {len(result)} segments from original content")
    return result


def create_text_over_item(text: str, word_items: List[Dict]) -> Optional[Dict]:
    """
    Tạo text_over item từ danh sách từ.

    Args:
        text: Nội dung văn bản
        word_items: Danh sách các từ với thông tin timing

    Returns:
        Optional[Dict]: Dictionary chứa thông tin text overlay hoặc None nếu không hợp lệ
    """
    if not word_items or not text.strip():
        return None

    # Lọc ra các từ có thông tin timing hợp lệ
    valid_words = [
        w for w in word_items if isinstance(w, dict) and "start" in w and "end" in w
    ]

    if not valid_words:
        return None

    # Tính toán thời gian bắt đầu và kết thúc (dựa theo span tự nhiên của nhóm từ)
    start_time = min(w["start"] for w in valid_words)
    end_time = max(w["end"] for w in valid_words)
    duration = max(0.1, end_time - start_time)

    return {
        "text": text,
        "start_time": start_time,
        "duration": duration,
        "word_count": len(valid_words),
    }


def normalize_text(text: str) -> List[str]:
    """
    Chuẩn hóa văn bản để phù hợp với tokenization của Gentle.

    Args:
        text: Văn bản cần chuẩn hóa

    Returns:
        List[str]: Danh sách các từ đã được chuẩn hóa
    """
    # Tách từ và loại bỏ dấu câu
    words = re.findall(r"\b\w+\b", text.lower())
    return words
