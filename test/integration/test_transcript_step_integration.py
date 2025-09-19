"""
Integration tests for transcript utilities.

Covers:
- Fallback-based transcript segmentation that preserves content
- Creating text_over items from aligned words
- Verifying Gentle alignment quality (no external calls)
"""

import logging
import os
from pathlib import Path

import pytest

from app.application.pipeline.base import PipelineContext
from app.application.pipeline.video.steps.transcription_align import (
    TranscriptionAlignStep,
)
from app.infrastructure.adapters.gentle_transcription_aligner import (
    GentleTranscriptionAligner,
)
from app.infrastructure.adapters.text_over_builder import TextOverBuilder
from utils.text_utils import (
    _fallback_split,  # type: ignore
    _validate_content_preservation,  # type: ignore
    create_text_over_item,
)
from utils.gentle_utils import verify_alignment_quality

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration


# Test data
SAMPLE_TRANSCRIPT = "Hello world this is a test audio content for transcription"
TEST_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "input_sample.json")
)


@pytest.fixture
def sample_transcript():
    return "Hello world this is a test audio content for transcription using utilities"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fallback_split_preserves_words(sample_transcript):
    content = sample_transcript
    segments = _fallback_split(content)
    assert isinstance(segments, list) and segments
    # Each segment should be readable
    for seg in segments:
        words = seg.split()
        assert 1 <= len(words) <= 12
        assert len(seg) <= 120
    # Preserve original words (order-insensitive strict token equality)
    assert _validate_content_preservation(content, segments)


@pytest.mark.integration
def test_create_text_over_item_from_words():
    # Simulate aligned words from Gentle
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.4, "case": "success"},
        {"word": "world", "start": 0.5, "end": 0.9, "case": "success"},
        {"word": "this", "start": 1.0, "end": 1.2, "case": "success"},
    ]
    text = "Hello world this"
    item = create_text_over_item(text, words)
    assert item is not None
    assert item["text"] == text
    assert item["start_time"] == pytest.approx(0.0, abs=1e-6)
    assert item["duration"] == pytest.approx(1.2 - 0.0, rel=1e-6)
    assert item["word_count"] == 3


@pytest.mark.integration
def test_verify_alignment_quality_stats():
    # 4 words, 3 successes → 0.75 success_ratio
    word_items = [
        {"word": "hello", "case": "success"},
        {"word": "world", "case": "success"},
        {"word": "test", "case": "not-found"},
        {"word": "audio", "case": "success"},
    ]
    result = verify_alignment_quality(word_items, min_success_ratio=0.7)
    assert result["is_verified"] is True
    assert result["success_ratio"] == pytest.approx(0.75)
    assert result["success_count"] == 3
    assert result["total_words"] == 4


@pytest.mark.integration
def test_validate_content_preservation_edge_cases():
    # Empty original
    assert _validate_content_preservation("", []) is True
    # Non-empty original but empty segments → False
    assert _validate_content_preservation("hello world", []) is False
    # Exact token sequence
    assert _validate_content_preservation("Hello, world!", ["Hello", "world"]) is True


@pytest.mark.integration
def test_tokenization_with_punctuation():
    content = "Hello, world! This is: a test? yes; indeed."
    segments = _fallback_split(content)
    assert _validate_content_preservation(content, segments)


@pytest.mark.integration
def test_alignment_quality_thresholds():
    # Only 1/4 success → fail for min_success_ratio=0.5
    word_items = [
        {"word": "one", "case": "success"},
        {"word": "two", "case": "not-found"},
        {"word": "three", "case": "not-found"},
        {"word": "four", "case": "not-found"},
    ]
    result = verify_alignment_quality(word_items, min_success_ratio=0.5)
    assert result["is_verified"] is False


@pytest.mark.integration
def test_create_text_over_item_invalid():
    assert create_text_over_item("text", []) is None
    assert create_text_over_item("", [{"start": 0, "end": 1}]) is None


@pytest.mark.integration
def test_fallback_split_minimal_cases():
    assert _fallback_split("") == []
    assert isinstance(_fallback_split("hello world test case"), list)


@pytest.mark.integration
def test_fallback_split_constraints(sample_transcript):
    segments = _fallback_split(sample_transcript)
    assert segments
    for seg in segments:
        words = seg.split()
        assert len(seg) <= 80 or len(words) <= 12


@pytest.mark.integration
def test_fallback_output_constraints(sample_transcript):
    segments = _fallback_split(sample_transcript)
    assert segments
    for seg in segments:
        words = seg.split()
        assert 1 <= len(words)  # allow small chunks in fallback
        assert len(seg) <= 120


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transcription_align_step_with_gentle_env(tmp_path):
    """Integration: run TranscriptionAlignStep against Gentle using files in test/source/.

    Auto-discovers pairs where an audio file (.wav/.mp3) has a matching .txt with the same stem.
    """
    source_dir = Path(__file__).resolve().parent.parent / "source"
    if not source_dir.exists():
        pytest.skip("test/source directory not found")

    audio_exts = {".wav", ".mp3"}
    pairs = []
    for audio in source_dir.rglob("*"):
        if audio.suffix.lower() in audio_exts:
            txt = audio.with_suffix(".txt")
            if txt.exists():
                pairs.append((audio, txt))

    if not pairs:
        pytest.skip(
            "No audio+transcript pairs found in test/source (expect .wav/.mp3 with .txt)"
        )

    ap, tp = pairs[0]
    content = tp.read_text(encoding="utf-8").strip()
    assert content, "Transcript content should not be empty"

    ctx = PipelineContext(input={})
    ctx.set(
        "segments",
        [
            {
                "id": "seg1",
                "voice_over": {
                    "local_path": str(ap),
                    "content": content,
                },
            }
        ],
    )

    # Use real adapter; step will record alignment_stats even if alignment fails
    aligner = GentleTranscriptionAligner(temp_dir=str(tmp_path))

    # Minimal splitter that returns the whole content as one chunk
    class _Splitter:
        async def split(self, content: str, seg_id=None):  # noqa: ANN001
            return [content] if content else []

    builder = TextOverBuilder(temp_dir=str(tmp_path))
    step = TranscriptionAlignStep(
        aligner=aligner, splitter=_Splitter(), builder=builder
    )
    await step.run(ctx)

    processed = ctx.get("segments")
    assert isinstance(processed, list) and processed
    seg = processed[0]
    assert isinstance(seg.get("text_over"), list)
    assert len(seg["text_over"]) > 0

    # Validate shape of items
    for item in seg["text_over"]:
        assert set(["text", "start_time", "duration", "word_count"]).issubset(item)
        assert isinstance(item["text"], str) and item["text"].strip()
        assert item["start_time"] >= 0
        assert item["duration"] > 0
        assert item["word_count"] >= 1

    # Content preservation check
    joined = [i["text"] for i in seg["text_over"]]
    assert _validate_content_preservation(content, joined)

    # Alignment stats
    stats = ctx.get("alignment_stats")
    assert isinstance(stats, list) and stats
    assert stats[0].get("total_words") is not None
    assert "alignment_issues" in stats[0]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_text_over_timing_monotonic():
    """Ensure text_over timings are non-decreasing and durations positive (core invariant)."""
    # Build a synthetic seg with fallback split
    content = (
        "Hello world this is a test audio content for transcription using utilities"
    )
    ctx = PipelineContext(input={})
    ctx.set(
        "segments",
        [
            {
                "id": "seg1",
                "voice_over": {
                    # no audio → fallback timing should kick in
                    "local_path": None,
                    "content": content,
                },
            }
        ],
    )

    # Provide required adapters: no-op aligner and simple splitter
    class _NoAligner:
        def align(self, *args, **kwargs):
            return [], {
                "is_verified": False,
                "success_ratio": 0.0,
                "success_count": 0,
                "total_words": 0,
            }

    class _Splitter:
        async def split(self, content: str, seg_id=None):  # noqa: ANN001
            return [content] if content else []

    # Minimal local builder to satisfy schema without temp_dir
    class _MinimalBuilder:
        def build(self, *, word_items, chunks, text_over_id=None):  # noqa: ANN001
            items = []
            t = 0.0
            for ch in chunks:
                txt = (ch or "").strip()
                if not txt:
                    continue
                wc = len(txt.split())
                dur = 1.0
                items.append(
                    {"text": txt, "start_time": t, "duration": dur, "word_count": wc}
                )
                t += dur
            return items

    step = TranscriptionAlignStep(
        aligner=_NoAligner(), splitter=_Splitter(), builder=_MinimalBuilder()
    )
    await step.run(ctx)

    seg = ctx.get("segments")[0]
    items = seg.get("text_over", [])
    assert items, "text_over should not be empty in fallback mode"
    last_end = 0.0
    for it in items:
        start = float(it["start_time"])  # type: ignore
        dur = float(it["duration"])  # type: ignore
        assert dur > 0
        assert start >= last_end - 1e-6
        last_end = start + dur
    # also ensure end times are non-decreasing explicitly
    end_times = [float(it["start_time"]) + float(it["duration"]) for it in items]
    assert end_times == sorted(end_times)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fallback_mode_preserves_content_when_no_audio():
    content = "This pipeline should preserve every token in fallback mode"
    ctx = PipelineContext(input={})
    ctx.set(
        "segments",
        [
            {
                "id": "s1",
                "voice_over": {"local_path": None, "content": content},
            }
        ],
    )

    class _NoAligner:
        def align(self, *args, **kwargs):
            return [], {
                "is_verified": False,
                "success_ratio": 0.0,
                "success_count": 0,
                "total_words": 0,
            }

    class _Splitter:
        async def split(self, content: str, seg_id=None):  # noqa: ANN001
            return [content] if content else []

    class _MinimalBuilder:
        def build(self, *, word_items, chunks, text_over_id=None):  # noqa: ANN001
            items = []
            t = 0.0
            for ch in chunks:
                txt = (ch or "").strip()
                if not txt:
                    continue
                wc = len(txt.split())
                dur = 1.0
                items.append(
                    {"text": txt, "start_time": t, "duration": dur, "word_count": wc}
                )
                t += dur
            return items

    step = TranscriptionAlignStep(
        aligner=_NoAligner(), splitter=_Splitter(), builder=_MinimalBuilder()
    )
    await step.run(ctx)
    seg = ctx.get("segments")[0]
    joined = [i["text"] for i in seg.get("text_over", [])]
    assert _validate_content_preservation(content, joined)


@pytest.mark.skip(
    reason="No real Gentle server call in tests; covered via verify_alignment_quality"
)
def test_real_gentle_alignment_accuracy():
    pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_align_all_pairs_in_source_directory(tmp_path):
    """Edge-case sweep: iterate all audio+txt pairs in test/source and ensure step is robust.

    - Works across multiple files
    - Handles varying lengths and punctuation
    - Records alignment_stats even if verification fails
    """
    source_dir = Path(__file__).resolve().parent.parent / "source"
    if not source_dir.exists():
        pytest.skip("test/source directory not found")

    audio_exts = {".wav", ".mp3"}
    pairs = []
    for audio in source_dir.rglob("*"):
        if audio.suffix.lower() in audio_exts:
            txt = audio.with_suffix(".txt")
            if txt.exists():
                pairs.append((audio, txt))

    if not pairs:
        pytest.skip("No audio+transcript pairs found in test/source")

    # Use real aligner + minimal splitter
    aligner = GentleTranscriptionAligner(temp_dir=str(tmp_path))

    class _Splitter:
        async def split(self, content: str, seg_id=None):  # noqa: ANN001
            return [content] if content else []

    for ap, tp in pairs:
        content = tp.read_text(encoding="utf-8").strip()
        ctx = PipelineContext(input={})
        ctx.set(
            "segments",
            [
                {
                    "id": ap.stem,
                    "voice_over": {"local_path": str(ap), "content": content},
                }
            ],
        )

        builder = TextOverBuilder(temp_dir=str(tmp_path))
        step = TranscriptionAlignStep(
            aligner=aligner, splitter=_Splitter(), builder=builder
        )
        await step.run(ctx)

        seg = ctx.get("segments")[0]
        # If transcript empty → no text_over expected; otherwise must have items
        items = seg.get("text_over", [])
        if content:
            assert items, f"text_over should exist for {ap.name}"
            # invariants
            for it in items:
                assert set(["text", "start_time", "duration", "word_count"]).issubset(
                    it
                )
                assert isinstance(it["text"], str)
                assert float(it["duration"]) > 0
            # ensure sorted by start_time
            starts = [float(it["start_time"]) for it in items]
            assert starts == sorted(starts), "text_over must be sorted by start_time"
            # ensure non-overlap and positive durations
            last_end = 0.0
            for it in items:
                start = float(it["start_time"])
                dur = float(it["duration"])
                assert dur > 0
                assert start >= last_end - 1e-6, "text_over items must not overlap"
                last_end = start + dur
            # end times non-decreasing
            end_times = [
                float(it["start_time"]) + float(it["duration"]) for it in items
            ]
            assert end_times == sorted(end_times)
            # content preservation across reconstructed text_over
            joined = [i["text"] for i in items]
            assert _validate_content_preservation(content, joined)
        else:
            assert items == [] or items is None

        # alignment_stats should be present regardless of verification
        stats = ctx.get("alignment_stats")
        assert isinstance(stats, list) and stats


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_present_but_empty_transcript_is_noop(tmp_path):
    """Edge case: audio file exists but transcript is empty → step should not crash and produce no text_over."""
    source_dir = Path(__file__).resolve().parent.parent / "source"
    if not source_dir.exists():
        pytest.skip("test/source directory not found")

    audio_exts = {".wav", ".mp3"}
    audio_files = [p for p in source_dir.rglob("*") if p.suffix.lower() in audio_exts]
    if not audio_files:
        pytest.skip("No audio files found in test/source")
    ap = audio_files[0]

    ctx = PipelineContext(input={})
    ctx.set(
        "segments",
        [
            {
                "id": ap.stem,
                "voice_over": {"local_path": str(ap), "content": ""},
            }
        ],
    )

    aligner = GentleTranscriptionAligner(temp_dir=str(tmp_path))

    class _Splitter:
        async def split(self, content: str, seg_id=None):  # noqa: ANN001
            return [content] if content else []

    builder = TextOverBuilder(temp_dir=str(tmp_path))
    step = TranscriptionAlignStep(
        aligner=aligner, splitter=_Splitter(), builder=builder
    )
    await step.run(ctx)

    seg = ctx.get("segments")[0]
    items = seg.get("text_over", [])
    assert items == [] or items is None
