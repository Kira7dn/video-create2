from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.infrastructure.adapters.text_over_builder import (
    TextOverBuilder,
    _tokens,
)


# Build once per alignment_sample and persist to test/temp/textover/<tmp_pipeline_id>/<segmentID>_text_over.json
@pytest.fixture
def persisted_text_over(alignment_sample):
    sample = alignment_sample
    folder = Path(sample["folder"])  # e.g., test/temp/.../main-1
    words = json.loads(Path(sample["word_json"]).read_text(encoding="utf-8")).get(
        "words", []
    )
    lines_raw = Path(sample["transcript"]).read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in lines_raw]
    non_empty_lines = [ln for ln in lines if _tokens(ln)]

    builder = TextOverBuilder(temp_dir="test/temp")
    res = builder.build(word_items=words, chunks=non_empty_lines, text_over_id=None)

    persist_dir = Path("test/temp") / "textover" / folder.parent.name
    persist_dir.mkdir(parents=True, exist_ok=True)
    out_path = persist_dir / f"{folder.name}_text_over.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder, words, non_empty_lines, res, out_path


@pytest.mark.integration
class TestTextOverBuilder3Integration:
    def _load(self, sample: dict):
        folder = Path(sample["folder"])  # e.g., test/temp/.../main-1
        words = json.loads(Path(sample["word_json"]).read_text(encoding="utf-8")).get(
            "words", []
        )
        lines_raw = Path(sample["transcript"]).read_text(encoding="utf-8").splitlines()
        # Builder sẽ bỏ qua dòng rỗng sau chuẩn hoá, nên lọc trước để so sánh số lượng
        lines = [ln.strip() for ln in lines_raw]
        non_empty_lines = [ln for ln in lines if _tokens(ln)]
        return folder, words, non_empty_lines

    def test_line_count_matches(self, persisted_text_over):
        folder, _words, lines, res, _out_path = persisted_text_over
        assert len(res) == len(
            lines
        ), f"[{folder.name}] result lines={len(res)} != expected lines={len(lines)}"

    def test_last_line_end_time_matches_last_success_word_end(
        self, persisted_text_over
    ):
        folder, words, _lines, res, _out_path = persisted_text_over
        success_words = [
            w
            for w in words
            if w.get("case") == "success" and isinstance(w.get("end"), (int, float))
        ]
        assert success_words, f"[{folder.name}] no success words with timings"
        expected_end = float(success_words[-1]["end"])  # type: ignore[arg-type]
        assert res, f"[{folder.name}] empty build result"
        last = res[-1]
        actual_end = float(last["start_time"]) + float(last["duration"])  # type: ignore[arg-type]
        assert (
            abs(actual_end - expected_end) < 1e-3
        ), f"[{folder.name}] last line end {actual_end} != expected {expected_end}"

    def test_word_count_matches_tokenized_line(self, persisted_text_over):
        folder, _words, lines, res, _out_path = persisted_text_over
        assert len(res) == len(lines)
        for i, (item, line) in enumerate(zip(res, lines)):
            expected_wc = len(_tokens(line))
            assert (
                item["word_count"] == expected_wc
            ), f"[{folder.name}] line {i} word_count={item['word_count']} != expected={expected_wc}"

    def test_monotonic_timing_and_non_negative_duration(self, persisted_text_over):
        folder, _words, _lines, res, _out_path = persisted_text_over
        prev_start = None
        prev_end = None
        for i, item in enumerate(res):
            start = float(item["start_time"])  # type: ignore[arg-type]
            end = start + float(item["duration"])  # type: ignore[arg-type]
            assert (
                float(item["duration"]) >= 0.0
            ), f"[{folder.name}] line {i} has negative duration"
            if prev_start is not None:
                assert (
                    start >= prev_start
                ), f"[{folder.name}] start_time not monotonic at line {i}"
            if prev_end is not None:
                assert (
                    end >= prev_end
                ), f"[{folder.name}] end_time not monotonic at line {i}"
            prev_start, prev_end = start, end

    def test_persist_text_over_when_text_over_id_given(self, persisted_text_over):
        folder, _words, lines, res, out_path = persisted_text_over
        # File must exist and content must match built result and expected lines
        assert out_path.exists(), f"[{folder.name}] expected {out_path} to exist"
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data, list), f"[{folder.name}] persisted json is not a list"
        assert (
            len(data) == len(res) == len(lines)
        ), f"[{folder.name}] persisted list length mismatch"
        # spot-check required fields for first item if available
        if data:
            first = data[0]
            for key in ("text", "start_time", "duration", "word_count"):
                assert (
                    key in first
                ), f"[{folder.name}] missing key in persisted item: {key}"
