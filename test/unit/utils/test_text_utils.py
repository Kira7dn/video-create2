import pytest

import utils.text_utils as tu


def test_create_text_overlay_clamps_and_kwargs():
    o = tu.create_text_overlay("hi", start_time=-2.0, duration=0.0, pos=(10, 20))
    assert o["start_time"] == 0
    assert o["duration"] >= 0.1
    assert o["text"] == "hi"
    assert o["pos"] == (10, 20)


def test_merge_consecutive_overlays_merge_and_no_merge():
    overlays = [
        {"text": "Hello.", "start_time": 0.0, "duration": 1.0},
        {"text": " world", "start_time": 1.3, "duration": 0.7},  # gap 0.3 -> merge
        {"text": "New", "start_time": 3.0, "duration": 0.5},      # far -> new
    ]
    merged = tu.merge_consecutive_overlays(overlays, max_gap=0.5)
    assert len(merged) == 2
    assert merged[0]["text"].startswith("Hello.")
    assert merged[0]["duration"] > 1.9  # spans first two

    # Non-merge case due to punctuation/space condition not met
    overlays2 = [
        {"text": "Hello", "start_time": 0.0, "duration": 1.0},
        {"text": "World", "start_time": 1.2, "duration": 0.5},
    ]
    merged2 = tu.merge_consecutive_overlays(overlays2, max_gap=0.5)
    assert len(merged2) == 2


def test_validate_segments_behaviors():
    assert tu.validate_segments(None) == []
    with pytest.raises(ValueError):
        tu.validate_segments("not a list")
    with pytest.raises(ValueError) as e:
        tu.validate_segments(["ok", 1])
    assert "string_type" in str(e.value)
    data = [" a  b ", "c"]
    assert tu.validate_segments(data) is data  # pass-through


def test_transcript_segments_model_uses_validator():
    m = tu.TranscriptSegments(segments=["x", "y"])  # should pass
    assert m.segments == ["x", "y"]
    with pytest.raises(ValueError):
        tu.TranscriptSegments(segments=["x", 2])


def test__validate_content_preservation_cases():
    orig = "Hello, world!"
    segs_ok = ["Hello", "world"]
    assert tu._validate_content_preservation(orig, segs_ok) is True

    assert tu._validate_content_preservation("", []) is True
    assert tu._validate_content_preservation("one", []) is False

    segs_bad = ["Hello", "there"]
    assert tu._validate_content_preservation(orig, segs_bad) is False


def test__fallback_split_rules():
    assert tu._fallback_split("") == []

    # Should split natural pauses and keep sizes reasonable
    content = "Now we test, splitting this long sentence because it is quite lengthy and should be chunked accordingly."
    out = tu._fallback_split(content)
    assert isinstance(out, list) and out
    assert all(len(s.strip()) > 0 for s in out)

    # Very short segments should combine when possible
    content2 = "Hi all. Go now."
    out2 = tu._fallback_split(content2)
    assert any(len(s.split()) >= 4 for s in out2)


@pytest.mark.asyncio
async def test_split_transcript_happy_and_fallback(monkeypatch):
    # Mock Agent to avoid external call
    class DummyResult:
        def __init__(self, segments):
            class O:
                def __init__(self, segs):
                    self.segments = segs
            self.output = O(segments)

    class DummyAgent:
        def __init__(self, *a, **k):
            pass

        async def run(self, user_prompt: str):
            # decide content based on prompt presence for coverage
            return DummyResult(["preserve", "content"])  # placeholder segments

    monkeypatch.setattr(tu, "Agent", DummyAgent)

    # Case 1: preserved content
    content_ok = "preserve content"
    out_ok = await tu.split_transcript(content_ok)
    assert out_ok == ["preserve", "content"]

    # Case 2: triggers fallback when preservation fails
    async def run_bad(self, user_prompt: str):
        return DummyResult(["changed", "words"])  # mismatch

    monkeypatch.setattr(DummyAgent, "run", run_bad)
    content_bad = "original words stay"
    out_bad = await tu.split_transcript(content_bad)
    # Fallback produces non-empty reasonable chunks
    assert isinstance(out_bad, list) and out_bad
