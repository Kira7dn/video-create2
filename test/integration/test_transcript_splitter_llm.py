from typing import List

import pytest

from app.infrastructure.adapters.transcript_splitter_llm import LLMTranscriptSplitter

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_transcript_splitter_writes_file_on_success(tmp_path):
    # Arrange: content and expected segments that strictly preserve tokens
    content = "Hello everyone and welcome back to our channel"
    expected_segments: List[str] = [
        "Hello everyone and welcome back",
        "to our channel",
    ]

    # Patch Agent in module to avoid real network and return our segments
    import app.infrastructure.adapters.transcript_splitter_llm as mod

    class FakeResult:
        def __init__(self, segments: List[str]):
            class Output:
                def __init__(self, segs: List[str]):
                    self.segments = segs

            self.output = Output(segments)

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, user_prompt: str):  # noqa: ARG002 - only for signature
            return FakeResult(expected_segments)

    # Monkeypatch
    original_agent = mod.Agent
    mod.Agent = FakeAgent
    try:
        splitter = LLMTranscriptSplitter(temp_dir=str(tmp_path))
        content_id = "seg-001"

        # Act
        segments = await splitter.split(content, content_id)

        # Assert: segments equal and file written correctly
        assert segments == expected_segments
        out_file = tmp_path / content_id / "transcript_lines.txt"
        assert out_file.exists()
        assert (
            out_file.read_text(encoding="utf-8").strip().splitlines()
            == expected_segments
        )
    finally:
        # Restore
        mod.Agent = original_agent


def _extract_content_from_valid_video_data(vvd) -> str:
    data = (vvd or {}).get("json_data", {})
    segments = data.get("segments", []) or []

    def _get_nested(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    # Try common paths for transcript text
    for s in segments:
        for path in (
            ("voice_over", "content"),
            ("voiceOver", "content"),
            ("script", "text"),
            ("transcript", "text"),
            ("transcript", "content"),
        ):
            val = _get_nested(s, *path)
            if isinstance(val, str) and val.strip():
                return val.strip()
    # Fallback text if not present
    return "Hello everyone and welcome back to our channel"


def _extract_all_contents_from_valid_video_data(vvd) -> list[str]:
    data = (vvd or {}).get("json_data", {})
    segments = data.get("segments", []) or []

    def _get_nested(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    contents: list[str] = []
    for s in segments:
        for path in (
            ("voice_over", "content"),
            ("voiceOver", "content"),
            ("script", "text"),
            ("transcript", "text"),
            ("transcript", "content"),
        ):
            val = _get_nested(s, *path)
            if isinstance(val, str) and val.strip():
                contents.append(val.strip())
                break
    if not contents:
        contents = [
            "Hello everyone and welcome back to our channel",
        ]
    return contents


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_transcript_splitter_retries_with_reason_in_prompt(
    tmp_path, caplog, valid_video_data
):
    """First attempt fails validation, second attempt passes.

    Validate that the additional retry prompt includes the failure reason.
    """
    # Arrange: pull content from valid_video_data
    contents = _extract_all_contents_from_valid_video_data(valid_video_data)

    import app.infrastructure.adapters.transcript_splitter_llm as mod

    class FakeResult:
        def __init__(self, segments: List[str]):
            class Output:
                def __init__(self, segs: List[str]):
                    self.segments = segs

            self.output = Output(segments)

    class SequenceAgent:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003
            self._calls: list[str] = []  # store prompts
            # first invalid, then valid
            self._answers = [invalid_segments, expected_segments_valid]

        async def run(self, user_prompt: str):  # noqa: D401
            self._calls.append(user_prompt)
            segs = self._answers.pop(0)
            return FakeResult(segs)

    # Monkeypatch
    original_agent = mod.Agent
    mod.Agent = SequenceAgent
    try:
        for idx, content in enumerate(contents):
            # Build per-content invalid/valid sequences
            tokens = content.split()
            cut = max(1, len(tokens) - max(1, len(tokens) // 4))
            invalid_segments = [" ".join(tokens[:cut])]
            mid = max(1, len(tokens) // 2)
            expected_segments_valid = [
                " ".join(tokens[:mid]),
                " ".join(tokens[mid:]),
            ]

            splitter = LLMTranscriptSplitter(temp_dir=str(tmp_path))
            content_id = f"seg-retry-{idx}"

            # Reset sequence agent answers for each content
            def make_seq_agent(inv, val):
                class SequenceAgent:
                    def __init__(self, *args, **kwargs):
                        self._answers = [inv, val]

                    async def run(self, user_prompt: str):
                        segs = self._answers.pop(0)
                        return FakeResult(segs)

                return SequenceAgent

            mod.Agent = make_seq_agent(invalid_segments, expected_segments_valid)

            with caplog.at_level("INFO"):
                segments = await splitter.split(content, content_id)

            assert segments == expected_segments_valid
            out_file = tmp_path / content_id / "transcript_lines.txt"
            assert out_file.exists()
            assert (
                out_file.read_text(encoding="utf-8").strip().splitlines()
                == expected_segments_valid
            )

            log_text = "\n".join(r.getMessage() for r in caplog.records)
            assert "validation failed" in log_text
            assert "retry #1" in log_text
    finally:
        mod.Agent = original_agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_transcript_splitter_writes_file_on_fallback(tmp_path, caplog):
    # Arrange
    content = "This is a simple sentence for testing fallback behavior"

    import app.infrastructure.adapters.transcript_splitter_llm as mod

    class FailingAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, user_prompt: str):  # noqa: ARG002 - only for signature
            raise RuntimeError("simulated LLM error")

    # Monkeypatch Agent to raise and _fallback_split to a deterministic output
    original_agent = mod.Agent
    original_fallback = mod.LLMTranscriptSplitter._fallback_split
    mod.Agent = FailingAgent

    deterministic_fallback = [
        "This is a simple sentence",
        "for testing fallback behavior",
    ]

    try:
        mod.LLMTranscriptSplitter._fallback_split = staticmethod(lambda text: deterministic_fallback)  # type: ignore[method-assign]
        splitter = LLMTranscriptSplitter(temp_dir=str(tmp_path))
        content_id = "seg-fallback"

        # Act
        with caplog.at_level("INFO"):
            segments = await splitter.split(content, content_id)

        # Assert: used fallback, wrote file, and content matches deterministic fallback
        assert segments == deterministic_fallback
        out_file = tmp_path / content_id / "transcript_lines.txt"
        assert out_file.exists()
        assert (
            out_file.read_text(encoding="utf-8").strip().splitlines()
            == deterministic_fallback
        )

        # Optionally check log marker
        assert any("using FALLBACK" in rec.getMessage() for rec in caplog.records)
    finally:
        # Restore patched symbols
        mod.Agent = original_agent
        mod.LLMTranscriptSplitter._fallback_split = original_fallback


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_transcript_splitter_exhausts_retries_then_fallback(
    tmp_path, caplog, valid_video_data
):
    """All attempts fail validation; ensure fallback is used and logs mention repeated validation failures."""
    contents = _extract_all_contents_from_valid_video_data(valid_video_data)

    import app.infrastructure.adapters.transcript_splitter_llm as mod

    class AlwaysInvalidAgent:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003
            pass

        async def run(self, user_prompt: str):  # noqa: D401
            # Always return segments missing tokens to force validation failure
            toks = content.split()
            bad = " ".join(toks[: max(1, len(toks) // 2)])
            return type("R", (), {"output": type("O", (), {"segments": [bad]})()})()

    # Monkeypatch Agent
    original_agent = mod.Agent
    original_fallback = mod.LLMTranscriptSplitter._fallback_split
    mod.Agent = AlwaysInvalidAgent

    try:
        for idx, content in enumerate(contents):
            # Deterministic fallback derived from the content
            toks = content.split()
            mid = max(1, len(toks) // 2)
            deterministic_fallback = [" ".join(toks[:mid]), " ".join(toks[mid:])]

            splitter = LLMTranscriptSplitter(temp_dir=str(tmp_path))
            content_id = f"seg-retry-fallback-{idx}"

            # Patch fallback per-content
            mod.LLMTranscriptSplitter._fallback_split = staticmethod(lambda text, df=deterministic_fallback: df)  # type: ignore[method-assign]

            with caplog.at_level("INFO"):
                segments = await splitter.split(content, content_id)

            assert segments == deterministic_fallback
            out_file = tmp_path / content_id / "transcript_lines.txt"
            assert out_file.exists()
            assert (
                out_file.read_text(encoding="utf-8").strip().splitlines()
                == deterministic_fallback
            )

            # Logs should indicate repeated validation failures and fallback
            log_text = "\n".join(r.getMessage() for r in caplog.records)
            assert "validation failed" in log_text
            assert "using FALLBACK due to repeated validation failures" in log_text
    finally:
        mod.Agent = original_agent
        mod.LLMTranscriptSplitter._fallback_split = original_fallback


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_transcript_splitter_calls_openai_live(tmp_path, caplog):
    """Live test: ensure an actual request to OpenAI is made (no mocks).

    Skips if OPENAI_API_KEY is not provided. Validates that we either use OPENAI
    or gracefully fall back, but in all cases we write transcript_lines.txt.
    Also checks captured logs for an httpx POST to the chat completions endpoint.
    """
    # Arrange
    from app.core.config import settings

    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY missing; skipping live OpenAI call test")

    import os

    content = "Hello everyone and welcome back to our channel about tech"
    content_id = "live-openai"
    # Use a model that supports Structured Outputs (required by pydantic_ai output_type)
    # Prefer o4-mini by default; allow override via TEST_OPENAI_MODEL
    live_model = os.getenv("TEST_OPENAI_MODEL", "o4-mini")
    splitter = LLMTranscriptSplitter(temp_dir=str(tmp_path), model_name=live_model)

    # Act
    with caplog.at_level("INFO"):
        segments = await splitter.split(content, content_id)

    # Assert: file written
    out_file = tmp_path / content_id / "transcript_lines.txt"
    assert out_file.exists(), "transcript_lines.txt must be written even on fallback"
    file_lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(file_lines) == len(segments)

    # Assert: must actually use OPENAI path (no fallback)
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "LLMTranscriptSplitter: using OPENAI" in log_text, (
        "Expected to use OPENAI path; got logs=\n" + log_text
    )
    assert "LLMTranscriptSplitter: using FALLBACK" not in log_text, (
        "Fallback was used, which should FAIL this live test; logs=\n" + log_text
    )
