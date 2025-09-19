"""
Minimal test configuration/fixtures for the new video pipeline.
"""

import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from typing import Generator
from datetime import datetime
import json

import pytest

from app.core.config import settings

# NOTE: Avoid importing pipeline adapter bundle at module import time to prevent
# cascading import errors when running isolated unit tests. We'll import inside
# the fixture with a safe fallback.

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging():
    """Cấu hình logging cho toàn bộ ứng dụng test."""
    # Tạo thư mục logs nếu chưa tồn tại
    log_dir = Path("test/test_output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_run.log"

    # Tạo formatter chung
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Tạo handlers
    file_handler = logging.FileHandler(
        filename=log_file, mode="a", encoding="utf-8"  # Ghi tiếp vào cuối file
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Xóa tất cả các handler hiện có để tránh trùng lặp
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Thêm handlers mới
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Đặt log level cho các thư viện ồn ào
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Đảm bảo các logger con kế thừa cấu hình
    logging.getLogger("app").setLevel(logging.DEBUG)
    logging.getLogger("test").setLevel(logging.DEBUG)

    return log_file


def pytest_configure(config):  # pylint: disable=unused-argument
    """Configure pytest for tests."""
    # Cấu hình logging
    log_file = setup_logging()

    # Bật log debug filtergraph của FFmpeg cho mọi test để dễ chẩn đoán
    try:

        setattr(settings, "debug_ffmpeg_cmd", True)
    except Exception:  # pragma: no cover - best-effort only
        pass

    # Lấy logger cho pytest
    logger = logging.getLogger("pytest")
    logger.info("=" * 80)
    logger.info("BẮT ĐẦU CHẠY TEST")
    logger.info("=" * 80)
    logger.info("Python: %s", os.sys.version)
    logger.info("Thư mục làm việc: %s", os.getcwd())
    logger.info("Log file: %s", log_file)
    logger.info("-" * 80)


@pytest.fixture(autouse=True)
def log_test_name(request):
    """Log test name when test starts and finishes."""
    logger = logging.getLogger(request.node.nodeid)
    logger.info("🚀 Bắt đầu test: %s", request.node.name)
    start_time = datetime.now()

    def log_test_end():
        duration = (datetime.now() - start_time).total_seconds()
        # Kiểm tra nếu có rep_call trước khi truy cập
        has_rep_call = hasattr(request.node, "rep_call")
        if has_rep_call and request.node.rep_call.failed:
            logger.error("❌ Test thất bại sau %.2fs", duration)
        else:
            logger.info("✅ Test hoàn thành sau %.2fs", duration)
        logger.info("-" * 80)

    request.addfinalizer(log_test_end)


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """
    Fixture cung cấp thư mục tạm cho các test case.

    Thư mục sẽ được tạo trong test/temp và tự động xóa sau khi test hoàn thành.
    """
    # Tạo thư mục test/temp nếu chưa tồn tại
    base_temp_dir = Path(__file__).parent / "temp"
    base_temp_dir.mkdir(exist_ok=True, parents=True)

    # Tạo thư mục tạm duy nhất cho mỗi lần chạy test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    temp_dir = base_temp_dir / f"{timestamp}{os.getpid()}"
    temp_dir.mkdir(exist_ok=True, parents=True)

    yield temp_dir

    # Dọn dẹp thư mục tạm sau khi test hoàn thành
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logging.warning(f"Không thể xóa thư mục tạm {temp_dir}: {e}")


# -------------------- New fixtures for Video Pipeline --------------------
@pytest.fixture(scope="session")
def fake_adapters(tmp_path_factory):
    """AsyncMock-based adapters for the video pipeline builder.

    - Assets.download_asset returns a path inside tmp_path
    - Renderer.render writes a small file to output_path
    - Renderer.concat_clips writes a small file to output_path
    - Uploader.upload_file returns a fake public URL
    """

    tmp_path = tmp_path_factory.mktemp("video_pipeline")

    class Downloader:
        async def download_asset(
            self, url: str, kind: str, seg_id: str | None = None, **kwargs
        ):  # noqa: ANN001
            # Simulate returning a local path inside temp dir and ensure the file exists
            p = tmp_path / f"{Path(url).name}"
            p.parent.mkdir(parents=True, exist_ok=True)
            # Create a tiny placeholder if not present
            if not p.exists():
                p.write_text("", encoding="utf-8")
            return str(p)

    class Renderer:

        async def duration(self, input_path: str) -> float:
            # Simulate probing duration based on file extension or name
            if "voice" in str(input_path).lower():
                return 5.25
            elif "video" in str(input_path).lower():
                return 10.5
            else:
                return 2.0

        async def render_segment(
            self,
            segment: dict,
            *,
            seg_id: str,
            canvas_width: int,
            canvas_height: int,
            frame_rate: int,
        ) -> str:
            # Validate core params
            if canvas_width <= 0 or canvas_height <= 0 or frame_rate <= 0:
                raise ValueError("Invalid canvas size or frame rate")

            source_type = segment.get("source_type", "static")
            primary_source = segment.get("primary_source", "")
            transforms = segment.get("transformations", []) or []
            out = Path(f"test/temp/clip_{seg_id}.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)

            # Main file content
            content = (
                f"rendered_{source_type}_"
                f"{canvas_width}x{canvas_height}_{frame_rate}fps_"
                f"{Path(primary_source).name if primary_source else 'none'}"
            )
            out.write_text(content, encoding="utf-8")

            # Meta sidecar for assertions
            meta = {
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
                "frame_rate": frame_rate,
                "source_type": source_type,
                "primary_source": primary_source,
                "transformations_count": len(transforms),
                "has_audio_delay": any(t.get("type") == "audio_delay" for t in transforms),
            }
            (out.with_suffix(".meta.json")).write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            return str(out)

        async def concat_clips(
            self, clip_paths, output_path: str, background_music: dict | None = None
        ) -> str:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Main output file
            content = f"concat_{len(clip_paths)}_clips"
            if background_music:
                content += "_with_bgm"
            output_path.write_text(content, encoding="utf-8")
            
            # Write manifest for assertions
            manifest = {
                "inputs": list(clip_paths),
                "background_music": background_music or None,
            }
            manifest_path = output_path.with_suffix(".concat_manifest.json")
            manifest_path.write_text(
                json.dumps(manifest, indent=2), 
                encoding="utf-8"
            )
            return str(output_path)

    class Uploader:
        async def upload_file(self, path: str, public: bool = True) -> str:
            return f"https://example.com/{Path(path).name}"

    class ImageSearch:
        def search_image(
            self, keywords: str, min_width: int, min_height: int
        ) -> str | None:
            safe_kw = str(keywords).replace(" ", "_") or "bg"
            return f"https://cdn/{safe_kw}.jpg"

    class KeywordAgent:
        async def extract_keywords(
            self,
            content: str,
            *,
            fields: list[str] | None = None,
            max_keywords: int | None = None,
        ) -> list[str]:
            base = list(fields or [])
            extra = [w for w in str(content).split()[: (max_keywords or 3)]]
            return base + extra

    class Aligner:
        def align(self, *, audio_path: str, transcript_text: str, min_success_ratio: float):  # type: ignore[override]
            # Return no matches but a verify dict
            return [], {
                "is_verified": False,
                "success_ratio": 0.0,
                "success_count": 0,
                "total_words": 0,
            }

    class ImageProcessor:
        async def process(
            self, img_path: str, *, target_width: int, target_height: int, **kwargs
        ):  # noqa: ANN001
            return [img_path]

    class Splitter:
        async def split(self, content: str) -> list[str]:
            return [content] if content else []

    class TextOverBuilderStub:
        def build(self, *, word_items, chunks, text_over_id=None):  # noqa: ANN001
            # Return simple synthesized items matching chunks
            items = []
            t = 0.0
            for ch in chunks:
                items.append({"text": ch, "start_time": t, "duration": 1.0})
                t += 1.0
            return items

    # Wrap with AsyncMock where needed
    downloader = Downloader()
    renderer = Renderer()
    uploader = Uploader()
    image_search = ImageSearch()
    keyword_agent = KeywordAgent()
    aligner = Aligner()
    image_processor = ImageProcessor()
    splitter = Splitter()

    # Allow assertion on calls
    downloader.download_asset = AsyncMock(side_effect=downloader.download_asset)  # type: ignore
    renderer.render_segment = AsyncMock(side_effect=renderer.render_segment)  # type: ignore
    renderer.concat_clips = AsyncMock(side_effect=renderer.concat_clips)  # type: ignore
    uploader.upload_file = AsyncMock(side_effect=uploader.upload_file)  # type: ignore

    return SimpleNamespace(
        downloader=downloader,
        renderer=renderer,
        uploader=uploader,
        keyword_agent=keyword_agent,
        image_search=image_search,
        aligner=aligner,
        image_processor=image_processor,
        splitter=splitter,
        text_over_builder=TextOverBuilderStub(),
    )


@pytest.fixture(autouse=True, scope="session")
def set_temp_base_dir():
    """Force temporary directories to be created under test/temp for all tests."""
    base = Path("test/temp")
    base.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP_BASE_DIR"] = str(base)
    yield
    # Keep artifacts for inspection; could clean up if desired


# -------------------- Fixtures: transcript + voice_over pairs --------------------
def _discover_transcript_voice_pairs(base: Path) -> list[dict]:
    """Scan test/temp recursively and collect pairs of transcript_lines.txt and voice_over.mp3.

    Returns a list of dicts with keys: seg_id, folder, transcript, voice, word_json (optional).
    Only includes entries where both transcript and voice exist and transcript is non-empty.
    If a file named 'word.json' exists in the same folder, include its path under 'word_json'.
    """
    results: list[dict] = []
    if not base.exists():
        return results

    for p in base.rglob("transcript_lines.txt"):
        folder = p.parent
        voice = folder / "voice_over.mp3"
        # Accept both 'word.json' and 'words.json'
        word_json_path = folder / "word.json"
        words_json_path = folder / "words.json"
        if not voice.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8").strip()
        except Exception:  # pragma: no cover - guard against read errors
            continue
        if not txt:
            continue
        # seg_id is the last folder name (e.g., hook, intro, main-0)
        seg_id = folder.name
        item = {
            "seg_id": seg_id,
            "folder": str(folder),
            "transcript": str(p),
            "voice": str(voice),
        }
        if word_json_path.exists():
            item["word_json"] = str(word_json_path)
        elif words_json_path.exists():
            item["word_json"] = str(words_json_path)
        results.append(item)
    return results


@pytest.fixture(scope="session")
def transcript_voice_pairs() -> list[dict]:
    """All (transcript_lines.txt, voice_over.mp3[, word.json]) pairs under test/temp/.

    Example item:
    {
      'seg_id': 'hook',
      'folder': 'test/temp/tmp_.../hook',
      'transcript': '.../transcript_lines.txt',
      'voice': '.../voice_over.mp3',
      'word_json': '.../word.json'  # optional
    }
    """
    base = Path("test/temp")
    return _discover_transcript_voice_pairs(base)


def _discover_alignment_pairs(base: Path) -> list[dict]:
    """Scan test/temp recursively and collect pairs that have:
    - transcript_lines.txt (non-empty)
    - word.json or words.json

    voice_over.mp3 is NOT required for this alignment-only suite.
    Returns list of dicts with keys: seg_id, folder, transcript, word_json.
    """
    results: list[dict] = []
    if not base.exists():
        return results

    for p in base.rglob("transcript_lines.txt"):
        folder = p.parent
        words_json_path = folder / "words.json"

        # Require words file present
        if not (words_json_path.exists()):
            continue

        try:
            txt = p.read_text(encoding="utf-8").strip()
        except Exception:  # pragma: no cover
            continue
        if not txt:
            continue

        seg_id = folder.name
        item = {
            "seg_id": seg_id,
            "folder": str(folder),
            "transcript": str(p),
            "word_json": str(words_json_path),
        }
        results.append(item)
    return results


@pytest.fixture(scope="session")
def alignment_pairs() -> list[dict]:
    """All alignment-only pairs (transcript_lines.txt + words.json/word.json) under test/temp."""
    base = Path("test/temp")
    return _discover_alignment_pairs(base)


def pytest_generate_tests(metafunc):  # noqa: D401 - pytest hook
    """Parametrize tests that request 'transcript_voice_sample'."""
    if "transcript_voice_sample" in metafunc.fixturenames:
        base = Path("test/temp")
        pairs = _discover_transcript_voice_pairs(base)
        ids = [
            f"{Path(item['folder']).parent.name}/{item['seg_id']}" for item in pairs
        ] or ["empty"]
        metafunc.parametrize("transcript_voice_sample", pairs or [{}], ids=ids)
    if "alignment_sample" in metafunc.fixturenames:
        base = Path("test/temp")
        pairs = _discover_alignment_pairs(base)
        ids = [
            f"{Path(item['folder']).parent.name}/{item['seg_id']}" for item in pairs
        ] or ["empty"]
        metafunc.parametrize("alignment_sample", pairs or [{}], ids=ids)


@pytest.fixture
def valid_video_data():
    """Load video input data from app/core/input_sample.json.

    Keeps the existing return shape {"json_data": <loaded_json>} for compatibility.
    """
    # Resolve project root relative to this file and load the JSON
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "app" / "core" / "input_sample.json"

    try:
        with sample_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"json_data": data}
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Falling back to inline sample due to error reading %s: %s", sample_path, e
        )
        # Fallback minimal payload to keep tests running
        return {
            "json_data": {
                "segments": [
                    {
                        "id": "s1",
                        "video": {
                            "url": "https://download.samplelib.com/mp4/sample-5s.mp4"
                        },
                    }
                ]
            }
        }


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Add test result to report object."""
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call
    setattr(item, f"rep_{rep.when}", rep)
