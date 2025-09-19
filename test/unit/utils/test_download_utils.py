import os
import types
import asyncio
import pytest

import utils.download_utils as du


@pytest.mark.asyncio
async def test_download_file_skips_if_exists(monkeypatch, tmp_path):
    dest = tmp_path / "out.bin"

    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    called = {"internal": 0}

    async def fake_internal(url, dest_path):
        called["internal"] += 1
        return {"success": True, "local_path": dest_path}

    monkeypatch.setattr(du, "_download_file_internal", fake_internal)

    result = await du.download_file("https://example.com/f.bin", str(dest))
    assert result == str(dest)
    assert called["internal"] == 0  # should skip actual download


@pytest.mark.asyncio
async def test_download_file_directory_destination(monkeypatch, tmp_path):
    dest_dir = tmp_path / "downloads"
    url = "https://example.com/path/file.bin"

    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    recorded = {"dest_path": None}

    async def fake_internal(u, dest_path):
        recorded["dest_path"] = dest_path
        return {"success": True, "local_path": dest_path}

    monkeypatch.setattr(du, "_download_file_internal", fake_internal)

    result = await du.download_file(url, str(dest_dir))
    assert result.endswith("file.bin")
    assert recorded["dest_path"].endswith("file.bin")


@pytest.mark.asyncio
async def test__download_file_internal_success(monkeypatch, tmp_path):
    # Patch settings timeout to avoid importing complexities
    monkeypatch.setattr(du.settings, "download_timeout", 5, raising=False)

    chunks = [b"abc", b"def"]

    class DummyContent:
        async def iter_chunked(self, n):
            for c in chunks:
                yield c

    class DummyResponse:
        def __init__(self):
            self.content = DummyContent()

        def raise_for_status(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return DummyResponse()

    # Fake aiofiles.open
    writes = []

    class DummyAIOFile:
        def __init__(self, path, mode):
            self.path = path
            self.mode = mode

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def write(self, data):
            writes.append(data)

    monkeypatch.setattr(
        du,
        "aiohttp",
        types.SimpleNamespace(
            ClientTimeout=lambda total: total,
            ClientSession=DummySession,
            ClientError=type("ClientError", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(du, "aiofiles", types.SimpleNamespace(open=lambda p, m: DummyAIOFile(p, m)))
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    res = await du._download_file_internal("https://example.com/a.bin", str(tmp_path / "a.bin"))
    assert res["success"] is True
    assert b"".join(writes) == b"abcdef"


@pytest.mark.asyncio
async def test__download_file_internal_http_error(monkeypatch, tmp_path):
    # Simulate aiohttp.ClientError
    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            class R:
                def raise_for_status(self):
                    raise du.aiohttp.ClientError("boom")

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

            return R()

    monkeypatch.setattr(
        du,
        "aiohttp",
        types.SimpleNamespace(
            ClientTimeout=lambda total: total,
            ClientSession=lambda timeout: DummySession(),
            ClientError=type("ClientError", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    res = await du._download_file_internal("https://example.com/a.bin", str(tmp_path / "a.bin"))
    assert res["success"] is False
    assert "Failed to download" in res["error"]


@pytest.mark.asyncio
async def test__download_file_internal_os_error(monkeypatch, tmp_path):
    # Raise OSError on open
    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            class R:
                def raise_for_status(self):
                    return None

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

                class content:
                    @staticmethod
                    async def iter_chunked(n):
                        yield b"data"

            return R()

    def raise_open(*a, **k):
        class Ctx:
            async def __aenter__(self):
                raise OSError("disk full")

            async def __aexit__(self, *a):
                return False

        return Ctx()

    monkeypatch.setattr(
        du,
        "aiohttp",
        types.SimpleNamespace(
            ClientTimeout=lambda total: total,
            ClientSession=lambda timeout: DummySession(),
            ClientError=type("ClientError", (Exception,), {}),
        ),
    )
    monkeypatch.setattr(du, "aiofiles", types.SimpleNamespace(open=raise_open))
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    res = await du._download_file_internal("https://example.com/a.bin", str(tmp_path / "a.bin"))
    assert res["success"] is False
    assert "File operation error" in res["error"]
