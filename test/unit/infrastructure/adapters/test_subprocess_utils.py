import subprocess
import pytest

from app.infrastructure.adapters.renderer.utils.ffmpeg_utils import (
    safe_subprocess_run,
    SubprocessError,
)


class DummyCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_safe_subprocess_run_success(monkeypatch):
    def fake_run(cmd, stdout, stderr, text, check):
        assert isinstance(cmd, (list, tuple))
        assert check is True
        return DummyCompleted(stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = safe_subprocess_run(["echo", "hi"], "Echo test")
    assert isinstance(res, DummyCompleted.__class__) or hasattr(res, "stdout")
    assert res.stdout == "ok"


def test_safe_subprocess_run_calledprocesserror(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], output="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SubprocessError) as ei:
        safe_subprocess_run(["ffmpeg"], "Op")
    msg = str(ei.value)
    assert "return code 1" in msg or "Exit code: 1" in msg
    assert "Error output:" in msg or "stderr" in msg or "FFmpeg stderr" in msg


def test_safe_subprocess_run_ffmpeg_not_found(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SubprocessError) as ei:
        safe_subprocess_run(["ffmpeg"], "Op")
    assert "FFmpeg not found" in str(ei.value)


def test_safe_subprocess_run_permission_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise PermissionError("no permission")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SubprocessError) as ei:
        safe_subprocess_run(["ffmpeg"], "Op")
    assert "OS/Permission error" in str(ei.value)


def test_safe_subprocess_run_windows_error_codes(monkeypatch):
    class CPE(subprocess.CalledProcessError):
        pass

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(-2147024896, ["ffmpeg"], output="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SubprocessError) as ei:
        safe_subprocess_run(["ffmpeg"], "Op")
    msg = str(ei.value)
    assert "0x80004005" in msg or "Access Denied" in msg or "Windows Error" in msg
