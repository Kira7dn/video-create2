import json
import os
import shutil
from types import SimpleNamespace

import pytest

from app.infrastructure.adapters.renderer.utils.ffmpeg_utils import (
    ffmpeg_concat_videos,
    VideoProcessingError,
)


class DummyResult:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def seg_files(tmp_path):
    # Create two tiny fake files (we don't need valid media when stubbing)
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    return [str(a), str(b)]


def test_validate_inputs_errors(tmp_path, monkeypatch):
    temp = tmp_path / "tmp"
    temp.mkdir()

    # 1) empty segments
    with pytest.raises(VideoProcessingError, match="cannot be empty"):
        ffmpeg_concat_videos([], str(tmp_path / "out.mp4"), str(temp))

    # 2) segment not dict
    with pytest.raises(VideoProcessingError, match="must be a dictionary"):
        ffmpeg_concat_videos(["bad"], str(tmp_path / "out.mp4"), str(temp))

    # 3) missing path key
    with pytest.raises(VideoProcessingError, match="missing 'path'"):
        ffmpeg_concat_videos([{"id": 1}], str(tmp_path / "out.mp4"), str(temp))

    # 4) path not exists
    with pytest.raises(VideoProcessingError, match="Video file not found"):
        ffmpeg_concat_videos(
            [{"id": 1, "path": str(tmp_path / "missing.mp4")}],
            str(tmp_path / "out.mp4"),
            str(temp),
        )

    # 5) output dir missing
    seg = tmp_path / "ok.mp4"
    seg.write_bytes(b"x")
    out_in_missing_dir = tmp_path / "no_such_dir" / "out.mp4"
    with pytest.raises(VideoProcessingError, match="Output directory not found"):
        ffmpeg_concat_videos(
            [{"id": 1, "path": str(seg)}], str(out_in_missing_dir), str(temp)
        )


def test_concat_only_flow(monkeypatch, tmp_path, seg_files):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"
    
    # Make os.access return True to pass permission check
    monkeypatch.setattr(os, "access", lambda p, mode: True)

    # Stub subprocess to capture concat call and simulate success
    calls = []
    
    def stub(cmd, desc, logger=None):
        calls.append((cmd, desc))
        # For ffprobe (not used here) return minimal JSON
        if "ffprobe" in cmd[0]:
            return DummyResult(stdout=json.dumps({"format": {"duration": "1.0"}}))
        return DummyResult(stdout="", stderr="info")

    monkeypatch.setattr(
        "app.infrastructure.adapters.renderer.utils.ffmpeg_utils.safe_subprocess_run",
        stub,
    )

    # Build segments
    segments = [{"id": "s1", "path": seg_files[0]}, {"id": "s2", "path": seg_files[1]}]

    result = ffmpeg_concat_videos(segments, str(out), str(temp), background_music=None)

    # Kiểm tra kết quả trả về thay vì kiểm tra copy
    assert result == str(out)
    
    # Kiểm tra lệnh concat được gọi
    assert calls, "Expected concat to be invoked"
    concat_cmd = calls[0][0]
    assert concat_cmd[0] == "ffmpeg" and "concat" in concat_cmd


def test_bgm_skip_when_too_short(monkeypatch, tmp_path, seg_files):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(os, "access", lambda p, mode: True)

    # Stub subprocess: ffprobe on temp video returns small duration (0.05s)
    calls = []

    def stub(cmd, desc, logger=None):
        calls.append((cmd, desc))
        if cmd and cmd[0] == "ffprobe":
            return DummyResult(stdout=json.dumps({"format": {"duration": "0.05"}}))
        return DummyResult(stdout="", stderr="mean_volume: -10.0 dB")

    monkeypatch.setattr(
        "app.infrastructure.adapters.renderer.utils.ffmpeg_utils.safe_subprocess_run",
        stub,
    )

    copied = {}
    monkeypatch.setattr(
        shutil, "copy2", lambda src, dst: copied.setdefault("done", True)
    )

    segments = [{"id": "s1", "path": seg_files[0]}]

    bgm = {"local_path": str(seg_files[1])}

    ffmpeg_concat_videos(segments, str(out), str(temp), background_music=bgm)

    # Should decide to skip mixing and copy directly due to too short video duration
    assert copied.get("done") is True
    # Ensure at least concat was run
    assert any("concat" in cmd for cmd, _ in calls)


def test_bgm_mixing_with_loop_and_filters(monkeypatch, tmp_path, seg_files):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(os, "access", lambda p, mode: True)

    # Prepare a sequence of ffprobe results: first for temp video duration, then for bgm duration
    probes = [
        DummyResult(
            stdout=json.dumps({"format": {"duration": "3.1"}})
        ),  # video duration
        DummyResult(stdout=json.dumps({"format": {"duration": "1.0"}})),  # bgm duration
    ]

    calls = []

    def stub(cmd, desc, logger=None):
        # Record every call
        calls.append((cmd, desc))
        if cmd and cmd[0] == "ffprobe":
            return probes.pop(0)
        # Loudnorm analyze pass (print_format=json) returns JSON in stderr
        if cmd and cmd[0] == "ffmpeg" and "-af" in cmd:
            af = cmd[cmd.index("-af") + 1]
            if "loudnorm=" in af and "print_format=json" in af:
                sample_json = {
                    "input_i": -20.0,
                    "input_lra": 5.0,
                    "input_tp": -2.0,
                    "input_thresh": -30.0,
                    "target_offset": 0.0,
                }
                return DummyResult(stdout="", stderr=json.dumps(sample_json))
            if "loudnorm=" in af and "print_format=summary" in af:
                return DummyResult(stdout="", stderr="")
        return DummyResult(stdout="", stderr="")

    monkeypatch.setattr(
        "app.infrastructure.adapters.renderer.utils.ffmpeg_utils.safe_subprocess_run",
        stub,
    )

    # Don't actually copy files at the end
    monkeypatch.setattr(shutil, "copy2", lambda src, dst: None)

    segments = [{"id": "s1", "path": seg_files[0]}]

    bgm = {"local_path": str(seg_files[1]), "start_delay": 0.0, "end_delay": 0.0}

    ffmpeg_concat_videos(segments, str(out), str(temp), background_music=bgm)

    # Find mixing command call
    mix_calls = [
        c for c in calls if c[0] and c[0][0] == "ffmpeg" and "-filter_complex" in c[0]
    ]
    assert mix_calls, "Expected background music mixing to be invoked"
    mix_cmd = mix_calls[-1][0]

    # Expect looping due to video 3.1s and bgm 1.0s -> loops_needed = ceil(3.1/1.0) - 1 >= 1
    assert "-stream_loop" in mix_cmd
    if "-stream_loop" in mix_cmd:
        idx = mix_cmd.index("-stream_loop") + 1
        assert idx < len(mix_cmd)
        assert str(mix_cmd[idx]).isdigit() and int(mix_cmd[idx]) >= 1

    # Ensure filter_complex contains our constructed chains and maps
    fc_index = mix_cmd.index("-filter_complex") + 1
    fc = mix_cmd[fc_index]
    assert "amix=inputs=2:duration=first:dropout_transition=2" in fc
    assert "adelay=" in fc or "atrim=" in fc or "volume=" in fc

    # Output mapping and codecs
    assert "-map" in mix_cmd and "0:v" in mix_cmd
    assert "-map" in mix_cmd and "[aout]" in mix_cmd
    assert "-c:v" in mix_cmd and "libx264" in mix_cmd
    assert "-c:a" in mix_cmd and "aac" in mix_cmd

    # Note: auto-volume (loudnorm) is disabled by default; do not assert loudnorm filters here.


@pytest.mark.skip(reason="auto-volume disabled by default")
def test_loudnorm_defaults_hardcoded(monkeypatch, tmp_path, seg_files):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(os, "access", lambda p, mode: True)

    # video duration (for temp concat) and bgm duration
    probes = [
        DummyResult(stdout=json.dumps({"format": {"duration": "2.0"}})),
        DummyResult(stdout=json.dumps({"format": {"duration": "5.0"}})),
    ]

    calls = []

    def stub(cmd, desc, logger=None):
        calls.append((cmd, desc))
        if cmd and cmd[0] == "ffprobe":
            return probes.pop(0)
        if cmd and cmd[0] == "ffmpeg" and "-af" in cmd:
            af = cmd[cmd.index("-af") + 1]
            # Both analyze and normalize should carry the hardcoded I/TP/LRA
            assert "loudnorm=I=-16.0:TP=-1.5:LRA=11.0" in af
            return DummyResult(stdout="", stderr="{}")
        return DummyResult(stdout="", stderr="")

    monkeypatch.setattr("utils.video_utils.safe_subprocess_run", stub)
    monkeypatch.setattr(shutil, "copy2", lambda src, dst: None)

    segments = [{"id": "s1", "path": seg_files[0]}]
    bgm = {"local_path": str(seg_files[1])}
    ffmpeg_concat_videos(segments, str(out), str(temp), background_music=bgm)

    # Expect at least one ffmpeg call with '-af loudnorm=I=-16.0:TP=-1.5:LRA=11.0'
    ln_calls = [c for c in calls if c[0] and c[0][0] == "ffmpeg" and "-af" in c[0]]
    assert ln_calls, "Expected loudnorm normalization calls"


@pytest.mark.skip(reason="auto-volume disabled by default")
def test_bgm_ducking_enabled_adds_sidechaincompress(monkeypatch, tmp_path, seg_files):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(os, "access", lambda p, mode: True)

    probes = [
        DummyResult(stdout=json.dumps({"format": {"duration": "2.5"}})),  # video
        DummyResult(stdout=json.dumps({"format": {"duration": "4.0"}})),  # bgm
    ]

    calls = []

    def stub(cmd, desc, logger=None):
        calls.append((cmd, desc))
        if cmd and cmd[0] == "ffprobe":
            return probes.pop(0)
        if cmd and cmd[0] == "ffmpeg" and "-af" in cmd:
            # emulate loudnorm analyze/normalize
            return DummyResult(stdout="", stderr="{}")
        return DummyResult(stdout="", stderr="")

    monkeypatch.setattr("utils.video_utils.safe_subprocess_run", stub)
    monkeypatch.setattr(shutil, "copy2", lambda src, dst: None)

    segments = [{"id": "s1", "path": seg_files[0]}]
    bgm = {"local_path": str(seg_files[1])}  # ducking default True
    ffmpeg_concat_videos(segments, str(out), str(temp), background_music=bgm)

    mix_calls = [
        c for c in calls if c[0] and c[0][0] == "ffmpeg" and "-filter_complex" in c[0]
    ]
    assert mix_calls, "Expected mixing call"
    fc = mix_calls[-1][0][mix_calls[-1][0].index("-filter_complex") + 1]
    assert "sidechaincompress" in fc, "Ducking should add sidechaincompress"
    assert "amix=inputs=2:duration=first:dropout_transition=2:normalize=0" in fc


@pytest.mark.skip(reason="auto-volume disabled by default")
def test_bgm_ducking_disabled_omits_sidechaincompress(monkeypatch, tmp_path, seg_files):
    temp = tmp_path / "tmp"
    temp.mkdir()
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(os, "access", lambda p, mode: True)

    probes = [
        DummyResult(stdout=json.dumps({"format": {"duration": "2.5"}})),  # video
        DummyResult(stdout=json.dumps({"format": {"duration": "4.0"}})),  # bgm
    ]

    calls = []

    def stub(cmd, desc, logger=None):
        calls.append((cmd, desc))
        if cmd and cmd[0] == "ffprobe":
            return probes.pop(0)
        if cmd and cmd[0] == "ffmpeg" and "-af" in cmd:
            return DummyResult(stdout="", stderr="{}")
        return DummyResult(stdout="", stderr="")

    monkeypatch.setattr("utils.video_utils.safe_subprocess_run", stub)
    monkeypatch.setattr(shutil, "copy2", lambda src, dst: None)

    segments = [{"id": "s1", "path": seg_files[0]}]
    bgm = {"local_path": str(seg_files[1]), "ducking": False}
    ffmpeg_concat_videos(segments, str(out), str(temp), background_music=bgm)

    mix_calls = [
        c for c in calls if c[0] and c[0][0] == "ffmpeg" and "-filter_complex" in c[0]
    ]
    assert mix_calls, "Expected mixing call"
    fc = mix_calls[-1][0][mix_calls[-1][0].index("-filter_complex") + 1]
    assert (
        "sidechaincompress" not in fc
    ), "Ducking disabled should omit sidechaincompress"
    assert "amix=inputs=2:duration=first:dropout_transition=2:normalize=0" in fc
