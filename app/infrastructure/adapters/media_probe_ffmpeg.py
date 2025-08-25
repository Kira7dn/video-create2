from __future__ import annotations

import asyncio
from app.application.interfaces.media_probe import IMediaProbe
from utils.subprocess_utils import safe_subprocess_run


class FFmpegMediaProbe(IMediaProbe):
    async def duration(self, input_path: str) -> float:
        def _probe() -> float:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                input_path,
            ]
            result = safe_subprocess_run(cmd, f"Probe duration for {input_path}")
            duration_str = (result.stdout or "").strip()
            if not duration_str:
                raise ValueError("Empty duration output from ffprobe")
            return float(duration_str)

        return await asyncio.to_thread(_probe)
