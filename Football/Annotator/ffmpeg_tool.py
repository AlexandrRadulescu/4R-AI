"""Find a usable ffmpeg, without requiring a system install.

Resolution order:
  1. ffmpeg on PATH (a real system install, always preferred)
  2. the binary bundled with the imageio-ffmpeg pip package

The fallback exists because installing ffmpeg on Windows means downloading a
zip and editing environment variables, which is a genuinely annoying gate to
put in front of the project. `pip install imageio-ffmpeg` sidesteps all of it.

That package ships ffmpeg but NOT ffprobe, so stream inspection here parses
ffmpeg's own stderr instead. Slightly less tidy than ffprobe's JSON, but it
means one binary covers everything.
"""

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?: (Video|Audio): ([A-Za-z0-9_]+)"
)
_DURATION_RE = re.compile(r"Duration: (\d+):(\d\d):(\d\d\.\d+)")


@lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    """Path to an ffmpeg executable, or raise with install instructions."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise SystemExit(
        "No ffmpeg available. Either:\n"
        "  pip install imageio-ffmpeg        (easiest -- no PATH setup)\n"
        "  winget install Gyan.FFmpeg        (system-wide, then reopen the terminal)"
    )


def ffmpeg_source() -> str:
    """'system' or 'bundled' -- for reporting which one is in use."""
    return "system" if shutil.which("ffmpeg") else "bundled (imageio-ffmpeg)"


def probe(path: Path):
    """-> (video_codec, audio_codec, duration_seconds); None where unknown.

    Runs `ffmpeg -i FILE` with no output, which makes ffmpeg print the stream
    layout to stderr and exit non-zero. That non-zero exit is expected.
    """
    proc = subprocess.run(
        [get_ffmpeg(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, errors="replace",
    )
    text = proc.stderr

    video = audio = None
    for kind, codec in _STREAM_RE.findall(text):
        if kind == "Video" and video is None:
            video = codec.lower()
        elif kind == "Audio" and audio is None:
            audio = codec.lower()

    duration = None
    m = _DURATION_RE.search(text)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = h * 3600 + mn * 60 + s

    return video, audio, duration


def run(args, **kwargs):
    """Invoke ffmpeg with the resolved binary."""
    return subprocess.run([get_ffmpeg(), *args], check=True, **kwargs)