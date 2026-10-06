"""Finding the external programs Phase 0 relies on: ffmpeg, ffprobe and MediaMTX."""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import PROJECT_ROOT

TOOLS_DIR = PROJECT_ROOT / "tools"
MEDIAMTX_DIR = TOOLS_DIR / "mediamtx"
_EXE = ".exe" if platform.system() == "Windows" else ""


def find_ffmpeg() -> str | None:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    return os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")


def find_mediamtx() -> str | None:
    """Look in $MEDIAMTX_BIN, then tools/mediamtx/ (where scripts/get_mediamtx.py puts it), then PATH."""
    override = os.environ.get("MEDIAMTX_BIN")
    if override:
        return override if Path(override).is_file() else None
    local = MEDIAMTX_DIR / f"mediamtx{_EXE}"
    if local.is_file():
        return str(local)
    return shutil.which("mediamtx")


def first_line_of(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Run a command and return the first non-empty line it prints, or None if it fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if line.strip():
            return line.strip()
    return None


def ffmpeg_encoders(ffmpeg: str) -> set[str]:
    """Names of the video encoders this ffmpeg build offers (e.g. libx264, h264_videotoolbox)."""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # Encoder lines look like: " V....D libx264   libx264 H.264 / AVC ..."
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] == "V":
            names.add(parts[1])
    return names


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# H.264 encoders to try, best first. On Apple Silicon the hardware encoder (VideoToolbox,
# the Mac's media engine) comes first: it keeps the CPU free for detection in later phases.
# Elsewhere libx264 (software) is the predictable choice. mpeg4 is not H.264; it is a last
# resort that every ffmpeg build has.
_APPLE_PREFERENCE = ["h264_videotoolbox", "libx264", "libopenh264", "mpeg4"]
_DEFAULT_PREFERENCE = ["libx264", "libopenh264", "h264_nvenc", "h264_videotoolbox", "mpeg4"]


def encoder_preference() -> list[str]:
    return _APPLE_PREFERENCE if is_apple_silicon() else _DEFAULT_PREFERENCE


def encoder_args(encoder: str, *, live: bool) -> list[str]:
    """Quality/latency settings for each encoder. `live` means streaming in real time."""
    if encoder == "libx264":
        return ["-preset", "veryfast"] + (["-tune", "zerolatency"] if live else ["-crf", "23"])
    if encoder == "h264_videotoolbox":
        # -g 600: don't add keyframes of its own (it did every ~0.8 s); the forced one-per-second
        # keyframes from the caller are the only ones wanted
        return (["-realtime", "1"] if live else []) + ["-b:v", "4M", "-g", "600"]
    if encoder == "h264_nvenc":
        return ["-preset", "p1", "-tune", "ll"] if live else ["-preset", "p4"]
    if encoder == "mpeg4":
        return ["-q:v", "4"]
    return []


@functools.lru_cache(maxsize=None)
def encoder_works(ffmpeg: str, encoder: str) -> bool:
    """Encode one second of test pattern with this encoder. ffmpeg lists every encoder it was
    built with, including hardware ones whose hardware isn't there (h264_nvenc without an
    NVIDIA GPU), so being listed doesn't mean it works."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
           "-i", "testsrc2=size=640x360:rate=15", "-frames:v", "15",
           "-c:v", encoder, *encoder_args(encoder, live=True), "-pix_fmt", "yuv420p", "-bf", "0",
           "-force_key_frames", "expr:gte(t,n_forced*1)", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def pick_encoder(ffmpeg: str, requested: str = "auto") -> str:
    """The encoder to use: `requested` if it works, or with "auto" the first working one
    from encoder_preference()."""
    available = ffmpeg_encoders(ffmpeg)
    if requested != "auto":
        if available and requested not in available:
            raise RuntimeError(f"ffmpeg has no '{requested}' encoder. Available: "
                               + ", ".join(e for e in encoder_preference() if e in available))
        if not encoder_works(ffmpeg, requested):
            raise RuntimeError(f"ffmpeg's '{requested}' encoder failed a test encode on this machine")
        return requested
    for name in encoder_preference():
        if name in available and encoder_works(ffmpeg, name):
            return name
    raise RuntimeError("None of ffmpeg's video encoders passed a test encode; reinstall ffmpeg")
