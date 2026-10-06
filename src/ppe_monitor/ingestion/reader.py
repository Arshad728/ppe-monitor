"""One reader thread per camera, keeping only its newest frame, and reconnecting when it drops out
(book, Chapters 11 and 21).

Each camera gets its own thread that reads frames as fast as the camera sends them and keeps
just the latest one. The processing loop takes whatever is newest when it is ready. A slow or
frozen camera can't hold up the others, and a busy processing loop skips stale frames instead
of falling behind real time.

When a camera stops sending, the reader closes it and tries again after a wait that doubles each
time (0.5 s, 1 s, 2 s, ... up to 10 s), so a camera that's down for an hour costs one connection
attempt every 10 s, a camera that blinks comes back within a second, and one that was down for
a long time is picked up within about 10 s of coming back. "Stops sending" means: the stream won't open, a
read fails, or no frame arrives for `read_timeout` seconds. RTSP always uses TCP, which doesn't
smear frames on a busy network the way UDP can.

A video file stands in for a camera during development. It is read at its own frame rate, not as
fast as possible, and it starts again from the beginning when it ends. With loop_files=False it
plays once and the reader stops (the Phase 6 evaluation plays each test clip once, as a camera).
"""

from __future__ import annotations

import os

# A camera that is down makes FFmpeg print an error at every reconnection attempt; the reader's own
# status already says so. Must be set before OpenCV opens its first stream.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np

from .rtsp import open_capture

try:                                   # OpenCV's own "can't open" warnings, for the same reason
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

CONNECTING, LIVE, RECONNECTING, STOPPED = "connecting", "live", "reconnecting", "stopped"


@dataclass
class Frame:
    image: np.ndarray
    seq: int              # 1, 2, 3, ... since the reader started (across reconnections)
    arrived: float        # time.monotonic() when it arrived
    wall: datetime        # the same moment on the wall clock (local time zone)


@dataclass
class ReaderStats:
    state: str = CONNECTING
    frames: int = 0             # frames received
    reconnects: int = 0         # times the camera dropped out and was reopened
    last_frame: float = 0.0     # time.monotonic() of the newest frame
    last_error: str = ""
    down_since: float | None = None   # when it dropped out (None while live)
    first_frame: float | None = None  # time.monotonic() of the first frame
    ended: bool = False               # a file played once (loop_files=False) has reached its end


def backoff(attempt: int, first: float = 0.5, longest: float = 10.0) -> float:
    """Seconds to wait before reconnection attempt number `attempt` (0, 1, 2, ...)."""
    return min(longest, first * 2 ** attempt)


class CameraReader:
    def __init__(self, camera_id: str, source: str, *, transport: str = "tcp", realtime_files: bool = True,
                 loop_files: bool = True, read_timeout: float = 5.0, open_timeout: float = 5.0,
                 first_wait: float = 0.5, longest_wait: float = 10.0):
        self.camera_id, self.source = camera_id, source
        self.transport = transport
        self.live = source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        self.realtime_files, self.loop_files = realtime_files, loop_files
        self.read_timeout, self.open_timeout = read_timeout, open_timeout
        self.first_wait, self.longest_wait = first_wait, longest_wait
        self.stats = ReaderStats()
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"reader-{camera_id}", daemon=True)
        # monotonic -> wall clock, fixed once so frame times never jump when the clock is adjusted
        self._wall0, self._mono0 = datetime.now().astimezone(), time.monotonic()

    # -- public ---------------------------------------------------------------------------------
    def start(self) -> "CameraReader":
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        self.stats.state = STOPPED

    def latest(self, after_seq: int = 0) -> Frame | None:
        """The newest frame, if it is newer than `after_seq`; otherwise None. Never waits."""
        with self._lock:
            f = self._frame
        return f if f is not None and f.seq > after_seq else None

    def wall(self, mono: float) -> datetime:
        return self._wall0 + timedelta(seconds=mono - self._mono0)

    # -- the thread -----------------------------------------------------------------------------
    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            cap = open_capture(self.source, transport=self.transport, open_timeout_s=self.open_timeout,
                               read_timeout_s=self.read_timeout)
            if not cap.isOpened():
                cap.release()
                self._down(f"could not open the stream")
                if self._stop.wait(backoff(attempt, self.first_wait, self.longest_wait)):
                    break
                attempt += 1
                continue
            got_any = self._read_until_failure(cap)
            cap.release()
            if self._stop.is_set() or self.stats.ended:
                break
            if got_any:
                attempt = 0          # it had been working: the first retry comes quickly
                self.stats.reconnects += 1
            self._down(self.stats.last_error or "the stream stopped")
            if self._stop.wait(backoff(attempt, self.first_wait, self.longest_wait)):
                break
            attempt += 1
        self.stats.state = STOPPED

    def _down(self, why: str) -> None:
        self.stats.state = RECONNECTING if self.stats.frames else CONNECTING
        self.stats.last_error = why
        if self.stats.down_since is None:
            self.stats.down_since = time.monotonic()

    def _read_until_failure(self, cap: cv2.VideoCapture) -> bool:
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        period = 1.0 / fps if fps > 0 else 1 / 15
        next_due = time.monotonic()
        got_any = False
        while not self._stop.is_set():
            ok, image = cap.read()
            if not ok:
                if not self.live and got_any and not self.loop_files:
                    self.stats.ended = True                     # a file played once: done
                    return got_any
                if not self.live and self.loop_files and got_any:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # a file: play it again
                    ok, image = cap.read()
                if not ok:
                    self.stats.last_error = "no frame (stream ended, or nothing for the read timeout)"
                    return got_any
            now = time.monotonic()
            if not self.live and self.realtime_files:            # a file pretending to be a camera
                next_due += period
                if next_due > now:
                    if self._stop.wait(next_due - now):
                        return got_any
                    now = time.monotonic()
                else:
                    next_due = now
            got_any = True
            with self._lock:
                seq = self._frame.seq + 1 if self._frame else 1
                self._frame = Frame(image, seq, now, self.wall(now))
            self.stats.frames += 1
            self.stats.last_frame = now
            if self.stats.first_frame is None:
                self.stats.first_frame = now
            if self.stats.state != LIVE:
                self.stats.state, self.stats.down_since, self.stats.last_error = LIVE, None, ""
        return got_any
