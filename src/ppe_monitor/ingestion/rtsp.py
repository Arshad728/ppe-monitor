"""Opening an RTSP stream with OpenCV and checking that it plays back correctly.

This is the Phase 0 "done when" check: a stream counts as playing back correctly when it
opens, delivers frames at (close to) the rate the camera sends them, never stalls for
long, and the frames are real pictures rather than the flat grey frames a broken decoder
produces.

Two details matter for a live safety system (Chapter 11):

* Low-latency opening. By default FFmpeg spends a couple of seconds analysing a stream
  before handing over the first frame and buffers everything it saw meanwhile, so the
  reader starts ~2 s behind live. `open_capture` turns that buffering off and shortens
  the analysis, and falls back to the default settings if a camera needs more time.
* Measuring steady-state frame rate. The first moment after opening can deliver a burst
  of buffered frames, which would make the measured fps look better than it is, so the
  first second is left out of the fps figure.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

import cv2
import numpy as np

# FFmpeg options for live streams: no input buffering, and at most 0.5 s / 500 kB of
# stream analysis before the first frame is returned.
LOW_LATENCY_OPTIONS = "fflags;nobuffer|analyzeduration;500000|probesize;500000"

# How often (in frames) to check a frame for being blank; checking every frame is wasteful.
_BLANK_CHECK_EVERY = 5
# A real scene has texture; a decoder failure tends to produce one flat colour.
_BLANK_STD_THRESHOLD = 3.0
# Frames arriving within this long of the first one are the backlog buffered while opening.
_BURST_WINDOW_S = 0.25
# Leave the first second out of the fps measurement.
_WARMUP_S = 1.0

# OpenCV reads FFmpeg options from this environment variable at the moment a capture opens,
# so setting it and opening must happen together when several threads open streams.
_OPTIONS_VAR = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_open_lock = threading.Lock()


def _ffmpeg_options(transport: str, low_latency: bool) -> str:
    options = f"rtsp_transport;{transport}"
    return f"{options}|{LOW_LATENCY_OPTIONS}" if low_latency else options


def open_capture(
    url: str,
    transport: str = "tcp",
    open_timeout_s: float = 10.0,
    read_timeout_s: float = 5.0,
    low_latency: bool = True,
) -> cv2.VideoCapture:
    """Open a stream (or file) with OpenCV's FFmpeg backend.

    TCP is the safer default for RTSP: UDP is lower-latency but drops packets on busy
    networks, which shows up as smeared or grey frames (Chapter 11). With `low_latency`
    the stream is first opened with LOW_LATENCY_OPTIONS and, if that fails, once more with
    FFmpeg's defaults (some cameras need longer analysis).
    """
    return open_capture_with_mode(url, transport, open_timeout_s, read_timeout_s, low_latency)[0]


def open_capture_with_mode(
    url: str,
    transport: str = "tcp",
    open_timeout_s: float = 10.0,
    read_timeout_s: float = 5.0,
    low_latency: bool = True,
) -> tuple[cv2.VideoCapture, bool]:
    """Like open_capture, but also says whether the low-latency settings were the ones that worked."""
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_s * 1000),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_s * 1000),
    ]
    is_rtsp = url.lower().startswith(("rtsp://", "rtsps://"))
    attempts = [True, False] if (is_rtsp and low_latency) else [False]
    cap, used = None, False
    for used in attempts:
        options = _ffmpeg_options(transport, used) if is_rtsp else None
        with _ffmpeg_options_set(options):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        if cap.isOpened():
            break
        cap.release()
    return cap, used


@contextmanager
def _ffmpeg_options_set(options: str | None):
    """Apply FFmpeg capture options only while one capture opens, then put back whatever was
    there before. OpenCV reads them from a process-wide environment variable, and leaving
    live-stream options behind is harmful: with `fflags;nobuffer` still set, reading a video
    FILE silently drops frames."""
    with _open_lock:
        previous = os.environ.pop(_OPTIONS_VAR, None)
        if options:
            os.environ[_OPTIONS_VAR] = options
        try:
            yield
        finally:
            os.environ.pop(_OPTIONS_VAR, None)
            if previous is not None:
                os.environ[_OPTIONS_VAR] = previous


def rtsp_describe(url: str, timeout: float = 2.0) -> int | None:
    """Ask an RTSP server about a stream (the DESCRIBE request every player sends first) and
    return the status code: 200 = the stream is live, 404 = nothing is publishing there,
    401 = a password is needed. None if the server can't be reached.

    Used to wait for the fake cameras without OpenCV printing an error for every early try.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 554
    netloc_host = f"[{host}]" if ":" in host else host
    request_url = parsed._replace(netloc=f"{netloc_host}:{port}").geturl()  # never send credentials here
    request = (f"DESCRIBE {request_url} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n"
               "User-Agent: ppe-monitor\r\n\r\n")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request.encode("ascii"))
            status_line = sock.recv(1024).decode("latin-1").split("\r\n", 1)[0]
    except OSError:
        return None
    parts = status_line.split()
    if len(parts) >= 2 and parts[0].startswith("RTSP/") and parts[1].isdigit():
        return int(parts[1])
    return None


@dataclass
class StreamReport:
    url: str
    opened: bool = False
    low_latency: bool = False
    open_seconds: float | None = None
    first_frame_seconds: float | None = None
    frames: int = 0
    elapsed_seconds: float = 0.0
    measured_fps: float = 0.0
    reference_fps: float | None = None
    startup_burst_frames: int = 0
    width: int = 0
    height: int = 0
    longest_gap_seconds: float = 0.0
    blank_frames: int = 0
    ended_early: bool = False
    stopped_by_user: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.opened and not self.problems

    def summary_lines(self) -> list[str]:
        ref = f"{self.reference_fps:.1f}" if self.reference_fps else "unknown"
        mode = "low-latency" if self.low_latency else "default"
        lines = [
            f"url              {self.url}",
            f"opened           {'yes' if self.opened else 'NO'}"
            + (f" in {self.open_seconds:.2f} s ({mode} mode)" if self.open_seconds is not None else ""),
        ]
        if self.opened:
            lines += [
                f"first frame      after {self.first_frame_seconds:.2f} s"
                if self.first_frame_seconds is not None
                else "first frame      never arrived",
                f"resolution       {self.width} x {self.height}",
                f"frames           {self.frames} in {self.elapsed_seconds:.1f} s",
                f"fps              {self.measured_fps:.1f} steady-state / {ref} expected",
                f"startup backlog  {self.startup_burst_frames} frame(s) buffered while opening",
                f"longest gap      {self.longest_gap_seconds:.2f} s between frames",
                f"blank frames     {self.blank_frames}",
            ]
        lines.append("result           " + ("PASS" if self.passed else "FAIL"))
        lines += [f"  - {p}" for p in self.problems]
        return lines


def _explain_open_failure(url: str, open_timeout_s: float) -> str:
    """Turn "OpenCV couldn't open it" into the most likely actual cause."""
    if not url.lower().startswith(("rtsp://", "rtsps://")):
        return "OpenCV could not open this file or URL (does it exist, and is it a video?)"
    parsed = urlparse(url)
    where = f"{parsed.hostname}:{parsed.port or 554}"
    status = rtsp_describe(url)
    if status is None:
        return (f"nothing is answering at {where}. If this is a fake camera, start the network first: "
                "python scripts/fake_cameras.py --with-server")
    if status == 404:
        return (f"the RTSP server at {where} is running, but nothing is publishing to {parsed.path}. "
                "Check the path, or that this camera's ffmpeg publisher is running (logs/fake_cameras/).")
    if status == 401:
        return "the camera wants a username and password; put them in the URL (see configs/cameras.yaml)"
    if status == 200:
        return (f"the server says the stream is live, but OpenCV could not open it within "
                f"{open_timeout_s:.0f} s - try --transport udp/tcp, or check the codec")
    return f"the RTSP server answered with status {status}"


def _is_blank(frame: np.ndarray) -> bool:
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < _BLANK_STD_THRESHOLD


def probe_stream(
    url: str,
    *,
    display_url: str | None = None,
    transport: str = "tcp",
    duration_s: float | None = 10.0,
    expected_fps: float | None = None,
    min_fps_ratio: float = 0.9,
    max_gap_s: float = 1.0,
    open_timeout_s: float = 10.0,
    low_latency: bool = True,
    show: bool = False,
    window_title: str = "stream",
) -> StreamReport:
    """Read a stream for `duration_s` seconds and report whether it played back correctly.

    With `show=True` the frames are displayed in a window (press q to stop); with
    `duration_s=None` it runs until q is pressed.
    """
    report = StreamReport(url=display_url or url)

    if show:
        # Create the window before the stream opens: the first window can take a while to
        # appear, and that delay must not be mistaken for the stream stalling.
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.waitKey(1)
    window_sized = False

    t0 = time.monotonic()
    cap, used_low_latency = open_capture_with_mode(url, transport=transport, open_timeout_s=open_timeout_s,
                                                   low_latency=low_latency)
    if not cap.isOpened():
        report.problems.append(_explain_open_failure(url, open_timeout_s))
        if show:
            cv2.destroyWindow(window_title)
        return report
    report.opened = True
    report.low_latency = used_low_latency
    report.open_seconds = time.monotonic() - t0

    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    # RTSP streams often report 0 or a clock rate (e.g. 90000) instead of a frame rate.
    if expected_fps:
        report.reference_fps = expected_fps
    elif 1.0 <= reported_fps <= 120.0:
        report.reference_fps = reported_fps

    first_t = last_t = None
    steady_first_t = None
    steady_frames = 0
    recent = deque(maxlen=30)
    try:
        while True:
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                report.ended_early = True
                break

            if first_t is None:
                first_t = now
                report.first_frame_seconds = now - t0
                report.height, report.width = frame.shape[:2]
            else:
                report.longest_gap_seconds = max(report.longest_gap_seconds, now - last_t)
            last_t = now
            report.frames += 1
            recent.append(now)
            if now - first_t <= _BURST_WINDOW_S:
                report.startup_burst_frames += 1
            if now - first_t >= _WARMUP_S:
                if steady_first_t is None:
                    steady_first_t = now
                steady_frames += 1

            if report.frames % _BLANK_CHECK_EVERY == 1 and _is_blank(frame):
                report.blank_frames += 1

            if show:
                span = recent[-1] - recent[0]
                rolling = (len(recent) - 1) / span if span > 0 else 0.0
                overlay = frame.copy()
                text = (f"{window_title}  {report.width}x{report.height}  {rolling:4.1f} fps  "
                        f"frames {report.frames}   [q] quit")
                cv2.rectangle(overlay, (0, report.height - 34), (report.width, report.height), (0, 0, 0), -1)
                cv2.putText(overlay, text, (10, report.height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                if not window_sized:  # fit big streams (e.g. 1080p) on a laptop screen
                    scale = min(1.0, 1280 / report.width)
                    cv2.resizeWindow(window_title, int(report.width * scale), int(report.height * scale))
                    window_sized = True
                cv2.imshow(window_title, overlay)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    report.stopped_by_user = True
                    break

            if duration_s is not None and now - first_t >= duration_s:
                break
    except KeyboardInterrupt:
        report.stopped_by_user = True
    finally:
        cap.release()
        if show:
            cv2.destroyWindow(window_title)
            cv2.waitKey(1)

    if first_t is None:
        report.problems.append("the stream opened but no frames arrived")
        return report

    report.elapsed_seconds = last_t - first_t
    if steady_frames > 1 and last_t > steady_first_t:
        report.measured_fps = (steady_frames - 1) / (last_t - steady_first_t)
    elif report.frames > 1 and report.elapsed_seconds > 0:  # very short run: use everything
        report.measured_fps = (report.frames - 1) / report.elapsed_seconds

    if report.ended_early and not report.stopped_by_user:
        report.problems.append(f"the stream stopped delivering frames after {report.frames} frames")
    if report.reference_fps and report.measured_fps < min_fps_ratio * report.reference_fps:
        report.problems.append(
            f"only {report.measured_fps:.1f} fps received, expected at least "
            f"{min_fps_ratio * report.reference_fps:.1f} (the publisher or this machine may be overloaded)"
        )
    if report.longest_gap_seconds > max_gap_s:
        report.problems.append(
            f"playback stalled for {report.longest_gap_seconds:.2f} s (limit {max_gap_s:.1f} s)"
        )
    if report.blank_frames:
        report.problems.append(
            f"{report.blank_frames} sampled frames were blank or flat grey (decoder or packet-loss problem)"
        )
    return report
