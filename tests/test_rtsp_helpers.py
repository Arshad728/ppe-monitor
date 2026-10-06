import socket
import threading
from pathlib import Path

import pytest

from ppe_monitor.ingestion.rtsp import rtsp_describe
from ppe_monitor.sim.fake_cameras import build_publish_command, host_port


class FakeRtspServer:
    """Answers every connection with a fixed RTSP status line, and records the request."""

    def __init__(self, status_line: str):
        self.status_line = status_line
        self.requests: list[str] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                self.requests.append(conn.recv(4096).decode())
                conn.sendall(f"{self.status_line}\r\nCSeq: 1\r\n\r\n".encode())

    def close(self):
        self.sock.close()


@pytest.mark.parametrize("status_line, code", [("RTSP/1.0 200 OK", 200),
                                               ("RTSP/1.0 404 Not Found", 404),
                                               ("RTSP/1.0 401 Unauthorized", 401)])
def test_rtsp_describe_returns_status_code(status_line, code):
    server = FakeRtspServer(status_line)
    try:
        assert rtsp_describe(f"rtsp://127.0.0.1:{server.port}/cam1") == code
    finally:
        server.close()


def test_rtsp_describe_never_sends_credentials():
    server = FakeRtspServer("RTSP/1.0 200 OK")
    try:
        rtsp_describe(f"rtsp://admin:hunter2@127.0.0.1:{server.port}/cam1")
        assert server.requests and "hunter2" not in server.requests[0]
        assert server.requests[0].startswith(f"DESCRIBE rtsp://127.0.0.1:{server.port}/cam1 RTSP/1.0")
    finally:
        server.close()


def test_rtsp_describe_unreachable_server_returns_none():
    with socket.socket() as s:  # grab a free port, then leave it closed
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert rtsp_describe(f"rtsp://127.0.0.1:{port}/cam1", timeout=0.5) is None


def test_publish_command_reencodes_for_fast_joining():
    cmd = build_publish_command("ffmpeg", Path("clip.mp4"), "rtsp://localhost:8554/cam1", encoder="libx264")
    joined = " ".join(cmd)
    assert "-re -stream_loop -1 -threads 1 -i clip.mp4" in joined  # real time, looping, no freeze at loop
    assert "-c:v libx264" in joined and "-bf 0" in joined        # H.264 without B-frames
    assert "expr:gte(t,n_forced*1)" in joined                    # a keyframe every second
    assert cmd[-5:] == ["-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://127.0.0.1:8554/cam1"]   # no name lookup


def test_publish_command_copy_mode_skips_encoding():
    cmd = build_publish_command("ffmpeg", Path("clip.mp4"), "rtsp://localhost:8554/cam1", copy=True)
    assert "-c:v copy" in " ".join(cmd)
    assert "-force_key_frames" not in cmd


def test_host_port_defaults_to_rtsp_port():
    assert host_port("rtsp://localhost:8554/cam1") == ("localhost", 8554)
    assert host_port("rtsp://10.0.0.5/stream") == ("10.0.0.5", 554)


def test_ffmpeg_options_do_not_leak_after_opening_a_stream(monkeypatch):
    """Regression test: low-latency RTSP options left in the environment made later reads of
    video FILES silently drop half their frames."""
    import os

    from ppe_monitor.ingestion.rtsp import open_capture

    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    open_capture("rtsp://127.0.0.1:1/nothing", open_timeout_s=1).release()  # port 1: fails fast
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ

    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "user;value")
    open_capture("rtsp://127.0.0.1:1/nothing", open_timeout_s=1).release()
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "user;value"  # a user's own setting survives
