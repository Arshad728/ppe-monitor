"""End-to-end: synthetic clip -> ffmpeg -> MediaMTX -> OpenCV, on a private port.

Skipped automatically when ffmpeg or MediaMTX isn't installed. Uses its own port and log
folder, so it can run while your normal fake camera network is up on 8554.
"""

import socket
import time

import pytest

from ppe_monitor.config import Camera
from ppe_monitor.executables import find_ffmpeg, find_mediamtx
from ppe_monitor.ingestion.rtsp import probe_stream, rtsp_describe
from ppe_monitor.sim.fake_cameras import FakeCameraNetwork, write_server_config
from ppe_monitor.sim.synthetic import make_synthetic_clip

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (find_ffmpeg() and find_mediamtx()), reason="needs ffmpeg and MediaMTX"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def network(tmp_path):
    port = free_port()
    clip = make_synthetic_clip(tmp_path / "clip.mp4", "IT", seconds=3, fps=10, size=(320, 240))
    cams = [Camera(id=f"it{i}", name=f"it{i}", url=f"rtsp://127.0.0.1:{port}/it{i}", sim_source=clip.path)
            for i in (1, 2)]
    net = FakeCameraNetwork(
        cams,
        start_server=True,
        server_config=write_server_config(tmp_path / "mediamtx.yml", port=port, tcp_only=True),
        log_dir=tmp_path / "logs",
    )
    net.start(ready_timeout=20)
    yield net, cams
    net.stop()


def test_streams_play_back_correctly(network):
    """The clip is 3 s long and each stream is read for 4 s, so this also covers the moment the
    clip loops. Regression guard: a multi-threaded decoder in the publisher froze every stream
    for ~(CPU cores / fps) seconds at each loop - 1.1 s on a 10-core Mac."""
    _, cams = network
    for cam in cams:
        report = probe_stream(cam.url, duration_s=4, expected_fps=10, max_gap_s=0.5)
        assert report.passed, report.summary_lines()
        assert (report.width, report.height) == (320, 240)
        assert report.low_latency


def test_crashed_publisher_is_restarted(network):
    net, cams = network
    victim = net._publishers[0]
    victim.process.kill()
    victim.process.wait()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        net.poll()
        if victim.restarts == 1 and victim.process.poll() is None and rtsp_describe(cams[0].url) == 200:
            break
        time.sleep(0.3)
    assert victim.restarts == 1
    assert probe_stream(cams[0].url, duration_s=3, expected_fps=10).passed


def test_stop_leaves_no_processes(tmp_path, network):
    net, cams = network
    processes = [p.process for p in net._publishers] + [net._server]
    net.stop()
    assert all(p.poll() is not None for p in processes)
    assert rtsp_describe(cams[0].url, timeout=0.5) is None


def test_a_camera_that_drops_out_reconnects_without_stalling_the_other(network):
    """Phase 4's "done when": one camera disconnects and comes back; the other never stops."""
    from ppe_monitor.ingestion.reader import LIVE, RECONNECTING, CameraReader

    net, cams = network
    readers = [CameraReader(c.id, c.url, first_wait=0.25, longest_wait=2.0, read_timeout=2.0).start() for c in cams]
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not all(r.stats.state == LIVE for r in readers):
            time.sleep(0.1)
        assert all(r.stats.state == LIVE for r in readers)

        net.pause(cams[0].id)                       # camera 1 loses power
        gaps, last, t_end = [], readers[1].latest().seq, time.monotonic() + 5
        prev_count, prev_t = readers[1].stats.frames, time.monotonic()
        while time.monotonic() < t_end:
            time.sleep(0.25)
            if readers[1].stats.frames > prev_count:
                gaps.append(time.monotonic() - prev_t)
                prev_count, prev_t = readers[1].stats.frames, time.monotonic()
        assert readers[0].stats.state == RECONNECTING
        assert readers[1].stats.state == LIVE and max(gaps) < 1.0 and readers[1].latest().seq > last

        net.resume(cams[0].id)                      # power back
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and readers[0].stats.state != LIVE:
            time.sleep(0.1)
        assert readers[0].stats.state == LIVE and readers[0].stats.reconnects >= 1
        before = readers[0].stats.frames
        time.sleep(1.0)
        assert readers[0].stats.frames > before
    finally:
        for r in readers:
            r.stop()
