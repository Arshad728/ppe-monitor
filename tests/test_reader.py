"""Camera readers: newest frame only, real-time pacing for files, and reconnecting with back-off."""

import time

import cv2
import numpy as np
import pytest

from ppe_monitor.ingestion.reader import CONNECTING, LIVE, STOPPED, CameraReader, backoff


def clip(path, seconds=1.0, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(int(seconds * fps)):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(img, str(k), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(img)
    w.release()
    return path


def test_backoff_doubles_up_to_a_limit():
    assert [backoff(a) for a in range(8)] == [0.5, 1, 2, 4, 8, 10, 10, 10]


def test_a_file_plays_in_real_time_and_loops(tmp_path):
    r = CameraReader("t", str(clip(tmp_path / "a.mp4", seconds=1.0, fps=10))).start()
    try:
        time.sleep(2.5)
        f = r.latest()
        assert f is not None and r.stats.state == LIVE
        assert 18 <= r.stats.frames <= 30          # ~10 per second, and it went round more than once
        assert r.latest(after_seq=f.seq) is None or r.latest(after_seq=f.seq).seq > f.seq
        assert abs((f.wall - r.wall(f.arrived)).total_seconds()) < 1e-6
    finally:
        r.stop()


def test_a_file_can_play_once_like_a_camera_that_is_switched_off(tmp_path):
    """The Phase 6 evaluation plays each test clip once; the frame times give the clip's own clock."""
    r = CameraReader("t", str(clip(tmp_path / "a.mp4", seconds=1.0, fps=10)), loop_files=False).start()
    try:
        time.sleep(2.0)
        assert r.stats.ended and r.stats.state == STOPPED
        assert r.stats.frames == 10 and r.stats.reconnects == 0
        assert (r.stats.last_frame - r.stats.first_frame) == pytest.approx(0.9, abs=0.1)
    finally:
        r.stop()


def test_only_the_newest_frame_is_kept(tmp_path):
    r = CameraReader("t", str(clip(tmp_path / "a.mp4", seconds=2.0, fps=10))).start()
    try:
        time.sleep(1.2)
        f1 = r.latest()
        time.sleep(0.5)                              # a slow consumer: ~5 frames went by
        f2 = r.latest(after_seq=f1.seq)
        assert f2.seq - f1.seq >= 3                  # it skips straight to the newest
    finally:
        r.stop()


def test_a_missing_camera_keeps_retrying_without_blocking(tmp_path):
    r = CameraReader("t", str(tmp_path / "nothing.mp4"), first_wait=0.05, longest_wait=0.2).start()
    time.sleep(0.6)
    assert r.latest() is None and r.stats.state == CONNECTING and r.stats.last_error
    t = time.monotonic()
    r.stop()
    assert time.monotonic() - t < 1.0                # stopping never waits for a back-off to finish
