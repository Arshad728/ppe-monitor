"""The multi-camera engine, with a stand-in model: rate per camera, batching, a dead camera."""

import time

import cv2
import numpy as np

from ppe_monitor.pipeline import Settings
from ppe_monitor.rules.engine import RuleSet
from ppe_monitor.service import CameraSource, MultiCameraMonitor, StreamSettings
from ppe_monitor.vision.backends import SPECS, Backend
from ppe_monitor.vision.matching import Det
from ppe_monitor.vision.pose import Pose


class FakeModel:
    names = {0: "person", 1: "helmet", 2: "vest"}


class CountingBackend(Backend):
    """Always sees one bare-headed worker; remembers how many frames each call got."""

    def __init__(self):
        super().__init__(SPECS["pytorch"], FakeModel(), None, "cpu", 16, 16, (None, None))
        self.calls = []

    def detect(self, frames, *, conf, imgsz=640):
        self.calls.append(len(frames))
        time.sleep(0.01)
        return [[Det(0, (0.4, 0.3, 0.5, 0.8), 0.9), Det(2, (0.41, 0.4, 0.49, 0.55), 0.8)] for _ in frames]


def clip(path, seconds=2.0, fps=20, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(int(seconds * fps)):
        w.write(np.full((size[1], size[0], 3), 90, np.uint8))
    w.release()
    return str(path)


def test_each_camera_is_analysed_at_the_set_rate_in_batches_and_a_dead_one_does_not_block(tmp_path):
    video = clip(tmp_path / "a.mp4")
    sources = [CameraSource("a", video), CameraSource("b", video), CameraSource("dead", str(tmp_path / "gone.mp4"))]
    backend = CountingBackend()
    mon = MultiCameraMonitor(sources, backend, Settings.load(), RuleSet.default(), process_fps=5, out_dir=tmp_path / "out",
                             reader_options={"first_wait": 0.1, "longest_wait": 0.2})
    mon.start()
    try:
        mon.run(6.0)
    finally:
        mon.stop()
    a, b, dead = (mon.stats[c] for c in ("a", "b", "dead"))
    assert 25 <= a.processed <= 32 and 25 <= b.processed <= 32     # 5 per second, not the files' 20
    assert dead.processed == 0 and mon.readers["dead"].stats.state == "stopped"
    assert max(backend.calls) == 2                                   # both live cameras in one call
    assert a.skipped > 0                                             # stale frames were dropped, not queued
    events = (tmp_path / "out" / "events.jsonl").read_text().splitlines()
    assert {line.split('"camera": "')[1][0] for line in events} == {"a", "b"}   # no helmet: one event per camera
    assert "dead" in (tmp_path / "out" / "cameras.jsonl").read_text()


def test_stream_settings_file(tmp_path):
    p = tmp_path / "streams.yaml"
    p.write_text("process_fps: 8\nbackend: onnx\nreconnect: {first_wait: 1, longest_wait: 10, read_timeout: 3}\n")
    s = StreamSettings.load(p)
    assert (s.process_fps, s.backend, s.first_wait, s.read_timeout) == (8, "onnx", 1, 3)
    p.write_text("backend: tensorrt\n")
    try:
        StreamSettings.load(p)
        assert False
    except ValueError as exc:
        assert "backend" in str(exc)
    assert StreamSettings.load().process_fps == 10                  # the project's own file


class PoseCountingBackend(CountingBackend):
    """CountingBackend with a keypoint model that counts the frames it is given."""

    def __init__(self):
        super().__init__()
        self.pose = object()
        self.pose_frames = 0

    def keypoints(self, frames, *, conf, keypoint_conf, imgsz=640):
        self.pose_frames += len(frames)
        xy, c = np.zeros((17, 2)), np.zeros(17)
        for i, p in {0: (0.45, 0.33), 5: (0.43, 0.40), 6: (0.47, 0.40), 11: (0.44, 0.58), 12: (0.46, 0.58)}.items():
            xy[i], c[i] = p, 0.9
        return [[Pose((0.4, 0.3, 0.5, 0.8), xy, c)] for _ in frames]


def test_keypoints_run_on_every_second_frame_of_each_camera(tmp_path):
    video = clip(tmp_path / "a.mp4")
    settings = Settings.load()
    settings.pose["every"] = 2
    backend = PoseCountingBackend()
    mon = MultiCameraMonitor([CameraSource("a", video), CameraSource("b", video)], backend, settings,
                             RuleSet.default(), process_fps=5, out_dir=None)
    mon.start(warmup=False)
    try:
        mon.run(3.0)
    finally:
        mon.stop()
    processed = mon.stats["a"].processed + mon.stats["b"].processed
    assert processed >= 20
    assert abs(backend.pose_frames - processed / 2) <= 2       # half the frames, not all of them
    r = mon.monitors["a"]
    assert r.pose_every == 2 and r._pose_memory                 # the frames between carry the last keypoints
