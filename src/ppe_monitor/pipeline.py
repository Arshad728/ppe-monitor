"""One camera's processing chain, frame by frame (book, Chapter 14 "the vision engine"):

    frame -> detector -> person tracker -> PPE verdict per person -> rules -> event engine -> events
                  +-> keypoint model (optional): head, feet, posture      (zones, hours)

The rules (rules/engine.py, configs/rules.yaml) turn each person's PPE verdict and position into
one vote per rule: no helmet, no vest, standing in a restricted zone, each only during its active
hours. The event engine turns votes into events of one shape, whatever the rule.

`PPEMonitor.process(frame, t)` runs one frame and returns what happened, so the same chain serves
the live monitor (scripts/monitor.py), offline evaluation (scripts/evaluate_events.py) and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .rules.engine import CameraRules, RuleSet
from .rules.events import Event, EventEngine, EventSettings
from .rules.ppe import Rules, Verdict, judge, load_config
from .vision.matching import Det
from .vision.pose import Pose, carried, match_poses, poses_from_result
from .vision.tracking import PersonTracker, Track, TrackerSettings

PPE_CONFIG = PROJECT_ROOT / "configs" / "ppe.yaml"
NAMES = ("person", "helmet", "vest")


POSE_DEFAULTS = {"enabled": True, "model": "yolo26n-pose.pt", "keypoint_conf": 0.5, "conf": 0.25, "every": 1}


class _Reuse:
    def __repr__(self) -> str:
        return "REUSE_POSES"


REUSE_POSES = _Reuse()   # process(poses=REUSE_POSES): the keypoint model skipped this frame; carry the last ones


@dataclass
class Settings:
    thresholds: dict
    rules: Rules
    tracking: TrackerSettings
    events: EventSettings
    model: str | None = None
    pose: dict = field(default_factory=lambda: dict(POSE_DEFAULTS))

    @classmethod
    def load(cls, path: str | Path = PPE_CONFIG) -> "Settings":
        cfg = load_config(path)
        thresholds = {"person": 0.4, "helmet": 0.25, "vest": 0.25, **(cfg.get("thresholds") or {})}
        pose = {**POSE_DEFAULTS, **(cfg.get("pose") or {})}
        unknown = set(pose) - set(POSE_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown pose settings: {sorted(unknown)}")
        if not isinstance(pose["every"], int) or isinstance(pose["every"], bool) or pose["every"] < 1:
            raise ValueError(f"pose: every must be a whole number, 1 or more (1 = every frame), not {pose['every']!r}")
        return cls(thresholds, Rules.from_dict(cfg.get("rules")), TrackerSettings.from_dict(cfg.get("tracking")),
                   EventSettings.from_dict(cfg.get("events")), cfg.get("model"), pose)


@dataclass
class FrameResult:
    t: float
    frame_index: int
    tracks: list[Track]
    verdicts: dict[int, Verdict]             # track id -> this frame's verdict
    status: dict[int, dict[str, str]]        # track id -> {rule id: ok / pending / violation}
    helmets: list[Det]
    vests: list[Det]
    events: list[Event] = field(default_factory=list)
    poses: dict[int, Pose | None] = field(default_factory=dict)  # track id -> keypoints (carried from an earlier
                                                                  # frame when the keypoint model skipped this one)
    feet: dict[int, tuple | None] = field(default_factory=dict)   # track id -> where they stand (None: out of view)
    inside: dict[int, list[str]] = field(default_factory=dict)    # track id -> zones they stand in
    rules: CameraRules | None = None                              # this camera's rules and zones
    active: set[str] = field(default_factory=set)                 # rules switched on in this frame
    now: datetime | None = None                                   # wall-clock time of the frame


class PPEMonitor:
    def __init__(self, model, camera: str, fps: float, settings: Settings | None = None, *, imgsz: int = 640,
                 device: str | None = None, pose_model=None, rules: RuleSet | CameraRules | None = None,
                 start_time: datetime | None = None):
        """pose_model: a loaded keypoint model, None to load the one in the settings (if enabled),
        or False for no keypoints. rules: None = configs/rules.yaml and configs/zones.yaml.
        start_time: the wall-clock time of t = 0 (default: now); a frame at t seconds is
        start_time + t, which decides the rules' active hours."""
        self.model, self.camera, self.imgsz, self.device = model, camera, imgsz, device
        self.s = settings or Settings.load()
        if rules is None:
            rules = RuleSet.load()
        self.rules = rules.for_camera(camera) if isinstance(rules, RuleSet) else rules
        self.start_time = (start_time or datetime.now()).astimezone()
        if pose_model is None and self.s.pose["enabled"]:
            from .vision.detector import load_model
            pose_model = load_model(self.s.pose["model"])
        self.pose_model = pose_model or None
        self.tracker = PersonTracker(fps, self.s.tracking)
        self.engine = EventEngine(camera, self.s.events, self.rules.rules)
        self.frame_index = 0
        names = [model.names[i] for i in sorted(model.names)] if hasattr(model, "names") else list(NAMES)
        if list(names[:3]) != list(NAMES):
            raise ValueError(f"model classes {names} are not {list(NAMES)}")
        self.min_conf = min(self.s.tracking.low, self.s.thresholds["helmet"], self.s.thresholds["vest"])
        self.pose_every = int(self.s.pose["every"])
        self._since_pose: int | None = None           # frames since the keypoint model last ran on people
        self._pose_memory: dict[int, tuple[Pose, tuple]] = {}   # track id -> (keypoints, their box then)

    def pose_due(self) -> bool:
        """Should the keypoint model run on the next frame? Every `pose.every`-th frame with people in
        it; the frames between reuse each person's last keypoints (see vision/pose.py, carried)."""
        return self._since_pose is None or self._since_pose + 1 >= self.pose_every

    def detect(self, frame: np.ndarray) -> list[Det]:
        r = self.model.predict(frame, conf=self.min_conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        b = r.boxes
        return [Det(int(c), tuple(float(v) for v in xy), float(s))
                for c, xy, s in zip(b.cls.tolist(), b.xyxyn.tolist(), b.conf.tolist())]

    def detect_poses(self, frame: np.ndarray) -> list[Pose]:
        h, w = frame.shape[:2]
        r = self.pose_model.predict(frame, conf=self.s.pose["conf"], imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return poses_from_result(r, w / h, self.s.pose["keypoint_conf"])

    def process(self, frame: np.ndarray, t: float, detections: list[Det] | None = None,
                poses: list[Pose] | None = None, frame_index: int | None = None,
                now: datetime | None = None) -> FrameResult:
        """Run one frame. `t` in seconds. Pass `detections` (and `poses`) to skip the models (tests, replays).
        poses: this frame's keypoints; REUSE_POSES to carry the last ones (a frame the keypoint model
        skips, see pose_due); None to run the keypoint model here when it is due (if there is one).
        `frame_index`: the frame's number in the video, when not every frame is processed.
        `now`: the frame's wall-clock time (default: start_time + t)."""
        if frame_index is not None:
            self.frame_index = frame_index
        h, w = frame.shape[:2]
        dets = self.detect(frame) if detections is None else detections
        th = self.s.thresholds
        people = [d for d in dets if d.cls == 0 and d.score >= self.s.tracking.low]
        reuse = poses is REUSE_POSES
        if poses is None and self.pose_model is not None:
            if not people:
                poses = []                                        # nobody in view: skip the keypoint model
            elif self.pose_due():
                poses = self.detect_poses(frame)
            else:
                reuse = True
        helmets = [d for d in dets if d.cls == 1 and d.score >= th["helmet"]]
        vests = [d for d in dets if d.cls == 2 and d.score >= th["vest"]]
        tracks = self.tracker.update(people, (w, h), t)
        input_height = self.imgsz * h / max(h, w)
        if reuse:                                  # keypoints from the last run, moved onto today's boxes
            mem = self._pose_memory
            matched = [carried(mem[tr.track_id][0], mem[tr.track_id][1], tr.box) if tr.track_id in mem else None
                       for tr in tracks]
            if self._since_pose is not None:
                self._since_pose += 1
        elif poses is not None:
            matched = match_poses([tr.box for tr in tracks], poses)
            self._pose_memory = {tr.track_id: (m, tr.box) for tr, m in zip(tracks, matched) if m is not None}
            self._since_pose = 0 if people else None   # nobody in view: run as soon as someone appears
        else:
            matched = None
        verdicts = judge([tr.box for tr in tracks], [d.box for d in helmets], [d.box for d in vests], self.s.rules,
                         input_height, matched)
        by_id = {tr.track_id: v for tr, v in zip(tracks, verdicts)}
        now = now or self.start_time + timedelta(seconds=t)
        active = self.rules.active(now)
        people, feet, inside = [], {}, {}
        for k, (tr, v) in enumerate(zip(tracks, verdicts)):
            votes, feet[tr.track_id], inside[tr.track_id] = self.rules.votes(v, tr.box, matched[k] if matched else None,
                                                                             active)
            people.append((tr.track_id, tr.box, votes))
        events = self.engine.update(t, self.frame_index, people, active, now)
        status = {tr.track_id: self.engine.status(tr.track_id) for tr in tracks}
        result = FrameResult(t, self.frame_index, tracks, by_id, status, helmets, vests, events,
                             {tr.track_id: m for tr, m in zip(tracks, matched)} if matched is not None else {},
                             feet, inside, self.rules, active, now)
        self.frame_index += 1
        return result
