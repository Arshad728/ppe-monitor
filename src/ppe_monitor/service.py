"""Several cameras at once, on one machine (book, Chapters 14, 16 and 21).

    camera 1 --reader thread--> newest frame -+
    camera 2 --reader thread--> newest frame -+--> one batch --> detector --> keypoints
    ...                                        |       (every camera due now)
    camera N --reader thread--> newest frame -+
            --> per camera: tracker -> PPE rule -> zone rules -> events -> events.jsonl + snapshots

- **One reader thread per camera** (ingestion/reader.py) keeps only the newest frame and
  reconnects on its own. A camera that drops out never holds up the others.
- **A fixed processing rate per camera** (`process_fps`, default 10). Cameras usually send 15–30
  frames a second, and analysing every one is wasted work: people don't move far in 0.1 s.
  Phase 2 showed that 10 fps keeps groups of people apart; 5 fps does not.
- **One clock for all cameras, and batching.** `process_fps` times a second (a "tick"), the
  newest frame of every camera that has sent a new one since the last tick goes into one call
  of the detector, and one call of the keypoint model for the frames with people in them. On a
  GPU, one call of N frames costs much less than N calls. A camera with nothing new at a tick
  (down, or slower than `process_fps`) simply sits that tick out. Backends that take one image
  at a time (Core ML) are called once per frame, in the same loop.
- **Falling behind.** When the machine can't keep up, every camera slows down equally, and
  each one is still analysed on its newest frame: the lag stays small and the frame rate drops.
  The status table shows it.

Each camera keeps its own tracker, rules and event engine (pipeline.PPEMonitor), so events, IDs
and snapshots never mix between cameras.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .draw import draw, snapshot
from .ingestion.reader import LIVE, CameraReader, Frame
from .pipeline import REUSE_POSES, FrameResult, PPEMonitor, Settings
from .rules.engine import RuleSet
from .config import PROJECT_ROOT
from .vision.backends import SPECS, Backend, available, is_macos

STREAMS_FILE = PROJECT_ROOT / "configs" / "streams.yaml"


@dataclass(frozen=True)
class StreamSettings:
    process_fps: float = 10.0
    backend: str = "auto"
    imgsz: int = 640
    first_wait: float = 0.5
    longest_wait: float = 10.0
    read_timeout: float = 5.0
    display_fps: float = 4.0

    @classmethod
    def load(cls, path: str | Path = STREAMS_FILE) -> "StreamSettings":
        import yaml

        path = Path(path)
        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
        rc = raw.pop("reconnect", None) or {}
        unknown = (set(raw) | set(rc)) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path.name}: unknown settings {sorted(unknown)}")
        s = cls(**{k: (str(v) if k == "backend" else int(v) if k == "imgsz" else float(v)) for k, v in {**raw, **rc}.items()})
        if s.backend != "auto" and s.backend not in SPECS:
            raise ValueError(f"{path.name}: backend must be auto or one of {', '.join(SPECS)}")
        if not 0.5 <= s.process_fps <= 60:
            raise ValueError(f"{path.name}: process_fps must be between 0.5 and 60")
        return s

    def reader_options(self) -> dict:
        return {"first_wait": self.first_wait, "longest_wait": self.longest_wait, "read_timeout": self.read_timeout}


def choose_backend(wanted: str, weights: Path, pose_weights: Path | None) -> str:
    """`auto`: Core ML on a Mac if exported, else PyTorch on a GPU, else ONNX on the CPU if exported,
    else PyTorch (on the CPU)."""
    if wanted != "auto":
        return wanted
    ok = {name for name, why in available(weights, pose_weights).items() if not why}
    from .device import best_device

    for name in (["coreml"] if is_macos() else []) + (["pytorch"] if best_device() != "cpu" else []) + ["onnx"]:
        if name in ok:
            return name
    return "pytorch"


@dataclass(frozen=True)
class CameraSource:
    id: str
    source: str               # RTSP URL, or a video file standing in for a camera
    transport: str = "tcp"


@dataclass
class CameraStats:
    processed: int = 0
    events: int = 0
    skipped: int = 0                                         # frames that arrived but were never analysed
    done_times: deque = field(default_factory=lambda: deque(maxlen=400))    # monotonic times of analysed frames
    latencies: deque = field(default_factory=lambda: deque(maxlen=400))     # seconds from arrival to done

    def fps(self, now: float, over: float = 5.0) -> float:
        recent = [t for t in self.done_times if t >= now - over]
        return len(recent) / over

    def latency_ms(self, q: float = 0.5) -> float:
        if not self.latencies:
            return float("nan")
        values = sorted(self.latencies)
        return 1000 * values[min(len(values) - 1, int(q * len(values)))]


class EventSink:
    """events.jsonl and one snapshot per event, per run."""

    def __init__(self, out_dir: Path | None, snapshots: bool = True):
        self.out_dir, self.snapshots = out_dir, snapshots
        self._file = None
        if out_dir is not None:
            (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
            self._file = (out_dir / "events.jsonl").open("a", encoding="utf-8")
            self._cams = (out_dir / "cameras.jsonl").open("a", encoding="utf-8")

    def event(self, ev, frame: np.ndarray, result: FrameResult) -> None:
        if self._file is None:
            return
        if self.snapshots:
            path = self.out_dir / "snapshots" / f"{ev.event_id}.jpg"
            cv2.imwrite(str(path), snapshot(frame, result, ev), [cv2.IMWRITE_JPEG_QUALITY, 90])
            ev.snapshot = str(path.relative_to(self.out_dir))
        self._file.write(json.dumps(ev.to_dict()) + "\n")
        self._file.flush()

    def camera(self, camera: str, state: str, detail: str, when) -> None:
        if self._file is None:
            return
        self._cams.write(json.dumps({"camera": camera, "state": state, "detail": detail,
                                     "time": when.isoformat(timespec="seconds")}) + "\n")
        self._cams.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._cams.close()
            self._file = None


class MultiCameraMonitor:
    def __init__(self, sources: list[CameraSource], backend: Backend, settings: Settings | None = None,
                 rules: RuleSet | None = None, *, process_fps: float = 10.0, imgsz: int = 640,
                 out_dir: Path | None = None, snapshots: bool = True, keep_frames: bool = False,
                 reader_options: dict | None = None, sinks: list | None = None):
        """sinks: more receivers of events and camera states besides the run's events.jsonl, such as
        the database (backend/sink.py DbSink). Each must return at once: they run in the video loop."""
        self.backend = backend
        self.s = settings or Settings.load()
        rules = rules or RuleSet.load()
        self.period = 1.0 / process_fps
        self.imgsz = imgsz
        self.use_pose = bool(self.s.pose["enabled"]) and backend.pose is not None
        self.ids = [src.id for src in sources]
        self.readers = {src.id: CameraReader(src.id, src.source, transport=src.transport, **(reader_options or {}))
                        for src in sources}
        self.monitors = {src.id: PPEMonitor(backend.detector, src.id, process_fps, self.s, imgsz=imgsz,
                                            pose_model=False, rules=rules) for src in sources}
        self.stats = {cid: CameraStats() for cid in self.ids}
        self.sink = EventSink(out_dir, snapshots)
        self.sinks = list(sinks or [])
        self.keep_frames = keep_frames
        self.latest: dict[str, tuple[np.ndarray, FrameResult]] = {}   # for display (keep_frames=True)
        self._last_seq = {cid: 0 for cid in self.ids}
        self._next_tick = 0.0                                        # one clock shared by every camera
        self._t0: dict[str, float] = {}
        self._state = {cid: "" for cid in self.ids}
        self._turn = 0
        self.batch_sizes: deque = deque(maxlen=400)
        self.step_times: deque = deque(maxlen=400)

    def start(self, warmup: bool = True) -> "MultiCameraMonitor":
        if warmup:                         # load the models before the first frames, not during them
            self.backend.warmup(self.imgsz)
        for r in self.readers.values():
            r.start()
        return self

    def stop(self) -> None:
        for r in self.readers.values():
            r.stop()
        self.sink.close()

    # -- one round --------------------------------------------------------------------------------
    def _due_frames(self) -> list[tuple[str, Frame]]:
        """The newest frame of every camera that has a new one, first camera rotating each tick."""
        order = self.ids[self._turn:] + self.ids[:self._turn]
        self._turn = (self._turn + 1) % max(1, len(self.ids))
        due = []
        for cid in order:
            f = self.readers[cid].latest(after_seq=self._last_seq[cid])
            if f is not None:
                due.append((cid, f))
        return due

    def _watch_states(self) -> None:
        for cid, r in self.readers.items():
            st = r.stats.state
            if st != self._state[cid]:
                self._state[cid] = st
                detail, when = r.stats.last_error if st != LIVE else "", r.wall(time.monotonic())
                self.sink.camera(cid, st, detail, when)
                for s in self.sinks:
                    s.camera(cid, st, detail, when)

    def step(self) -> list[tuple[str, FrameResult]]:
        """Analyse the newest frame of every camera that is due. Returns what happened; returns []
        (after a short sleep) when no camera has a new frame due yet."""
        self._watch_states()
        now = time.monotonic()
        if now < self._next_tick:                                    # not time yet
            time.sleep(min(0.01, self._next_tick - now))
            return []
        due = self._due_frames()
        if not due:                                                  # nothing new from any camera yet
            time.sleep(0.002)
            return []
        t_start = time.monotonic()
        # next tick one period on; when a batch took longer than that, start again from now
        # (every camera slows down equally, and is still analysed on its newest frame)
        self._next_tick = max(self._next_tick + self.period, t_start)
        frames = [f.image for _, f in due]
        low = self.s.tracking.low
        min_conf = min(low, self.s.thresholds["helmet"], self.s.thresholds["vest"])
        dets = self.backend.detect(frames, conf=min_conf, imgsz=self.imgsz)
        poses: list = [None] * len(due)
        if self.use_pose:
            # keypoints for the cameras whose turn it is (pose.every), on frames with people in them;
            # the other cameras carry each person's last keypoints (PPEMonitor.pose_due)
            turn = [self.monitors[cid].pose_due() for cid, _ in due]
            people = [any(x.cls == 0 and x.score >= low for x in d) for d in dets]
            with_people = [i for i in range(len(due)) if turn[i] and people[i]]
            found = self.backend.keypoints([frames[i] for i in with_people], conf=self.s.pose["conf"],
                                           keypoint_conf=self.s.pose["keypoint_conf"],
                                           imgsz=self.imgsz) if with_people else []
            poses = [REUSE_POSES if people[i] and not turn[i] else [] for i in range(len(due))]
            for i, p in zip(with_people, found):
                poses[i] = p
        out = []
        for i, (cid, f) in enumerate(due):
            t0 = self._t0.setdefault(cid, f.arrived)
            r = self.monitors[cid].process(f.image, f.arrived - t0, detections=dets[i], poses=poses[i], now=f.wall)
            st = self.stats[cid]
            if self._last_seq[cid]:
                st.skipped += max(0, f.seq - self._last_seq[cid] - 1)
            self._last_seq[cid] = f.seq
            done = time.monotonic()
            st.processed += 1
            st.done_times.append(done)
            st.latencies.append(done - f.arrived)
            for ev in r.events:
                st.events += 1
                self.sink.event(ev, f.image, r)
                for s in self.sinks:
                    s.event(ev, f.image, r)
            if self.keep_frames:
                self.latest[cid] = (f.image, r)
            out.append((cid, r))
        self.batch_sizes.append(len(due))
        self.step_times.append(time.monotonic() - t_start)
        return out

    def run(self, seconds: float | None = None, on_step=None) -> None:
        end = None if seconds is None else time.monotonic() + seconds
        while end is None or time.monotonic() < end:
            results = self.step()
            if on_step is not None and on_step(results) is False:
                break

    # -- reporting --------------------------------------------------------------------------------
    def status_rows(self) -> list[dict]:
        now = time.monotonic()
        rows = []
        for cid in self.ids:
            r, st = self.readers[cid], self.stats[cid]
            down = r.stats.down_since
            rows.append({"camera": cid, "state": r.stats.state, "fps": st.fps(now), "lag_ms": st.latency_ms(0.5),
                         "lag95_ms": st.latency_ms(0.95), "processed": st.processed, "events": st.events,
                         "reconnects": r.stats.reconnects, "skipped": st.skipped,
                         "down_for": (now - down) if down is not None else 0.0, "error": r.stats.last_error})
        return rows

    def status_text(self) -> str:
        lines = [f"{'camera':10s} {'state':12s} {'fps':>5s} {'lag ms':>7s} {'p95':>6s} {'events':>6s} {'reconn':>6s}"]
        for row in self.status_rows():
            lag = "-" if row["lag_ms"] != row["lag_ms"] else f"{row['lag_ms']:.0f}"
            p95 = "-" if row["lag95_ms"] != row["lag95_ms"] else f"{row['lag95_ms']:.0f}"
            extra = f"  down {row['down_for']:.0f} s: {row['error']}" if row["state"] != LIVE and row["down_for"] else ""
            lines.append(f"{row['camera']:10s} {row['state']:12s} {row['fps']:5.1f} {lag:>7s} {p95:>6s} "
                         f"{row['events']:6d} {row['reconnects']:6d}{extra}")
        if self.batch_sizes:
            lines.append(f"batch of {statistics.mean(self.batch_sizes):.1f} frames on average, "
                         f"{1000 * statistics.mean(self.step_times):.0f} ms per batch")
        return "\n".join(lines)

    def mosaic(self, tile_width: int = 640) -> np.ndarray | None:
        """All cameras' latest analysed frames, annotated, in a grid (needs keep_frames=True)."""
        if not self.latest:
            return None
        tiles = []
        for cid in self.ids:
            if cid in self.latest:
                img, r = self.latest[cid]
                h, w = img.shape[:2]
                small = cv2.resize(img, (tile_width, int(h * tile_width / w)))
                tile = draw(small, r, f"{cid}  {self.readers[cid].stats.state}  {self.stats[cid].fps(time.monotonic()):.1f} fps")
            else:
                tile = np.zeros((tile_width * 9 // 16, tile_width, 3), np.uint8)
                cv2.putText(tile, f"{cid}: {self.readers[cid].stats.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
            tiles.append(tile)
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        cols = int(np.ceil(np.sqrt(len(tiles))))
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        return np.vstack(rows)
