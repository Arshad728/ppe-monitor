"""Giving each person a track ID that stays the same from frame to frame (book, Chapter 9).

Uses ByteTrack as implemented in Ultralytics, but fed only with the *person* detections: helmets
and vests don't need identities, they are matched to a person afresh in every frame
(rules/ppe.py). Feeding detections in directly, instead of calling `model.track()`, keeps the
tracker independent of the detector, so tests can drive it with hand-made detections.

ByteTrack's idea: a person hidden behind a pole is often still detected, just with low confidence.
Detections above `high` start and continue tracks. Detections between `low` and `high` are only used
to keep existing tracks alive, never to start new ones. A track that gets no detection at all is kept
"lost" for `lost_seconds`, and picks its old ID back up if the person reappears nearby.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils.matching import fuse_score
from ultralytics.utils.metrics import bbox_ioa

from .matching import Det


@dataclass(frozen=True)
class TrackerSettings:
    high: float = 0.40          # detections at or above this start and continue tracks
    low: float = 0.10           # detections between low and high only keep existing tracks alive
    new_track: float = 0.50     # a new track needs at least this confidence (fewer tracks on false people)
    lost_seconds: float = 2.0   # how long a track survives without any detection
    match: float = 0.7          # association threshold on (1 - IoU x score): IoU x score >= 0.3 to match
    lost_damping: float = 0.8   # a lost track's speed is multiplied by this every frame (1 = plain ByteTrack)
    duplicate: float = 0.85     # a person box this much inside a bigger one, with the head in the same
                                # place, is the same person detected twice (see merge_duplicates)
    relink: float = 0.5         # a new track starting within this many person-heights of where a
                                # recently lost track was last seen gets its ID back
    walk_speed: float = 0.8     # ... plus this many person-heights per second the track was lost
    buffer_rate: float = 1.8    # boxes are widened by (buffer_rate x seconds between frames - 0.1) of their size

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrackerSettings":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown tracking settings: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items()})


@dataclass(frozen=True)
class Track:
    track_id: int
    box: tuple[float, float, float, float]  # the person's detected box this frame, 0-1 fractions
    score: float
    det_index: int                          # index into the person detections passed to update()


class _Boxes:
    """The small subset of Ultralytics' Boxes interface that BYTETracker reads."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        return _Boxes(self.xyxy[mask], self.conf[mask], self.cls[mask])


def merge_duplicates(people: list[Det], containment: float) -> list[Det]:
    """Drop the second box when the detector finds one person twice.

    The detector sometimes returns two boxes for one person, e.g. a full-body box and a shorter
    upper-body box. Their overlap (IoU) can be too small for non-maximum suppression to remove
    either, so the tracker would follow two "people" and every violation would be reported twice.
    A box counts as a duplicate when it is clearly smaller than another person box, most of it
    (`containment`) lies inside that box, and the top edges and centre lines match: the same head.
    Two different people standing one behind the other have their heads in different places, and
    two people passing each other have boxes of the same size, so both are kept.
    """
    order = sorted(range(len(people)), key=lambda i: -people[i].score)
    kept: list[int] = []
    for i in order:
        a = people[i].box
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        duplicate = False
        for j in kept:
            b = people[j].box
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            small, big = min(area_a, area_b), max(area_a, area_b)
            big_h = max(a[3] - a[1], b[3] - b[1])
            big_w = max(a[2] - a[0], b[2] - b[0])
            same_head = abs(a[1] - b[1]) <= 0.15 * big_h and abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= 0.25 * big_w
            truncated = small <= 0.8 * big  # one box is a cut-down version of the other
            if small > 0 and inter / small >= containment and same_head and truncated:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return [people[i] for i in sorted(kept)]


class _ByteTrack(BYTETracker):
    """ByteTrack whose lost tracks slow down instead of coasting at their last speed.

    While a person is being hidden (behind a pole, say), their box shrinks from one side, and the
    Kalman filter takes that for fast sideways movement. A lost track that kept that speed would
    drift onto a neighbour and could take over their ID. Damping keeps a real walker's short-term
    motion but stops the drift.
    """

    damping = 0.8
    buffer = 0.0   # set per frame by PersonTracker: how much to widen boxes before comparing them

    def multi_predict(self, tracks):
        for t in tracks:
            if t.state != TrackState.Tracked and t.mean is not None:
                t.mean[4:8] *= self.damping
        super().multi_predict(tracks)

    def get_dists(self, tracks, detections):
        """IoU cost, on boxes widened by `buffer` of their size on every side ("buffered IoU").
        When frames are far apart (a slow processing rate, frames skipped to stay live), a walking
        person's boxes in consecutive frames barely overlap, and plain ByteTrack would start a new
        track every frame. Widening both boxes in proportion to the time gap restores the overlap."""
        if self.buffer <= 0 or not tracks or not detections:
            return super().get_dists(tracks, detections)
        a = np.array([t.xyxy for t in tracks], dtype=np.float64)
        b = np.array([d.xyxy for d in detections], dtype=np.float64)
        for m in (a, b):
            w, h = m[:, 2] - m[:, 0], m[:, 3] - m[:, 1]
            m[:, 0] -= self.buffer * w
            m[:, 2] += self.buffer * w
            m[:, 1] -= self.buffer * h
            m[:, 3] += self.buffer * h
        dists = 1 - bbox_ioa(a, b, iou=True)
        return fuse_score(dists, detections) if self.args.fuse_score else dists


class PersonTracker:
    """ByteTrack on person detections, plus one repair: re-linking.

    ByteTrack can lose a person who is partly hidden, e.g. behind a pole, and restart them under a
    new ID when they reappear. While the person is half hidden, the detector only boxes the visible
    part. The Kalman filter takes that shrinking box for movement, so its prediction drifts away.
    So every new track is compared with the tracks that disappeared in the last `lost_seconds`.
    If it starts close to where one of them would be now, it gets that track's ID back:
      - "where it would be now": where it was last seen, or that point moved on at the speed it had
        over its last second (up to 1 s ahead), whichever is closer. A walker hidden by a pillar is
        expected further on; someone who stopped, where they were;
      - "close": within (`relink` + `walk_speed` x seconds lost) person-heights;
      - its height within 0.5-2x of the lost track's height (bending and crouching change it).
    When several lost tracks qualify, the closest one wins.
    The IDs this class reports are these repaired IDs.
    """

    def __init__(self, fps: float, settings: TrackerSettings = TrackerSettings()):
        self.settings = settings
        self.fps = fps
        self._frame = 0
        self._public: dict[int, int] = {}                 # ByteTrack's id -> the id we report
        self._last: dict[int, tuple[tuple, float]] = {}   # reported id -> (last box, time it was seen)
        self._trail: dict[int, deque] = {}                # reported id -> (time, box centre) over the last second
        self._next_id = 1
        self._t = None
        args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=settings.high,
                               track_low_thresh=settings.low, new_track_thresh=settings.new_track,
                               track_buffer=max(1, round(settings.lost_seconds * fps)),
                               match_thresh=settings.match, fuse_score=True)
        self._tracker = _ByteTrack(args)
        self._tracker.damping = settings.lost_damping

    def update(self, people: list[Det], frame_size: tuple[int, int], t: float | None = None) -> list[Track]:
        """People detected in this frame (any confidence >= low) -> the tracks seen this frame.
        Track.det_index refers to `people` after duplicates are merged; use Track.box.

        frame_size = (width, height) in pixels. Tracking runs in pixels so the Kalman filter's
        motion model sees true proportions. t = the frame's time in seconds; when frames are further
        apart than 1/fps (skipped frames), boxes are compared with a wider buffer."""
        w, h = frame_size
        now = self._frame / self.fps if t is None else t
        dt = 1.0 / self.fps if self._t is None else max(1.0 / self.fps, now - self._t)
        self._t = now
        # a person moves up to ~walk_speed heights/s; about half the width of their box every 1/15 s
        # at 15 fps needs no buffer, a third of a second needs about half a box width
        self._tracker.buffer = min(1.0, max(0.0, self.settings.buffer_rate * dt - 0.1))
        if self.settings.duplicate < 1:
            people = merge_duplicates(people, self.settings.duplicate)
        if people:
            xyxy = np.array([[d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h] for d in people], dtype=np.float32)
            conf = np.array([d.score for d in people], dtype=np.float32)
        else:
            xyxy, conf = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        out = self._tracker.update(_Boxes(xyxy, conf, np.zeros(len(conf), np.float32)))
        self._frame += 1
        rows = [(int(row[4]), int(row[7])) for row in out]
        seen_now = {self._public[raw] for raw, _ in rows if raw in self._public}
        tracks = []
        for raw, idx in rows:
            box = people[idx].box
            public = self._public.get(raw)
            if public is None or public in {t.track_id for t in tracks}:   # new, or its ID is in use this frame
                public = self._relinked(box, seen_now, now, w, h)
                for other in [r for r, p in self._public.items() if p == public]:
                    del self._public[other]   # an old ByteTrack track with this ID is not this person any more
                self._public[raw] = public
            seen_now.add(public)
            self._last[public] = (box, now)
            trail = self._trail.setdefault(public, deque())
            trail.append((now, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)))
            while trail and trail[0][0] < now - 1.0:
                trail.popleft()
            tracks.append(Track(public, box, float(people[idx].score), idx))
        self._last = {k: v for k, v in self._last.items() if now - v[1] <= self.settings.lost_seconds + 1e-6}
        self._trail = {k: v for k, v in self._trail.items() if k in self._last}
        return tracks

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _relinked(self, box, taken: set[int], now: float, w: float, h: float) -> int:
        """The ID of a recently lost track this new one starts next to, or a fresh ID. Distances are
        measured in pixels, in person-heights."""
        best, best_dist = None, None
        cx, cy, bh = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h, (box[3] - box[1]) * h
        for public, (last, seen) in self._last.items():
            if public in taken or seen >= now:
                continue
            lh = (last[3] - last[1]) * h
            if lh <= 0 or not 0.5 <= bh / lh <= 2.0:   # bending or standing up changes the height
                continue
            lost = now - seen
            lx, ly = (last[0] + last[2]) / 2 * w, (last[1] + last[3]) / 2 * h
            places = [(lx, ly)]
            trail = self._trail.get(public)
            if trail and trail[-1][0] - trail[0][0] >= 0.3:  # its speed over (up to) its last second
                (t0, c0), (t1, c1) = trail[0], trail[-1]
                ahead = min(lost, 1.0) / (t1 - t0)
                places.append((lx + (c1[0] - c0[0]) * w * ahead, ly + (c1[1] - c0[1]) * h * ahead))
            dist = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in places) / max(bh, lh)
            if dist <= self.settings.relink + self.settings.walk_speed * lost:
                if best_dist is None or dist < best_dist:
                    best, best_dist = public, dist
        return best if best is not None else self._new_id()
