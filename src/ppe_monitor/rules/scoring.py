"""Scoring events and tracks against clips with known answers (the Phase 2 "done when" check).

Done when (book, Chapter 19): every real, sustained violation produces exactly one confirmed event,
and momentary misdetections produce none. For each clip:

  - each person known to be without a helmet (or vest), and visible long enough to be judged,
    should get exactly ONE event of that kind: caught once / caught with duplicates / missed;
  - each person known to be wearing it should get NO event of that kind: any is a false event;
  - events on people whose status isn't known (unlabelled people, or an item nobody labelled) are
    counted separately: they can't be scored either way;
  - tracking: each person should keep one track ID for the whole clip (ID switches, and the share
    of frames where their track was found).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .events import Event, KINDS
from .zones import point_in_polygon

ITEM_OF = {v: k for k, v in KINDS.items()}
MIN_VISIBLE = 0.7  # a person must be in view (not cut off) for this share of the clip to be scored


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _in_view(box) -> bool:
    return box is not None and box[1] > 0.005 and box[3] < 0.995 and box[0] > 0.0 and box[2] < 1.0


def _event_person(ev: Event, people: list, tracks_per_frame: dict, fps: float) -> int | None:
    """The person an event is about: the one its track covered during the violation that raised it
    (from the start of the dwell to the event), or failing that the one under the event's box. The
    box alone can mislead: at the moment of the event the track may sit on a half-hidden or wrong
    detection for a frame or two."""
    lo, hi = int(ev.started * fps), ev.frame_index
    votes = Counter()
    for k in range(lo, hi + 1):
        tb = tracks_per_frame.get(k, {}).get(ev.track_id)
        if tb is None:
            continue
        for p, person in enumerate(people):
            b = person["boxes"][k] if k < len(person["boxes"]) else None
            if b is not None and iou(tb, b[:4]) >= 0.5:
                votes[p] += 1
    if votes:
        return votes.most_common(1)[0][0]
    best, best_iou = None, 0.3
    k = ev.frame_index
    for p, person in enumerate(people):
        b = person["boxes"][k] if k < len(person["boxes"]) else None
        if b is not None and iou(ev.box, b[:4]) >= best_iou:
            best, best_iou = p, iou(ev.box, b[:4])
    return best


@dataclass
class ClipScore:
    caught_once: Counter = field(default_factory=Counter)       # per item
    caught_more: Counter = field(default_factory=Counter)
    missed: Counter = field(default_factory=Counter)
    false_events: Counter = field(default_factory=Counter)
    compliant_people: Counter = field(default_factory=Counter)
    unscored_events: int = 0
    delays: list = field(default_factory=list)
    id_switches: int = 0
    people_tracked: int = 0
    track_coverage: list = field(default_factory=list)
    details: list = field(default_factory=list)                 # human-readable notes per outcome

    def add(self, other: "ClipScore") -> None:
        for name in ("caught_once", "caught_more", "missed", "false_events", "compliant_people"):
            getattr(self, name).update(getattr(other, name))
        self.unscored_events += other.unscored_events
        self.delays += other.delays
        self.id_switches += other.id_switches
        self.people_tracked += other.people_tracked
        self.track_coverage += other.track_coverage
        self.details += other.details


def score_clip(truth: dict, events: list[Event], tracks_per_frame) -> ClipScore:
    """truth: from sim/photo_clips.make_clip. tracks_per_frame: {track_id: box} for every frame (a
    list), or {frame index: {track_id: box}} for the frames that were processed (a dict)."""
    sc = ClipScore()
    tracks_per_frame = tracks_per_frame if isinstance(tracks_per_frame, dict) else dict(enumerate(tracks_per_frame))
    people = truth["people"]
    n = truth["frames"]

    # which person each event belongs to
    per_person = [Counter() for _ in people]
    first = [dict() for _ in people]
    fps = truth["fps"]
    for ev in events:
        if ev.kind not in ITEM_OF:          # zone events are scored by score_zone
            continue
        best = _event_person(ev, people, tracks_per_frame, fps)
        item = ITEM_OF[ev.kind]
        if best is None or people[best][item] is None:
            sc.unscored_events += 1
            continue
        per_person[best][item] += 1
        first[best].setdefault(item, ev.confirmed)

    for p, person in enumerate(people):
        visible = sum(_in_view(b) for b in person["boxes"]) / n
        for item in ("helmet", "vest"):
            truth_item = person[item]
            if truth_item is None:
                continue
            got = per_person[p][item]
            if truth_item:                                   # wearing it: any event is false
                sc.compliant_people[item] += 1
                if got:
                    sc.false_events[item] += got
                    sc.details.append(f"{truth['clip']}: false no_{item} event on person {p} (wearing it)")
            elif visible >= MIN_VISIBLE:                     # really without it, and in view
                if got == 1:
                    sc.caught_once[item] += 1
                    sc.delays.append(first[p][item])
                elif got > 1:
                    sc.caught_more[item] += 1
                    sc.delays.append(first[p][item])
                    sc.details.append(f"{truth['clip']}: {got} no_{item} events on person {p} (should be 1)")
                else:
                    sc.missed[item] += 1
                    sc.details.append(f"{truth['clip']}: missed no_{item} on person {p}")

        # tracking: the track that covers this person in each frame
        ids, seen, in_view = [], 0, 0
        for k in range(n):
            b = person["boxes"][k]
            if b is None or b[4] or k not in tracks_per_frame:   # out of view, behind the occluder, or skipped
                continue
            in_view += 1
            best, best_iou = None, 0.5
            for tid, tb in tracks_per_frame[k].items():
                o = iou(tb, b[:4])
                if o >= best_iou:
                    best, best_iou = tid, o
            if best is not None:
                seen += 1
                if not ids or ids[-1] != best:
                    ids.append(best)
        if in_view:
            sc.people_tracked += 1
            sc.track_coverage.append(seen / in_view)
            sc.id_switches += max(0, len(ids) - 1)
    return sc


# -- restricted zones (Phase 3) -----------------------------------------------------------------
@dataclass
class ZoneScore:
    """Zone intrusion events against where each person's feet really are (their labelled box).
    should_alert: walked into the zone and stayed long enough to be confirmed (dwell + `margin`);
    never_inside: feet always in view and never inside the zone."""
    should_alert: int = 0
    caught_once: int = 0
    caught_more: int = 0
    missed: int = 0
    never_inside: int = 0
    false_events: int = 0
    unscored_events: int = 0
    unscored_people: int = 0
    delays: list = field(default_factory=list)      # event time - time the feet entered the zone
    details: list = field(default_factory=list)

    def add(self, o: "ZoneScore") -> None:
        for name in ("should_alert", "caught_once", "caught_more", "missed", "never_inside", "false_events",
                     "unscored_events", "unscored_people"):
            setattr(self, name, getattr(self, name) + getattr(o, name))
        self.delays += o.delays
        self.details += o.details


def zone_truth(person: dict, polygon, n: int) -> list:
    """Per frame: True (feet inside), False (outside) or None (feet out of view)."""
    out = []
    for k in range(n):
        b = person["boxes"][k] if k < len(person["boxes"]) else None
        if b is None or b[3] >= 0.995:
            out.append(None)
        else:
            out.append(point_in_polygon(((b[0] + b[2]) / 2, b[3]), polygon))
    return out


def score_zone(truth: dict, events: list[Event], tracks_per_frame, dwell: float, margin: float = 1.5) -> ZoneScore:
    sc = ZoneScore()
    if "zone" not in truth:
        return sc
    tracks_per_frame = tracks_per_frame if isinstance(tracks_per_frame, dict) else dict(enumerate(tracks_per_frame))
    fps, n, people = truth["fps"], truth["frames"], truth["people"]
    got = Counter()
    first = {}
    for ev in events:
        if ev.kind != "zone_intrusion":
            continue
        p = _event_person(ev, people, tracks_per_frame, fps)
        if p is None:
            sc.unscored_events += 1
            continue
        got[p] += 1
        first.setdefault(p, ev.confirmed)
    for p, person in enumerate(people):
        inside = zone_truth(person, truth["zone"]["polygon"], n)
        seen = [v for v in inside if v is not None]
        if len(seen) < MIN_VISIBLE * n:                        # feet mostly out of view: can't say
            sc.unscored_people += 1
            continue
        if not any(seen):
            sc.never_inside += 1
            if got[p]:
                sc.false_events += got[p]
                sc.details.append(f"{truth['clip']}: zone event on person {p}, who never entered")
            continue
        entry = next(k for k, v in enumerate(inside) if v)
        stays = all(v is not False for v in inside[entry:])
        if stays and (n - entry) / fps >= dwell + margin:
            sc.should_alert += 1
            if got[p] == 1:
                sc.caught_once += 1
                sc.delays.append(first[p] - entry / fps)
            elif got[p] > 1:
                sc.caught_more += 1
                sc.delays.append(first[p] - entry / fps)
                sc.details.append(f"{truth['clip']}: {got[p]} zone events on person {p} (should be 1)")
            else:
                sc.missed += 1
                sc.details.append(f"{truth['clip']}: missed zone intrusion by person {p} (inside from {entry / fps:.1f} s)")
        else:
            sc.unscored_people += 1                            # entered too late to be confirmed, or left again
    return sc
