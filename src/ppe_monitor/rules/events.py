"""From flickering per-frame verdicts to one trustworthy event per violation (book, Chapters 10 and 20).

For each tracked person and each rule of the camera (rules/engine.py: no helmet, no vest, inside a
restricted zone, ...), every frame adds one vote: violating, fine, or can't tell (e.g. the person
is too small, or their feet are out of view). Then:

  1. Smoothing. Over the last `window` seconds, if at least `min_votes` votes are violating/fine,
     spread over at least half the window, and more than `ratio` of them say violating, the person
     counts as "violating" right now. "Can't tell" votes are ignored, not counted as either answer.
  2. Dwell. A violation only becomes an event after it has lasted the rule's `dwell` seconds. Short breaks
     (up to `gap` seconds without a violating verdict, e.g. the person turns away) don't restart
     the clock; a longer break does.
  3. One event per episode. After an event the person is "in violation" until they have been
     compliant for `clear` seconds. No further events are raised during that time, however long it lasts.
  4. Cooldown. The same person can't trigger the same rule again within its `cooldown`
     seconds of the last one. If a new episode starts inside the cooldown and is still going when
     the cooldown ends, it is raised then.
  5. Same place, same incident. Trackers sometimes hand a person a new ID (Chapter 9), for example
     after walking behind a pole, and the detector sometimes finds one person twice. Each event
     therefore remembers where its incident is: the owning track's box, followed from frame to
     frame as long as it moves plausibly for a person (walking pace, even with frames skipped). If
     the track suddenly jumps (its ID was handed to a neighbour), the incident stays where it was.
     If the owner goes out of sight and a new track (one that appeared at most `handover` seconds
     before) is in its place, that track takes the incident over and the incident follows it: the
     same person under a new ID, still walking. Before a new event is raised, ongoing incidents of
     the same rule are checked. One overlapping this person's box is the same incident: the new
     track joins it and no new alert is sent. An incident can be joined until `forget` seconds after
     its last violating frame.
  6. Active hours. A rule that is switched off (outside its hours) counts nothing, and whoever was
     being confirmed for it starts again from zero when it comes back on.

`dwell` and `cooldown` below are the defaults for rules that don't set their own.

All times are in seconds (a video's own clock, or wall-clock for live cameras), so behaviour does
not depend on the frame rate or on frames skipped to stay live.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from datetime import datetime

from .engine import DEFAULT_RULES, PPE_ITEM, Rule
from .ppe import MISSING, UNKNOWN, Verdict

KINDS = {"helmet": "no_helmet", "vest": "no_vest"}
OK, PENDING, VIOLATION = "ok", "pending", "violation"


@dataclass(frozen=True)
class EventSettings:
    window: float = 1.0      # seconds of votes in the rolling majority
    min_votes: int = 3       # worn/missing votes needed in the window before judging at all
    ratio: float = 0.6       # more than this share of missing votes = violating
    dwell: float = 3.0       # a violation must last this long before it becomes an event
    gap: float = 1.0         # breaks shorter than this don't restart the dwell clock
    clear: float = 2.0       # compliant this long = the episode is over
    cooldown: float = 60.0   # no new event of the same kind for the same person within this time
    same_place_iou: float = 0.3  # a person overlapping an ongoing incident this much = the same incident
    same_place_inside: float = 0.7  # ... or with one box this much inside the other
    forget: float = 10.0     # drop a lost track's state, and let incidents go, after this many seconds
    buffer_rate: float = 1.8  # following a moving person: boxes widened by (buffer_rate x seconds - 0.1) of their size
    handover: float = 1.0    # a track that appeared up to this long before an incident's owner was lost can take it over

    @classmethod
    def from_dict(cls, d: dict | None) -> "EventSettings":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown event settings: {sorted(unknown)}")
        return cls(**{k: (int(v) if k == "min_votes" else float(v)) for k, v in d.items()})


@dataclass
class Event:
    """A confirmed violation. PPE and zone events have exactly this shape, which later phases
    store, send and show."""
    kind: str                 # the rule's type: "no_helmet", "no_vest" or "zone_intrusion"
    camera: str
    track_id: int
    started: float            # when the violation began (start of the dwell), seconds on the stream's clock
    confirmed: float          # when it became an event
    box: tuple                # person box at confirmation, 0-1 fractions of the frame
    share: float              # share of violating votes in the window at confirmation
    frame_index: int          # the frame the event was confirmed on (the snapshot's frame)
    event_id: str = ""
    rule: str = ""            # id of the rule that raised it (configs/rules.yaml)
    zone: str | None = None   # zone id, for zone rules (and PPE rules limited to a zone)
    severity: str = ""        # low / medium / high / critical
    time: str = ""            # wall-clock time of confirmation, ISO 8601 with time zone
    snapshot: str = ""        # evidence image, filled in by whoever saves it

    def to_dict(self) -> dict:
        d = asdict(self)
        d["started"], d["confirmed"] = round(self.started, 3), round(self.confirmed, 3)
        d["box"] = [round(float(v), 4) for v in self.box]
        return d


@dataclass
class _ItemState:
    votes: deque = field(default_factory=deque)   # (time, answer)
    phase: str = OK
    since: float | None = None      # start of the current pending/violation period
    last_bad: float | None = None   # last time the smoothed verdict was "violating"
    last_event: float | None = None
    event: Event | None = None
    alerted: bool = False           # has the current episode produced (or joined) an event?


@dataclass
class _TrackState:
    items: dict = field(default_factory=dict)    # rule id -> _ItemState
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    last_seen: float = 0.0
    first_seen: float = 0.0


@dataclass
class _Incident:
    event: Event
    item: str                 # rule id
    owner: int | None         # the track currently responsible for it (None: its ID moved to someone else)
    box: tuple                # where it is: the owner's box, while the owner moves plausibly
    last_violating: float     # last time the owner was judged violating
    seen: float = 0.0         # when `box` was last updated


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _widen(box, b: float) -> tuple:
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - b * w, box[1] - b * h, box[2] + b * w, box[3] + b * h)


def _inside(a, b) -> float:
    """Share of the smaller box that lies inside the other: 1.0 when one contains the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


class EventEngine:
    def __init__(self, camera: str, settings: EventSettings = EventSettings(), rules: list[Rule] | None = None):
        """rules: this camera's rules (rules/engine.py); default: no helmet and no vest, anywhere."""
        self.camera = camera
        self.s = settings
        self.rules = {r.id: r for r in (DEFAULT_RULES if rules is None else rules)}
        self.tracks: dict[int, _TrackState] = {}
        self.events: list[Event] = []
        self.incidents: list[_Incident] = []

    def _dwell(self, rule: Rule) -> float:
        return self.s.dwell if rule.dwell is None else rule.dwell

    def _cooldown(self, rule: Rule) -> float:
        return self.s.cooldown if rule.cooldown is None else rule.cooldown

    def _votes_from(self, x) -> dict:
        """A Verdict (Phase 2 callers and tests) -> votes for this engine's PPE rules."""
        if not isinstance(x, Verdict):
            return x
        out = {}
        for r in self.rules.values():
            if r.type in PPE_ITEM and r.zone is None:
                a = getattr(x, PPE_ITEM[r.type])
                out[r.id] = None if a == UNKNOWN else a == MISSING
        return out

    # -- smoothing ---------------------------------------------------------------------------
    def _window_votes(self, st: _ItemState, t: float) -> list:
        """Known (worn / missing) votes of the last `window` seconds. At a low frame rate (frames
        skipped to keep up), the window stretches back, up to 3x, to still hold `min_votes`."""
        while st.votes and st.votes[0][0] < t - 3 * self.s.window:
            st.votes.popleft()
        known = [(ts, a) for ts, a in st.votes if a is not None]
        recent = [(ts, a) for ts, a in known if ts >= t - self.s.window]
        if len(recent) < self.s.min_votes:
            recent = known[-self.s.min_votes:]
        return recent

    def _smoothed(self, st: _ItemState, t: float) -> str:
        """'violating', 'compliant' or 'unsure' from the votes in the window."""
        known = self._window_votes(st, t)
        # judge only on enough votes spread over at least half the window, so a new track's
        # first few frames (or a burst of misses right after an unknown spell) can't decide alone
        if len(known) < self.s.min_votes or t - known[0][0] < self.s.window / 2:
            return "unsure"
        return "violating" if sum(a is True for _, a in known) / len(known) > self.s.ratio else "compliant"

    def _violating_share(self, st: _ItemState, t: float) -> float:
        known = [a for _, a in self._window_votes(st, t)]
        return sum(a is True for a in known) / len(known) if known else 0.0

    # -- incidents: one per real violation, whatever the track IDs do -----------------------------
    def _incident_at(self, item: str, box) -> _Incident | None:
        """An ongoing incident of this kind at this person's place: overlapping boxes, or one box
        mostly inside the other (a person seen only in part, then whole, e.g. behind a pole)."""
        best, best_score = None, 0.0
        for inc in self.incidents:
            if inc.item != item:
                continue
            o = _iou(inc.box, box)
            if o >= self.s.same_place_iou or _inside(inc.box, box) >= self.s.same_place_inside:
                if o + _inside(inc.box, box) > best_score:
                    best, best_score = inc, o + _inside(inc.box, box)
        return best

    def _plausible_move(self, old, new, dt: float) -> bool:
        """Could a person have moved from box `old` to box `new` in `dt` seconds? Yes if the boxes
        overlap by `same_place_iou` once both are widened in proportion to the time between them
        (the tracker's "buffered IoU"): a walker's boxes barely overlap when frames are skipped, but
        a jump onto a neighbour a box-width away never passes between consecutive frames. A jump
        means the track ID now belongs to someone else."""
        b = min(1.0, max(0.0, self.s.buffer_rate * max(0.0, dt) - 0.1))
        return _iou(_widen(old, b), _widen(new, b)) >= self.s.same_place_iou

    def _raise(self, item: str, st: _ItemState, track_id: int, box, t: float, frame_index: int,
               now: datetime | None) -> Event:
        rule = self.rules[item]
        ev = Event(rule.type, self.camera, track_id, st.since, t, tuple(box), round(self._violating_share(st, t), 3),
                   frame_index, rule=rule.id, zone=rule.zone, severity=rule.severity,
                   time=now.isoformat(timespec="seconds") if now is not None else "")
        ev.event_id = f"{self.camera}-{rule.id}-{track_id}-{int(t * 1000)}"
        st.event, st.last_event, st.alerted = ev, t, True
        self.events.append(ev)
        self.incidents.append(_Incident(ev, item, track_id, tuple(box), t, t))
        return ev

    # -- main step ------------------------------------------------------------------------------
    def update(self, t: float, frame_index: int, people: list, active: set[str] | None = None,
               now: datetime | None = None) -> list[Event]:
        """One frame. people: (track_id, box, votes) for each tracked person, where votes is
        {rule id: True violating / False fine / None can't tell} (or a PPE Verdict). active: the
        rules switched on right now (None = all). now: the wall-clock time, stamped on events.
        Returns the new events."""
        people = [(tid, box, self._votes_from(v)) for tid, box, v in people]
        for track_id, box, _ in people:
            if track_id not in self.tracks:
                self.tracks[track_id] = _TrackState(first_seen=t)
            ts = self.tracks[track_id]
            ts.box, ts.last_seen = tuple(box), t
        if active is not None:          # rules switched off: everyone starts again from zero
            for ts in self.tracks.values():
                for rid, st in ts.items.items():
                    if rid not in active and (st.phase != OK or st.votes):
                        st.votes.clear()
                        st.phase, st.since, st.alerted = OK, None, False
        boxes = {track_id: tuple(box) for track_id, box, _ in people}
        for inc in self.incidents:  # follow each incident's owner, unless the owner's box jumped
            b = boxes.get(inc.owner)
            if b is None:
                continue
            if self._plausible_move(inc.box, b, t - inc.seen):
                inc.box, inc.seen = b, t
            else:
                inc.owner = None  # that ID is now on someone else; the incident stays where it was
        for inc in self.incidents:  # an owner out of sight: a new track in its place takes the incident over
            if inc.owner in boxes:
                continue
            for track_id, b in boxes.items():
                if self.tracks[track_id].first_seen < inc.seen - self.s.handover:
                    continue      # there before the owner was lost: someone else
                if any(o.owner == track_id and o.item == inc.item for o in self.incidents):
                    continue
                if _iou(inc.box, b) >= self.s.same_place_iou or _inside(inc.box, b) >= self.s.same_place_inside:
                    inc.owner, inc.box, inc.seen = track_id, b, t
                    break
        new = []
        for track_id, box, votes in people:
            ts = self.tracks[track_id]
            for item, vote in votes.items():
                if item not in self.rules or (active is not None and item not in active):
                    continue
                rule = self.rules[item]
                st = ts.items.setdefault(item, _ItemState())
                st.votes.append((t, vote))
                smooth = self._smoothed(st, t)
                if smooth == "violating":
                    st.last_bad = t

                if st.phase == OK:
                    if smooth == "violating":
                        st.phase, st.since = PENDING, t
                elif st.phase == PENDING:
                    if smooth == "compliant" or t - st.last_bad > self.s.gap:
                        st.phase, st.since = OK, None
                    elif smooth == "violating" and t - st.since >= self._dwell(rule):
                        st.phase, st.alerted = VIOLATION, False
                        inc = self._incident_at(item, box)
                        if inc is not None:  # an incident already alerted here: join it, no new alert
                            st.event, st.last_event, st.alerted = inc.event, inc.event.confirmed, True
                            inc.owner, inc.box, inc.last_violating, inc.seen = track_id, tuple(box), t, t
                        elif st.last_event is None or t - st.last_event >= self._cooldown(rule):
                            new.append(self._raise(item, st, track_id, box, t, frame_index, now))
                elif st.phase == VIOLATION:
                    if smooth == "violating":
                        for inc in self.incidents:
                            if inc.owner == track_id and inc.item == item:
                                inc.last_violating = t
                    if smooth == "compliant" and t - st.last_bad >= self.s.clear:
                        st.phase, st.since = OK, None
                    elif not st.alerted and smooth == "violating" and t - st.last_event >= self._cooldown(rule):
                        # this episode began during the cooldown and is still going: alert now,
                        # unless another track's incident is already here
                        inc = self._incident_at(item, box)
                        if inc is not None:
                            st.event, st.alerted = inc.event, True
                            inc.owner, inc.box, inc.last_violating, inc.seen = track_id, tuple(box), t, t
                        else:
                            new.append(self._raise(item, st, track_id, box, t, frame_index, now))

        for track_id in [k for k, v in self.tracks.items() if t - v.last_seen > self.s.forget]:
            del self.tracks[track_id]
        self.incidents = [i for i in self.incidents if t - i.last_violating <= self.s.forget]
        return new

    def status(self, track_id: int) -> dict[str, str]:
        """Current phase per rule, for drawing: ok / pending / violation."""
        ts = self.tracks.get(track_id)
        return {rid: (ts.items[rid].phase if ts and rid in ts.items else OK) for rid in self.rules}
