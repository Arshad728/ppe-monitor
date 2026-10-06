"""Ground truth: the violations really in each test clip (book, Chapter 23).

A clip's truth is a list of violation *episodes*. Each has a kind, a start and an end (seconds from
the clip's first frame) and, when known, where the person is: a box at a few moments (0-1 fractions
of the frame), followed in a straight line between them. An episode is either

  - **sustained**: in view long enough that the system should alert. No alert = a missed violation.
  - **not sustained** ("don't care"): real, but too short or too hidden to expect an alert, such as a
    man without a helmet who is in view for 2 s when an alert needs 3. An alert on it is neither
    right nor wrong, so it counts in neither precision nor recall.

A third sort marks a person known to be **compliant** (wearing the item, or never in the zone): an
alert on them is a false alarm.

When a clip's truth is **complete** (hand-annotated: every person near enough to judge was checked),
an alert that matches no episode is a false alarm too. When it isn't (the generated clips: the
source photos label only some of the people in them), such an alert is about someone nobody
labelled and is counted apart, as "not scored". Either way this holds only for the kinds the truth
covers (`checked`); an alert of another kind can't be judged.

Two sources, one format:
  - hand-annotated YAML files, data/ground_truth/<clip>.yaml, for real footage
    (scripts/annotate_clip.py helps to write them);
  - the known answers of the generated test clips, datasets/event_clips*/clips.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT
from ..rules.scoring import MIN_VISIBLE, _in_view, zone_truth

TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
KINDS = ("no_helmet", "no_vest", "zone_intrusion")
ITEM = {"no_helmet": "helmet", "no_vest": "vest"}


@dataclass
class Episode:
    kind: str
    start: float
    end: float
    sustained: bool = True
    boxes: list = field(default_factory=list)       # [(t, (x1, y1, x2, y2)), ...] sorted by t
    note: str = ""
    compliant: bool = False                         # a person known NOT to violate: an alert on them is false

    def box_at(self, t: float):
        """The person's box at time t: between two known boxes, in a straight line; before the first
        or after the last, the nearest one. None when no box is known."""
        if not self.boxes:
            return None
        if t <= self.boxes[0][0]:
            return self.boxes[0][1]
        for (t0, b0), (t1, b1) in zip(self.boxes, self.boxes[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return tuple(v0 + a * (v1 - v0) for v0, v1 in zip(b0, b1))
        return self.boxes[-1][1]

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "start": round(self.start, 3), "end": round(self.end, 3), "sustained": self.sustained}
        if self.compliant:
            d = {"kind": self.kind, "start": round(self.start, 3), "end": round(self.end, 3), "compliant": True}
        if self.boxes:
            d["boxes"] = {round(t, 3): [round(v, 4) for v in b] for t, b in self.boxes}
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class ClipTruth:
    clip: str                        # the video's file name
    path: Path                       # the video itself
    fps: float
    duration: float                  # seconds
    episodes: list[Episode]
    checked: tuple[str, ...]         # the kinds whose truth is complete in this clip
    source: str = ""                 # who or what made this truth, and how
    notes: str = ""
    zone: dict | None = None         # {id, name, polygon}: the clip's restricted zone, if it has one
    private: bool = False            # real people: its pictures must not leave this machine
    complete: bool = True            # every person near enough to judge was checked (see above)

    @property
    def sustained(self) -> list[Episode]:
        return [e for e in self.episodes if e.sustained and not e.compliant]


# -- hand-annotated YAML ------------------------------------------------------------------------------
def _parse_boxes(raw, where: str) -> list:
    out = []
    for t, b in (raw or {}).items():
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            raise ValueError(f"{where}: box at {t} s must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(v) for v in b)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"{where}: box at {t} s must be fractions of the frame, x1 < x2 and y1 < y2")
        out.append((float(t), (x1, y1, x2, y2)))
    return sorted(out)


def load_yaml(path: str | Path, clips_dir: Path | None = None) -> ClipTruth:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    where = path.name
    for key in ("clip", "fps", "duration"):
        if key not in raw:
            raise ValueError(f"{where}: '{key}' is missing")
    checked = tuple(raw.get("checked") or ())
    bad = [k for k in checked if k not in KINDS]
    if bad or not checked:
        raise ValueError(f"{where}: 'checked' must list one or more of {', '.join(KINDS)}")
    episodes = []
    for i, e in enumerate(raw.get("episodes") or []):
        w = f"{where}, episode {i + 1}"
        if e.get("kind") not in KINDS:
            raise ValueError(f"{w}: kind must be one of {', '.join(KINDS)}")
        if e["kind"] not in checked:
            raise ValueError(f"{w}: kind {e['kind']} is not in 'checked'")
        start, end = float(e["start"]), float(e["end"])
        if not 0 <= start < end:
            raise ValueError(f"{w}: needs 0 <= start < end")
        compliant = bool(e.get("compliant", False))
        episodes.append(Episode(e["kind"], start, end, bool(e.get("sustained", True)) and not compliant,
                                _parse_boxes(e.get("boxes"), w), str(e.get("note", "")).strip(), compliant))
    clips_dir = clips_dir or PROJECT_ROOT / "data" / "clips"
    return ClipTruth(raw["clip"], clips_dir / raw["clip"], float(raw["fps"]), float(raw["duration"]), episodes,
                     checked, str(raw.get("source", "hand-annotated")).strip(), str(raw.get("notes", "")).strip(),
                     raw.get("zone"), bool(raw.get("private", False)), bool(raw.get("complete", True)))


def save_yaml(truth: ClipTruth, path: str | Path, header: str = "") -> None:
    d = {"clip": truth.clip, "fps": truth.fps, "duration": round(truth.duration, 3), "private": truth.private,
         "checked": list(truth.checked), "complete": truth.complete, "source": truth.source}
    if truth.notes:
        d["notes"] = truth.notes
    if truth.zone:
        d["zone"] = truth.zone
    d["episodes"] = [e.to_dict() for e in truth.episodes]
    text = yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100, default_flow_style=None)
    Path(path).write_text((header.rstrip() + "\n" if header else "") + text, encoding="utf-8")


def load_dir(folder: str | Path = TRUTH_DIR, clips_dir: Path | None = None) -> list[ClipTruth]:
    return [load_yaml(p, clips_dir) for p in sorted(Path(folder).glob("*.yaml"))]


# -- the generated clips (known answers) ----------------------------------------------------------------
def _track(boxes: list, fps: float, lo: int, hi: int, every: int = 3) -> list:
    out = []
    for k in list(range(lo, hi + 1, every)) + [hi]:
        b = boxes[k] if k < len(boxes) else None
        if b is not None and (not out or out[-1][0] < k / fps):
            out.append((k / fps, tuple(float(v) for v in b[:4])))
    return out


def from_photo_clips(clips_dir: str | Path, kinds: tuple[str, ...] = ("no_helmet", "no_vest"),
                     zone_dwell: float = 2.0, zone_margin: float = 1.5) -> list[ClipTruth]:
    """The truth of the clips made by scripts/make_event_clips.py, in the format above.

    PPE clips: a person known to be without the item and in view for at least 70 % of the clip is a
    sustained episode, from the first frame they are in view to the last. Without it but in view
    less = not sustained. Nobody knows (the item wasn't labelled) = not sustained, so an alert on
    them can't count either way; a person wearing it is known compliant. Zone clips (with a zone): a
    person whose feet enter the zone and stay, for at least the rule's dwell plus `zone_margin` s, is
    sustained from the moment they enter; one who enters later, or leaves again, is not; one whose
    feet are in view and never inside is compliant. The truth is not complete: the photos' labels
    miss some people, and an alert on one of them can't be judged."""
    import json

    clips_dir = Path(clips_dir)
    meta = json.loads((clips_dir / "clips.json").read_text(encoding="utf-8"))
    out = []
    for c in meta["clips"]:
        fps, n = float(c["fps"]), int(c["frames"])
        eps: list[Episode] = []
        zone = c.get("zone")
        if zone:
            checked = ("zone_intrusion",)
            for p, person in enumerate(c["people"]):
                inside = zone_truth(person, zone["polygon"], n)
                seen = [v for v in inside if v is not None]
                if not any(seen):
                    if len(seen) >= MIN_VISIBLE * n:            # feet in view, never inside: known compliant
                        eps.append(Episode("zone_intrusion", 0.0, n / fps, False, _track(person["boxes"], fps, 0, n - 1),
                                           f"person {p}: never in the zone", compliant=True))
                    continue
                entry = next(k for k, v in enumerate(inside) if v)
                last = max(k for k, v in enumerate(inside) if v)
                stays = all(v is not False for v in inside[entry:])
                ok = len(seen) >= MIN_VISIBLE * n and stays and (n - entry) / fps >= zone_dwell + zone_margin
                note = "" if ok else ("feet mostly out of view" if len(seen) < MIN_VISIBLE * n else
                                      "leaves the zone again" if not stays else "enters too late to be confirmed")
                eps.append(Episode("zone_intrusion", entry / fps, (last + 1) / fps, ok,
                                   _track(person["boxes"], fps, entry, last), note or f"person {p}"))
        else:
            checked = tuple(kinds)
            for p, person in enumerate(c["people"]):
                in_view = [k for k, b in enumerate(person["boxes"]) if _in_view(b)]
                for kind in kinds:
                    has = person[ITEM[kind]]
                    whole = _track(person["boxes"], fps, 0, n - 1)
                    if has is True:                     # wearing it (even if cut off by the frame's edge)
                        eps.append(Episode(kind, 0.0, n / fps, False, whole, f"person {p}: wearing it", compliant=True))
                    elif has is None:
                        eps.append(Episode(kind, 0.0, n / fps, False, whole, f"person {p}: not labelled, can't be judged"))
                    elif len(in_view) >= MIN_VISIBLE * n:
                        eps.append(Episode(kind, in_view[0] / fps, (in_view[-1] + 1) / fps, True,
                                           _track(person["boxes"], fps, in_view[0], in_view[-1]), f"person {p}"))
                    else:
                        eps.append(Episode(kind, 0.0, n / fps, False, whole,
                                           f"person {p}: in view for under 70% of the clip"))
        out.append(ClipTruth(c["clip"], clips_dir / c["clip"], fps, n / fps, eps, checked,
                             f"generated with the clip (scripts/make_event_clips.py) from the labels of "
                             f"{c.get('image', 'a test photo')}", zone=zone, complete=False))
    return out
