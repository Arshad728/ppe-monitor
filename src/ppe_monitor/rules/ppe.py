"""Is this person wearing a helmet and a vest in this frame? (book, Chapter 10)

The detector finds people, helmets and vests as separate boxes. A helmet counts as worn by a
person when its centre falls in that person's head region; a vest counts as worn when its centre
falls in the torso region and most of the vest lies inside the person's box. Regions are
fractions of the person box, so they work at any distance:

        x1          x2
    y1  +-----------+   head region: centre of the helmet between head_top and head_bottom
        |   (o)     |   (fractions of the person's height from the top of the box)
        |  /|#|\\    |   torso region: centre of the vest between torso_top and torso_bottom,
        |   |#|     |   and at least `vest_inside` of the vest's area inside the person box
        |   / \\     |
    y2  +-----------+

Each item is given to at most one person and each person gets at most one of each item. The
closest pairs are matched first, so in a crowd a helmet goes to the head it sits on, not to the
taller neighbour whose head region it also touches.

Bending, crouching, lying. The regions above assume an upright person. When body keypoints are
available (vision/pose.py), a helmet also counts as worn if its centre is within `head_pose_radius`
neck-lengths of the head keypoints, and not more than `head_pose_below` neck-lengths below the head
along the body's axis. That finds the helmet of a worker bent over or lying down, whose head is not
at the top of the box. On the labelled training and validation images it raised the share of worn
helmets found on people bending over from 92% to 98% (51 people), and on people lying down from 0% to
55% (11). Crouching (98%, 49 people) and upright people (99.7%) were unchanged. Without keypoints
the box rule is used alone.

While someone is bent over, crouching or lying, a high camera may see only their back, with the
head hidden behind the body. So with `bent_head_hidden_is_unknown`, no helmet found on a bent person
whose head can't be seen (no confident nose / eye / ear keypoint) means "unknown", not "missing":
they are judged once their head is back in view. When the head can be seen, a missing helmet is
missing whatever the posture, so a worker without a helmet who stays bent over (tying rebar, say)
is still caught. On the labelled images, all 17 bare-headed people who were bending or crouching
had their head in view.

Every judgement is one of three answers:
  worn     a matching helmet / vest was found
  missing  none was found, and the person is clearly visible enough to say so
  unknown  can't judge: the person is too small in the image for the detector to find a helmet
           reliably, or the box touches the frame edge so the head or torso may be cut off.
           Unknown never counts as a violation; the event logic just waits for a better view.

Boxes are (x1, y1, x2, y2) as 0-1 fractions of the frame; sizes are judged in pixels of the image
the detector saw, because that is what limits whether a small helmet can be found.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

WORN, MISSING, UNKNOWN = "worn", "missing", "unknown"
ITEMS = ("helmet", "vest")
BENT = ("bending", "crouching", "lying")


@dataclass(frozen=True)
class Rules:
    head_top: float = -0.10       # a helmet centre may sit a little above the person box ...
    head_bottom: float = 0.35     # ... down to 35% of the person's height
    head_side: float = 0.05       # and up to 5% of the width outside either side
    torso_top: float = 0.15
    torso_bottom: float = 0.85
    torso_side: float = 0.0       # vest centre inside the box horizontally
    vest_inside: float = 0.5      # share of the vest's area inside the person box
    min_height_helmet: float = 0  # person height (pixels at the model's input) below which helmet = unknown
    min_height_vest: float = 0    # ... and vest = unknown
    edge: float = 0.005           # a box closer than this to the frame edge counts as cut off
    head_pose_radius: float = 1.25  # with keypoints: helmet centre within this many neck-lengths of the head
    head_pose_below: float = -1.5   # ... and not further below the head than this, along the body axis
    bent_head_hidden_is_unknown: bool = True  # bent over / crouching / lying, head not in view, no helmet: unknown

    _BOOL = ("bent_head_hidden_is_unknown",)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rules":
        d = d or {}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown PPE rule settings: {sorted(unknown)}; allowed: {sorted(known)}")
        return cls(**{k: (bool(v) if k in cls._BOOL else float(v)) for k, v in d.items()})


@dataclass(frozen=True)
class Verdict:
    """One person's PPE in one frame."""
    helmet: str
    vest: str
    helmet_index: int | None = None  # which helmet / vest detection was matched (index into the input list)
    vest_index: int | None = None
    posture: str | None = None       # upright / bending / crouching / lying, when keypoints were available


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _head_cost(p, h, r: Rules, pose=None) -> float | None:
    """How far a helmet sits from where the head is, in person heights; None if it can't be this
    person's helmet. Uses the box's head region and, when available, the head keypoints."""
    pw, ph = p[2] - p[0], p[3] - p[1]
    if pw <= 0 or ph <= 0:
        return None
    costs = []
    u = ((h[0] + h[2]) / 2 - p[0]) / pw
    v = ((h[1] + h[3]) / 2 - p[1]) / ph
    if -r.head_side <= u <= 1 + r.head_side and r.head_top <= v <= r.head_bottom:
        costs.append(abs(u - 0.5) * pw / ph + abs(v - 0.1))  # distance to the usual head centre
    if pose is not None:
        head, scale, sh = pose.head, pose.scale, pose.shoulders
        if head is not None and scale:
            c = np.array([(h[0] + h[2]) / 2 * pose.aspect, (h[1] + h[3]) / 2])
            d = float(np.linalg.norm(c - head)) / scale
            below_ok = True
            if sh is not None and np.linalg.norm(head - sh) > 0:
                up = (head - sh) / np.linalg.norm(head - sh)
                below_ok = float((c - head) @ up) / scale >= r.head_pose_below
            if d <= r.head_pose_radius and below_ok:
                costs.append(d * scale / ph)
    return min(costs) if costs else None


def _torso_cost(p, b, r: Rules, pose=None) -> float | None:
    pw, ph = p[2] - p[0], p[3] - p[1]
    area = (b[2] - b[0]) * (b[3] - b[1])
    if pw <= 0 or ph <= 0 or area <= 0:
        return None
    u = ((b[0] + b[2]) / 2 - p[0]) / pw
    v = ((b[1] + b[3]) / 2 - p[1]) / ph
    if not (r.torso_side <= u <= 1 - r.torso_side and r.torso_top <= v <= r.torso_bottom):
        return None
    inside = max(0.0, min(b[2], p[2]) - max(b[0], p[0])) * max(0.0, min(b[3], p[3]) - max(b[1], p[1])) / area
    if inside < r.vest_inside:
        return None
    return abs(u - 0.5) * pw / ph + abs(v - 0.45)


def _assign(people, items, cost_fn, r: Rules, poses=None) -> dict[int, int]:
    """Greedy one-to-one matching, cheapest pairs first. Returns {person index: item index}."""
    pairs = []
    for i, p in enumerate(people):
        pose = poses[i] if poses else None
        for j, it in enumerate(items):
            c = cost_fn(p, it, r, pose)
            if c is not None:
                pairs.append((c, i, j))
    taken_p, taken_i, out = set(), set(), {}
    for _, i, j in sorted(pairs):
        if i not in taken_p and j not in taken_i:
            out[i] = j
            taken_p.add(i)
            taken_i.add(j)
    return out


def judge(people: list, helmets: list, vests: list, rules: Rules = Rules(), input_height: float = 640.0,
          poses: list | None = None) -> list[Verdict]:
    """Verdicts for each person box. Each box is a (x1, y1, x2, y2) 0-1 tuple.

    input_height: height of the image the detector saw, in pixels (640 for a 640x640 letterboxed
    square frame; for a 16:9 frame letterboxed into 640 it is 360). Used for the size limits.
    poses: optional, one vision.pose.Pose (or None) per person, for bending / crouching / lying people.
    """
    helmet_of = _assign(people, helmets, _head_cost, rules, poses)
    vest_of = _assign(people, vests, _torso_cost, rules, poses)
    out = []
    for i, p in enumerate(people):
        height_px = (p[3] - p[1]) * input_height
        top_cut = p[1] <= rules.edge
        bottom_cut = p[3] >= 1 - rules.edge
        pose = poses[i] if poses else None
        posture = pose.posture() if pose is not None else None
        if i in helmet_of:
            helmet = WORN
        elif top_cut or height_px < rules.min_height_helmet:
            helmet = UNKNOWN
        elif rules.bent_head_hidden_is_unknown and posture in BENT and pose.head is None:
            helmet = UNKNOWN   # bent away from the camera, head out of sight: judge them once it's back in view
        else:
            helmet = MISSING
        if i in vest_of:
            vest = WORN
        elif top_cut or bottom_cut or height_px < rules.min_height_vest:
            vest = UNKNOWN
        else:
            vest = MISSING
        out.append(Verdict(helmet, vest, helmet_of.get(i), vest_of.get(i), posture))
    return out
