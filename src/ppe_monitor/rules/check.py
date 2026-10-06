"""Checking the PPE rules against people whose helmet / vest status is known from the labels.

For every labelled person with a known answer (see data/compliance.py), the rule is run twice:
  - on the labelled boxes: does the geometry itself give the right answer?
  - on the detector's boxes: what the live system would decide from one frame. Each frame's
    answer is compared with the truth, which gives the per-frame error rates the temporal logic
    (rules/events.py) has to smooth over.

Rates are split by the person's height in pixels at the detector's input, because small people
are where the detector's helmet recall drops (docs/phase1_failure_cases.md, case 6).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..data.compliance import PersonTruth, absence_labels, person_truths
from ..data.labels import ClassMap
from ..data.readers import read_source
from ..data.sources import SOURCES, fetch
from ..vision.matching import Det, iou_matrix
from ..vision.pose import match_poses, poses_from_result
from .ppe import ITEMS, MISSING, UNKNOWN, WORN, Rules, judge

SIZE_BUCKETS = ((0, 64, "< 64 px"), (64, 128, "64-128 px"), (128, 256, "128-256 px"), (256, 1e9, ">= 256 px"))
POSTURES = ("upright", "bending", "crouching", "lying", "not known")
NOT_FOUND = "person not found"


@dataclass
class Case:
    image: Path
    input_height: float           # height of the image at the detector's input (640 on the long side)
    truths: list[PersonTruth]
    gt_boxes: dict[str, list]     # labelled person / helmet / vest boxes, for the geometry-only check
    poses: list | None = None     # every pose the keypoint model found in the image (attach_poses)
    aspect: float = 1.0           # image width / height


@dataclass
class Tally:
    """counts[item][truth][answer], and the same split by person size and by posture."""
    counts: dict = field(default_factory=lambda: {i: {True: Counter(), False: Counter()} for i in ITEMS})
    by_size: dict = field(default_factory=lambda: {i: {True: defaultdict(Counter), False: defaultdict(Counter)}
                                                   for i in ITEMS})
    by_posture: dict = field(default_factory=lambda: {i: {True: defaultdict(Counter), False: defaultdict(Counter)}
                                                      for i in ITEMS})

    def add(self, item: str, truth: bool, answer: str, size: str, posture: str | None = None) -> None:
        self.counts[item][truth][answer] += 1
        self.by_size[item][truth][size][answer] += 1
        self.by_posture[item][truth][posture or "not known"][answer] += 1


def size_label(height_px: float) -> str:
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= height_px < hi:
            return name
    return SIZE_BUCKETS[-1][2]


def load_cases(data_yaml: str | Path, split: str) -> list[Case]:
    """Known-status people for every image of a ppe3 split, read from the raw source labels."""
    data_dir = Path(data_yaml).parent
    class_map, absent = ClassMap.from_yaml(), absence_labels()
    raw = {}
    for s in SOURCES.values():
        samples, _ = read_source(s.fmt, fetch(s), s.name, s.splits)
        for x in samples:
            raw[f"{s.name}__{x.split}__{x.image.stem}"] = x
    cases = []
    for image in sorted((data_dir / "images" / split).iterdir()):
        sample = raw.get(image.stem)
        if sample is None:
            continue
        truths = [t for t in person_truths(sample.boxes, class_map, absent) if t.helmet is not None or t.vest is not None]
        if not truths:
            continue
        h, w = cv2.imread(str(image)).shape[:2]
        gt = defaultdict(list)
        for b in sample.boxes:
            d = class_map.decide(b.label)
            if d.kind == "class":
                gt[d.class_name].append((b.x1, b.y1, b.x2, b.y2))
        cases.append(Case(image, 640.0 * h / max(h, w), truths, dict(gt), aspect=w / h))
    return cases


def attach_poses(cases: list[Case], pose_model, *, imgsz: int = 640, device: str | None = None,
                 keypoint_conf: float = 0.5, batch: int = 16) -> None:
    """Run the keypoint model on every case image and keep the poses it finds."""
    batch = getattr(pose_model, "ppe_batch", batch)
    extra = dict(getattr(pose_model, "ppe_predict_args", {}))
    device = extra.pop("device", device)
    for start in range(0, len(cases), batch):
        chunk = cases[start:start + batch]
        images = [str(c.image) for c in chunk]
        results = pose_model.predict(images if batch > 1 else images[0], conf=0.25, imgsz=imgsz, device=device,
                                     verbose=False, **extra)
        for c, r in zip(chunk, results):
            c.poses = poses_from_result(r, c.aspect, keypoint_conf)


def _truth_postures(c: Case) -> list[str | None]:
    """Posture of each known-status person, from the keypoints matched to their labelled box."""
    if c.poses is None:
        return [None] * len(c.truths)
    return [p.posture() if p is not None else None for p in match_poses([t.box for t in c.truths], c.poses)]


def check_geometry(cases: list[Case], rules: Rules, use_poses: bool = True) -> Tally:
    """The rule applied to the labelled boxes themselves (with keypoints, if attached and use_poses)."""
    tally = Tally()
    for c in cases:
        people = c.gt_boxes.get("person", [])
        poses = match_poses(people, c.poses) if (use_poses and c.poses is not None) else None
        verdicts = judge(people, c.gt_boxes.get("helmet", []), c.gt_boxes.get("vest", []), rules, c.input_height, poses)
        index = {p: k for k, p in enumerate(people)}
        for t, posture in zip(c.truths, _truth_postures(c)):
            v = verdicts[index[t.box]]
            size = size_label((t.box[3] - t.box[1]) * c.input_height)
            for item in ITEMS:
                truth = getattr(t, item)
                if truth is not None:
                    tally.add(item, truth, getattr(v, item), size, posture)
    return tally


def check_detections(cases: list[Case], detections: list[list[Det]], rules: Rules,
                     thresholds: dict[str, float], names: list[str], use_poses: bool = True) -> Tally:
    """The rule applied to the detector's boxes, one frame per image."""
    tally = Tally()
    cls = {n: names.index(n) for n in ("person", *ITEMS)}
    for c, dets in zip(cases, detections):
        pick = {n: [d.box for d in dets if d.cls == k and d.score >= thresholds[n]] for n, k in cls.items()}
        people = pick["person"]
        poses = match_poses(people, c.poses) if (use_poses and c.poses is not None) else None
        verdicts = judge(people, pick["helmet"], pick["vest"], rules, c.input_height, poses)
        ious = iou_matrix(np.array([t.box for t in c.truths]), np.array(people)) if people else None
        used = set()
        postures = _truth_postures(c)
        for k, t in enumerate(c.truths):
            size = size_label((t.box[3] - t.box[1]) * c.input_height)
            match = None
            if ious is not None:
                for j in np.argsort(-ious[k]):
                    if ious[k, j] < 0.5:
                        break
                    if j not in used:
                        match = int(j)
                        used.add(match)
                        break
            for item in ITEMS:
                truth = getattr(t, item)
                if truth is None:
                    continue
                tally.add(item, truth, NOT_FOUND if match is None else getattr(verdicts[match], item), size,
                          postures[k])
    return tally


def rates(counter: Counter) -> dict:
    """Shares of each answer among people the detector found."""
    found = counter[WORN] + counter[MISSING] + counter[UNKNOWN]
    judged = counter[WORN] + counter[MISSING]
    return {"people": sum(counter.values()), "found": found, "judged": judged,
            "worn": counter[WORN] / judged if judged else float("nan"),
            "missing": counter[MISSING] / judged if judged else float("nan"),
            "unknown": counter[UNKNOWN] / found if found else float("nan")}


def summary(tally: Tally) -> dict:
    """Per item: the per-frame error rates that matter for alerts.
    false_alarm: wearing it, but judged missing (would count toward a false alert)
    missed:      not wearing it, but judged worn (a violation the frame hides)"""
    out = {}
    for item in ITEMS:
        wearing, bare = rates(tally.counts[item][True]), rates(tally.counts[item][False])
        sizes = {}
        for _, _, name in SIZE_BUCKETS:
            w = rates(tally.by_size[item][True][name])
            b = rates(tally.by_size[item][False][name])
            sizes[name] = {"false_alarm": w["missing"], "missed": b["worn"], "unknown_wearing": w["unknown"],
                           "unknown_bare": b["unknown"], "judged_wearing": w["judged"], "judged_bare": b["judged"],
                           "people_wearing": w["people"], "people_bare": b["people"]}
        postures = {}
        for name in POSTURES:
            w = rates(tally.by_posture[item][True][name])
            b = rates(tally.by_posture[item][False][name])
            postures[name] = {"false_alarm": w["missing"], "missed": b["worn"], "unknown_wearing": w["unknown"],
                              "unknown_bare": b["unknown"], "judged_wearing": w["judged"], "judged_bare": b["judged"],
                              "people_wearing": w["people"], "people_bare": b["people"]}
        out[item] = {"by_posture": postures, "false_alarm": wearing["missing"], "missed": bare["worn"],
                     "unknown_wearing": wearing["unknown"], "unknown_bare": bare["unknown"],
                     "people_wearing": wearing["people"], "people_bare": bare["people"],
                     "found_wearing": wearing["found"], "found_bare": bare["found"],
                     "judged_wearing": wearing["judged"], "judged_bare": bare["judged"], "by_size": sizes}
    return out
