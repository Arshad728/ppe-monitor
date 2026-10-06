"""From the safety officer's "false alarm" button to training data (book, Chapters 14 and 23).

Every event marked "false alarm" on the dashboard keeps its clean frame (Phase 5). Each is turned
into a training image with labels:

1. **Labels to start from:** what the model in use finds in the frame, at the thresholds the rules
   use (configs/ppe.yaml).
2. **What the verdict adds.** A false "no helmet" alarm means the person in the event's box WAS
   wearing one, and the model missed it. So the helmet on their head is looked for again, down to
   confidence 0.05 (the model often sees a navy or dirty helmet, just not surely), at least 15 % of
   the person's width (not the small helmet of someone further back). If one is found,
   it is added as a label: that is exactly the lesson for the model. If none is found at all, the
   image needs someone to draw the box ("needs a box") and is not used until then. The same for
   vests, on the torso. If the "person" was not a person at all (Phase 6 met a tractor at a frame's
   edge), the person label on it is deleted by hand instead: that is the lesson then.
3. **Zone false alarms** are listed for a person to look at but not used automatically: the
   "person" may not have been a person at all, or the zone may be drawn wrongly, and only a person
   can tell which.

Test footage never becomes training data: the model would then be tested on what it was taught.
An event is used only if it comes from a real camera, or from a clip listed in
data/clips/clips_catalog.csv with split "train". The event clips, the workshop test clips and the
simulated cameras are left out.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import PROJECT_ROOT
from ..vision.matching import Det

FEEDBACK_DIR = PROJECT_ROOT / "datasets" / "feedback"
CATALOG = PROJECT_ROOT / "data" / "clips" / "clips_catalog.csv"
NAMES = ("person", "helmet", "vest")
ITEM_CLASS = {"no_helmet": 1, "no_vest": 2}
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts")
TRAIN_SPLITS = ("train", "feedback")


@dataclass
class FeedbackItem:
    event_id: str
    camera: str
    kind: str
    when: str
    box: list                           # the event's person box, 0-1 fractions
    status: str = ""                    # ready / needs_box / review / not_needed / excluded
    reason: str = ""
    labels: list = field(default_factory=list)      # [[cls, x1, y1, x2, y2], ...] 0-1 fractions
    added: list = field(default_factory=list)       # the labels the verdict added (a subset of labels)
    note: str = ""


def load_catalog(path: Path = CATALOG) -> dict[str, str]:
    """clip file name -> split, from data/clips/clips_catalog.csv."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["file"].strip(): (r.get("split") or "").strip() for r in csv.DictReader(f) if r.get("file")}


def eligible_camera(camera_id: str, name: str, catalog: dict[str, str], simulated: set[str]) -> tuple[bool, str]:
    """Whether a camera's events may become training data, and why not."""
    if camera_id in simulated:
        return False, "a simulated camera (configs/cameras.yaml sim_source)"
    for label in (name or "", camera_id):
        base = Path(label).name
        if base.lower().endswith(VIDEO_SUFFIXES):
            split = catalog.get(base)
            if split in TRAIN_SPLITS:
                return True, ""
            return False, (f"a clip ({base}) in the catalog as {split!r}: test footage is never trained on" if split
                           else f"a clip ({base}) not in data/clips/clips_catalog.csv with split train")
    return True, ""


def head_region(box, geom: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    s = geom.get("head_side", 0.05)
    return (x1 - s * w, y1 + geom.get("head_top", -0.10) * h, x2 + s * w, y1 + geom.get("head_bottom", 0.35) * h)


def torso_region(box, geom: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    s = geom.get("torso_side", 0.0)
    return (x1 - s * w, y1 + geom.get("torso_top", 0.15) * h, x2 + s * w, y1 + geom.get("torso_bottom", 0.85) * h)


def _inside(box, region) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def label_false_alarm(item: FeedbackItem, dets: list[Det], thresholds: dict, geom: dict,
                      recover_conf: float = 0.05) -> FeedbackItem:
    """Fill in item.labels / status / reason from the model's detections on the clean frame (all of
    them, down to a low confidence) and the verdict (see the module's description)."""
    geom = geom if isinstance(geom, dict) else dict(vars(geom))        # configs/ppe.yaml's rules: dict or Rules
    kept = [d for d in dets if d.score >= thresholds.get(NAMES[d.cls], 0.25)]
    labels = [[d.cls, *d.box] for d in kept]
    person = tuple(item.box)
    if item.kind == "zone_intrusion":
        item.status, item.reason = "review", "a zone false alarm: look at it (was it a person? is the zone right?)"
        item.labels = labels
        return item
    cls = ITEM_CLASS.get(item.kind)
    if cls is None:
        item.status, item.reason = "review", f"unknown kind {item.kind}"
        return item
    if not any(d.cls == 0 and _iou(d.box, person) >= 0.5 for d in kept):
        labels.append([0, *person])             # the person the event was about
        item.added.append([0, *person])
    region = head_region(person, geom) if cls == 1 else torso_region(person, geom)
    # the person's own helmet (vest) is about their size: not the small one of someone behind them
    pw = max(1e-6, person[2] - person[0])
    min_w = 0.15 if cls == 1 else 0.3

    def theirs(d: Det) -> bool:
        return d.cls == cls and _inside(d.box, region) and (d.box[2] - d.box[0]) / pw >= min_w

    if any(theirs(d) for d in kept):
        item.status = "not_needed"
        item.reason = (f"the model already finds the {NAMES[cls]} at the normal threshold: the rule, not the detector, "
                       "judged wrongly here, so this image teaches the detector nothing")
        item.labels = labels
        return item
    cands = [d for d in dets if d.score >= recover_conf and theirs(d)]
    if cands:
        best = max(cands, key=lambda d: d.score)
        labels.append([cls, *best.box])
        item.added.append([cls, *best.box])
        item.status = "ready"
        item.reason = f"{NAMES[cls]} on the person found again at confidence {best.score:.2f}, added as a label"
    else:
        item.status = "needs_box"
        item.reason = (f"no {NAMES[cls]} found on the person even at confidence {recover_conf}: draw its box by hand "
                       f"(labels/{item.event_id}.txt, YOLO format), then set its status to ready in feedback.json. "
                       f"If it is not a person at all (a machine, a sign, a shadow), delete that person line instead: "
                       f"that is the lesson")
    item.labels = labels
    return item


def write_label_file(path: Path, labels: list) -> None:
    lines = []
    for cls, x1, y1, x2, y2 in labels:
        x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
        if x2 - x1 > 1e-4 and y2 - y1 > 1e-4:
            lines.append(f"{int(cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_review(img: np.ndarray, item: FeedbackItem, width: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / w
    out = cv2.resize(img, (width, int(h * scale)))
    H, W = out.shape[:2]
    colours = {0: (200, 200, 200), 1: (0, 200, 0), 2: (0, 200, 200)}
    added = {tuple(round(v, 4) for v in a) for a in item.added}
    for lab in item.labels:
        cls, x1, y1, x2, y2 = lab
        new = tuple(round(v, 4) for v in lab) in added
        cv2.rectangle(out, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 140, 255) if new else colours[int(cls)],
                      3 if new else 1)
    x1, y1, x2, y2 = item.box
    cv2.rectangle(out, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 0, 255), 1, cv2.LINE_4)
    return out


def save_index(folder: Path, items: list[FeedbackItem]) -> None:
    (folder / "feedback.json").write_text(json.dumps([asdict(i) for i in items], indent=1), encoding="utf-8")


def load_index(folder: Path = FEEDBACK_DIR) -> list[FeedbackItem]:
    p = folder / "feedback.json"
    if not p.is_file():
        return []
    return [FeedbackItem(**d) for d in json.loads(p.read_text(encoding="utf-8"))]


def skipped(folder: Path = FEEDBACK_DIR) -> set[str]:
    p = folder / "skip.txt"
    if not p.is_file():
        return set()
    return {line.split("#")[0].strip() for line in p.read_text(encoding="utf-8").splitlines() if line.split("#")[0].strip()}


def base_dataset(models_dir: Path | None = None) -> Path:
    """The data.yaml the model in use (the newest .pt in models/) was trained on, from its model card:
    a retrain on false alarms adds to that. datasets/ppe4 if the card doesn't say or the dataset is gone."""
    models_dir = models_dir or PROJECT_ROOT / "models"
    fallback = PROJECT_ROOT / "datasets" / "ppe4" / "data.yaml"
    newest = max(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if newest is None or not newest.with_suffix(".json").is_file():
        return fallback
    data = json.loads(newest.with_suffix(".json").read_text(encoding="utf-8")).get("data")
    if not data:
        return fallback
    path = Path(data) if Path(data).is_absolute() else PROJECT_ROOT / data
    return path if path.is_file() else fallback


def build_dataset(base_yaml: Path, out_dir: Path, feedback_dir: Path = FEEDBACK_DIR, repeat: int = 5) -> dict:
    """datasets/<out>: the base dataset's splits unchanged, plus every "ready" feedback image (not in
    skip.txt) added to the TRAINING split `repeat` times, so a few dozen hard examples aren't lost
    among thousands. Validation and test stay exactly as they were, so the old and new models are
    compared on the same photos. Returns counts."""
    import yaml

    base = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    root = Path(base["path"])
    if not root.is_absolute():
        root = (base_yaml.parent / root).resolve()
    skip = skipped(feedback_dir)
    ready = [i for i in load_index(feedback_dir) if i.status == "ready" and i.event_id not in skip
             and (feedback_dir / "images" / f"{i.event_id}.jpg").is_file()]

    def files(spec) -> list[str]:
        p = Path(spec) if Path(spec).is_absolute() else root / spec
        if p.is_dir():
            return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        return [str(x if Path(x).is_absolute() else root / x) for x in
                (line.strip() for line in p.read_text(encoding="utf-8").splitlines()) if x]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    train = files(base["train"]) + [str((feedback_dir / "images" / f"{i.event_id}.jpg").resolve())
                                    for i in ready for _ in range(repeat)]
    (out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    cfg = {"path": str(out_dir.resolve()), "train": "train.txt"}
    for key, spec in base.items():
        if key in ("path", "train", "names", "nc"):
            continue
        (out_dir / f"{key}.txt").write_text("\n".join(files(spec)) + "\n", encoding="utf-8")
        cfg[key] = f"{key}.txt"
    cfg["names"] = base["names"]
    header = (f"# Written by scripts/feedback.py build: {base_yaml} plus {len(ready)} false alarm(s) from the dashboard\n"
              f"# (each {repeat} times, training split only). Rebuild rather than edit.\n")
    (out_dir / "data.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"base_train": len(train) - len(ready) * repeat, "feedback": len(ready), "repeat": repeat,
            "train": len(train), "skipped": len(skip)}
