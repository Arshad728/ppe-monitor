"""Checking the extra datasets' labels before training on them (datasets/ppe4).

A training image where a helmet, vest or person is visible but not labelled teaches the model
that such a thing is background. Public datasets have such gaps (Phase 1, failure case 1), and
two of the extra ones have them by design: GDUT-HWD labels only heads and helmets, SH17 misses
many vests. Two steps, both with models that are already trained:

1. **People for datasets without person boxes** (Source.persons == "pseudo", GDUT-HWD): a large
   COCO model (yolo26l) draws them. Its people can't be checked against labels, so a different
   check is used: every labelled head (with or without a helmet) must sit at the top of one of
   those people. If a head has no person, the model missed someone, and the image is left out.
2. **Unlabelled objects** (Source.audit, e.g. {"vest": 0.5}): when the current PPE model (or, for
   people, the COCO model) finds that class with at least that confidence where no label of the
   class is, the image is left out.

Only whole images are dropped; no label is changed. Every drop is counted in the build report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .labels import ClassMap, normalise
from .readers import Box, Sample
from .sources import Source

COCO_MODEL = "yolo26l.pt"      # person class 0; downloaded into models/pretrained/ on first use
PSEUDO_CONF = 0.35             # people drawn by the COCO model at or above this confidence
PSEUDO_IMGSZ = 960             # GDUT-HWD has many small, distant people
MATCH_IOU = 0.3                # a detection this close to a label of its class counts as labelled


@dataclass
class AuditStats:
    checked: int = 0
    kept: int = 0
    pseudo_people: int = 0
    dropped: Counter = field(default_factory=Counter)      # reason -> images
    dropped_images: list = field(default_factory=list)     # (image name, reason), for review


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def head_has_person(head: Box, people: list[tuple]) -> bool:
    """The head's centre lies in the top part of a person box (above its middle, 5 % margin above)."""
    cx, cy = (head.x1 + head.x2) / 2, (head.y1 + head.y2) / 2
    for x1, y1, x2, y2 in people:
        h = y2 - y1
        if x1 <= cx <= x2 and y1 - 0.05 * h <= cy <= y1 + 0.45 * h:
            return True
    return False


def audit_source(source: Source, samples: list[Sample], class_map: ClassMap, *, ppe_model, coco_model,
                 absent_helmet: set[str], device: str | None = None, log=print) -> tuple[list[Sample], AuditStats]:
    """The samples to keep (people added where the source has none), and what was dropped and why."""
    from ..vision.detector import predict

    stats = AuditStats(checked=len(samples))
    if not samples:
        return [], stats
    images = [s.image for s in samples]
    need_coco = source.persons == "pseudo" or "person" in source.audit
    classes = {"person": 0, "helmet": 1, "vest": 2}
    ppe_preds = predict(ppe_model, images, conf=0.25, device=device) if set(source.audit) - {"person"} else None
    coco_preds = None
    if need_coco:
        name = Path(getattr(coco_model, "ckpt_path", None) or COCO_MODEL).name
        log(f"    {source.name}: finding people with {name} in {len(images)} images")
        coco_preds = predict(coco_model, images, conf=min(PSEUDO_CONF, source.audit.get("person", 1.0)),
                             imgsz=PSEUDO_IMGSZ, device=device)
    kept = []
    for k, s in enumerate(samples):
        labelled = {"person": [], "helmet": [], "vest": []}
        heads = []
        for b in s.boxes:
            d = class_map.decide(b.label)
            if d.kind == "class":
                labelled[d.class_name].append((b.x1, b.y1, b.x2, b.y2))
                if d.class_name == "helmet":
                    heads.append(b)
            elif normalise(b.label) in absent_helmet:
                heads.append(b)
        reason = ""
        boxes = list(s.boxes)
        if source.persons == "pseudo":
            people = [d.box for d in coco_preds[k] if d.cls == 0 and d.score >= PSEUDO_CONF]
            if any(not head_has_person(h, people) for h in heads):
                reason = "a labelled head with no person found around it"
            else:
                boxes += [Box("person", *p) for p in people]
                labelled["person"] = people
        for cls, conf in source.audit.items():
            if reason:
                break
            preds = coco_preds[k] if cls == "person" else ppe_preds[k]
            want = 0 if cls == "person" else classes[cls]
            for d in preds:
                if d.cls == want and d.score >= conf and all(_iou(d.box, b) < MATCH_IOU for b in labelled[cls]):
                    reason = f"an unlabelled {cls} (the model is {d.score:.2f} sure)"
                    break
        if reason:
            key = reason.split(" (")[0]
            stats.dropped[key] += 1
            stats.dropped_images.append((s.image.name, reason))
            continue
        if source.persons == "pseudo":
            stats.pseudo_people += len(labelled["person"])
        kept.append(Sample(s.source, s.split, s.image, boxes))
    stats.kept = len(kept)
    return kept, stats
