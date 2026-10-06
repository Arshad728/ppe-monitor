"""Measuring the detector: per-class AP@50, the working confidence threshold, and recall by
object size (small objects are the known weak spot: distant workers on CCTV)."""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..data.build import size_bucket
from .detector import class_names, predict, read_labels, split_items
from .matching import Det, average_precision, best_f1_threshold, match_image

SIZES = ("small", "medium", "large")


@dataclass
class ClassResult:
    name: str
    objects: int
    ap50: float
    precision: float   # at the working threshold
    recall: float      # at the working threshold
    recall_by_size: dict[str, tuple[int, float]] = field(default_factory=dict)  # size -> (objects, recall)


@dataclass
class EvalResult:
    split: str
    images: int
    threshold: float
    map50: float
    classes: list[ClassResult]
    map50_ultralytics: float | None = None
    map50_95_ultralytics: float | None = None

    def table(self) -> str:
        head = "| Class | Objects | AP@50 | Precision | Recall | Recall: small / medium / large |\n|---|---|---|---|---|---|\n"
        rows = []
        for c in self.classes:
            sizes = " / ".join(f"{r:.2f} ({n})" if n else "- (0)" for n, r in (c.recall_by_size[s] for s in SIZES))
            rows.append(f"| {c.name} | {c.objects} | {c.ap50:.3f} | {c.precision:.3f} | {c.recall:.3f} | {sizes} |")
        return head + "\n".join(rows) + f"\n| **all** | {sum(c.objects for c in self.classes)} | **{self.map50:.3f}** | | | |\n"

    def to_dict(self) -> dict:
        return asdict(self)


def pick_threshold(pairs: list[tuple[list[Det], list[Det]]]) -> float:
    """The confidence threshold with the best F1 (the best balance of false alarms and misses),
    rounded down to 3 decimals so that the chosen prediction itself is still kept."""
    threshold, _ = best_f1_threshold(pairs)
    return math.floor(threshold * 1000) / 1000


def evaluate(model, data_yaml: str | Path, split: str, *, threshold: float | None = None, imgsz: int = 640,
             device: str | None = None, with_ultralytics: bool = True) -> EvalResult:
    items = split_items(data_yaml, split)
    names = class_names(data_yaml)
    gts = [read_labels(lbl) for _, lbl in items]
    preds = predict(model, [img for img, _ in items], imgsz=imgsz, device=device)
    pairs = list(zip(preds, gts))
    if threshold is None:
        threshold = pick_threshold(pairs)

    classes = []
    for c, name in enumerate(names):
        scores, hits, n_gt = [], [], 0
        size_total = {s: 0 for s in SIZES}
        size_found = {s: 0 for s in SIZES}
        tp_t = fp_t = 0
        for p, g in pairs:
            pc = [d for d in p if d.cls == c]
            gc = [d for d in g if d.cls == c]
            n_gt += len(gc)
            m = match_image(pc, gc)
            for i, d in enumerate(pc):
                scores.append(d.score)
                hits.append(i in m.pairs)
            kept = [d for d in pc if d.score >= threshold]
            mk = match_image(kept, gc)
            tp_t, fp_t = tp_t + len(mk.tp), fp_t + len(mk.fp)
            found = set(mk.pairs.values())
            for j, d in enumerate(gc):
                area = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                size_total[size_bucket(area)] += 1
                size_found[size_bucket(area)] += j in found
        classes.append(ClassResult(
            name=name, objects=n_gt, ap50=average_precision(scores, hits, n_gt),
            precision=tp_t / max(tp_t + fp_t, 1), recall=tp_t / max(n_gt, 1),
            recall_by_size={s: (size_total[s], size_found[s] / size_total[s] if size_total[s] else float("nan"))
                            for s in SIZES}))
    aps = [c.ap50 for c in classes if not math.isnan(c.ap50)]
    result = EvalResult(split=split, images=len(items), threshold=threshold,
                        map50=float(np.mean(aps)) if aps else float("nan"), classes=classes)

    if with_ultralytics:  # cross-check against Ultralytics' own implementation
        # Written to a scratch folder that is overwritten each time, instead of a new runs/detect/val-N per call.
        metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, plots=False, verbose=False,
                            project=str(Path(tempfile.gettempdir()) / "ppe_monitor_val"), name=split, exist_ok=True)
        result.map50_ultralytics = float(metrics.box.map50)
        result.map50_95_ultralytics = float(metrics.box.map)
    return result
