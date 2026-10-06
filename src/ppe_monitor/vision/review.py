"""Looking at the detector's mistakes, one picture at a time (book, Chapter 18: "manually review
a sample of the model's mistakes").

Every error at the working threshold gets a type:
  missed <class>                    a labelled object no prediction found
  false <class> (duplicate)         a second box on an object that was already found
  false <class> (wrong class)       right place, wrong label (e.g. a helmet called a vest)
  false <class> (misplaced)         near a labelled object of that class (IoU 0.1-0.5), but the
                                    box is too far off to count: cut short, or covering two people
  false <class> (false alarm)       nothing labelled there: background, a look-alike (an orange
                                    shirt as a vest, a cap as a helmet), or an object the
                                    labeller missed. Confident ones are worth a second look.

Each miss also gets a likely cause, which says what would fix it:
  crowded          a good box was there, but it was matched to a neighbour (overlapping people)
  misplaced box    a box of that class was nearby, but too far off (IoU 0.1-0.5)
  below threshold  the model found it (IoU >= 0.5) with a confidence under the working threshold
  not detected     nothing of that class nearby at any confidence

These are the error types of the TIDE toolbox (Bolya et al., 2020), simplified.
Annotated examples of each type are saved to a folder with an index.html to browse them.
"""

from __future__ import annotations

import html
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..data.build import size_bucket
from .detector import class_names, predict, read_labels, split_items
from .matching import Det, iou_matrix, match_image

GREEN, RED, MAGENTA, GREY = (80, 200, 80), (40, 40, 230), (200, 60, 200), (170, 170, 170)
NEAR = 0.1          # IoU from which a box counts as "near" an object rather than elsewhere
LOW_CONF = 0.01     # predictions kept (below the working threshold) to tell "below threshold" misses apart
MISS_CAUSES = ("crowded", "misplaced box", "below threshold", "not detected")


@dataclass
class Error:
    kind: str
    image: Path
    det: Det
    size: str


def _max_iou(box, boxes) -> float:
    return float(iou_matrix(np.array([box]), np.array(boxes)).max()) if len(boxes) else 0.0


def classify(preds: list[Det], gts: list[Det], names: list[str]) -> tuple[list[tuple[str, Det]], set[int], set[int]]:
    """Errors in one image, plus the indices of correct predictions and found labelled boxes."""
    errors, good_preds, found = [], set(), set()
    for c, name in enumerate(names):
        pi = [i for i, p in enumerate(preds) if p.cls == c]
        gi = [j for j, g in enumerate(gts) if g.cls == c]
        m = match_image([preds[i] for i in pi], [gts[j] for j in gi])
        good_preds |= {pi[k] for k in m.tp}
        found |= {gi[k] for k in m.pairs.values()}
        errors += [(f"missed {name}", gts[gi[k]]) for k in m.fn]
        same = [gts[j].box for j in gi]
        other = [g.box for g in gts if g.cls != c]
        for k in m.fp:
            p = preds[pi[k]]
            iou_same = _max_iou(p.box, same)
            if iou_same >= 0.5:
                sub = "duplicate"
            elif _max_iou(p.box, other) >= 0.5:
                sub = "wrong class"
            elif iou_same >= NEAR:
                sub = "misplaced"
            else:
                sub = "false alarm"
            errors.append((f"false {name} ({sub})", p))
    return errors, good_preds, found


def miss_cause(missed: Det, preds: list[Det], threshold: float) -> str:
    """Why a labelled object was missed, judged from all predictions of its class (any confidence)."""
    same = [p for p in preds if p.cls == missed.cls]
    above = [p.box for p in same if p.score >= threshold]
    below = [p.box for p in same if p.score < threshold]
    best_above = _max_iou(missed.box, above)
    if best_above >= 0.5:
        return "crowded"            # a good box existed, but matching gave it to an overlapping neighbour
    if best_above >= NEAR:
        return "misplaced box"
    if _max_iou(missed.box, below) >= 0.5:
        return "below threshold"
    return "not detected"


def _draw(image: np.ndarray, gts: list[Det], preds: list[Det], good: set[int], errors: list[tuple[str, Det]],
          names, focus: str):
    h, w = image.shape[:2]
    scale = min(1.0, 960 / w)
    img = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image.copy()
    h, w = img.shape[:2]

    def rect(d: Det, colour, thick, text):
        x1, y1, x2, y2 = int(d.box[0] * w), int(d.box[1] * h), int(d.box[2] * w), int(d.box[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), colour, -1)
            cv2.putText(img, text, (x1 + 2, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    focus_cls = names.index(focus.split()[1])
    for g in gts:  # labelled boxes: thin grey
        rect(g, GREY, 1, "")
    for i, p in enumerate(preds):  # the model's other boxes of this class, for context: thin
        if p.cls == focus_cls:
            rect(p, GREEN if i in good else MAGENTA, 1, "")
    for kind, d in errors:
        if kind == focus:
            label = f"MISSED {names[d.cls]}" if kind.startswith("missed") else f"{names[d.cls]} {d.score:.2f}?"
            rect(d, RED if kind.startswith("missed") else MAGENTA, 3, label)
    return img


def review(model, data_yaml: str | Path, split: str, threshold: float, out_dir: Path, *, imgsz: int = 640,
           device: str | None = None, per_kind: int = 24, confident: float = 0.7) -> dict:
    items = split_items(data_yaml, split)
    names = class_names(data_yaml)
    gts_all = [read_labels(lbl) for _, lbl in items]
    all_preds = predict(model, [img for img, _ in items], conf=min(LOW_CONF, threshold), imgsz=imgsz, device=device)
    preds_all = [[p for p in ps if p.score >= threshold] for ps in all_preds]

    by_kind: dict[str, list[tuple[int, int]]] = defaultdict(list)  # kind -> [(image index, errors in image)]
    counts, missed_by_size, missed_by_cause = Counter(), defaultdict(Counter), defaultdict(Counter)
    per_image = []
    confident_false = 0
    for n, (preds, gts) in enumerate(zip(preds_all, gts_all)):
        errors, good, _ = classify(preds, gts, names)
        per_image.append((errors, good))
        kinds = Counter(k for k, _ in errors)
        for kind, cnt in kinds.items():
            counts[kind] += cnt
            by_kind[kind].append((n, cnt))
        for kind, d in errors:
            if kind.startswith("missed"):
                missed_by_size[kind][size_bucket((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))] += 1
                missed_by_cause[kind][miss_cause(d, all_preds[n], threshold)] += 1
            elif d.score >= confident and "false alarm" in kind:
                confident_false += 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    sections = []
    for kind in sorted(by_kind, key=lambda k: -counts[k]):
        slug = kind.replace(" ", "_").replace("(", "").replace(")", "")
        (out_dir / slug).mkdir()
        chosen = sorted(by_kind[kind], key=lambda t: -t[1])[:per_kind]
        thumbs = []
        for rank, (n, _) in enumerate(chosen):
            image_path = items[n][0]
            errors, good = per_image[n]
            img = _draw(cv2.imread(str(image_path)), gts_all[n], preds_all[n], good, errors, names, kind)
            name = f"{rank:02d}_{image_path.stem[:60]}.jpg"
            cv2.imwrite(str(out_dir / slug / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumbs.append(f'<a href="{slug}/{name}"><img src="{slug}/{name}" loading="lazy" '
                          f'title="{html.escape(image_path.name)}"></a>')
        sections.append(f"<h2>{html.escape(kind)} <small>{counts[kind]} in {len(by_kind[kind])} images"
                        f"{' - showing ' + str(len(chosen)) if len(chosen) < len(by_kind[kind]) else ''}</small></h2>"
                        f"<div class=grid>{''.join(thumbs)}</div>")

    def cause_cells(k):
        return "".join(f"<td>{missed_by_cause[k].get(c, 0) if k.startswith('missed') else ''}</td>" for c in MISS_CAUSES)

    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{counts[k]}</td><td>"
                   + (" / ".join(f"{missed_by_size[k].get(s, 0)}" for s in ("small", "medium", "large"))
                      if k.startswith("missed") else "") + f"</td>{cause_cells(k)}</tr>"
                   for k in sorted(counts, key=lambda k: -counts[k]))
    cause_heads = "".join(f"<th>{c}</th>" for c in MISS_CAUSES)
    page = f"""<!doctype html><meta charset=utf-8><title>Detector mistakes - {split}</title>
<style>body{{font:14px -apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px}}
.grid img{{width:100%;border:1px solid #ccc}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 10px}}
small{{color:#666;font-weight:normal}}</style>
<h1>Detector mistakes on the {split} split</h1>
<p>{len(items)} images, confidence threshold {threshold:.3f}. Thin grey boxes are the labels; thin green and
magenta boxes are the model's other correct and incorrect boxes of the same class. In each section the thick
<b style="color:#e62828">red</b> boxes are missed objects and thick <b style="color:#c83cc8">magenta</b> boxes are
false predictions of that type. {confident_false} false alarms had confidence >= {confident}: check those for objects
the labeller missed.</p>
<p>Causes of misses: <b>crowded</b> - a good box existed but went to an overlapping neighbour; <b>misplaced box</b> -
a box was nearby but too far off; <b>below threshold</b> - found, but with confidence under {threshold:.3f};
<b>not detected</b> - nothing nearby at any confidence.</p>
<table><tr><th>Error type</th><th>Count</th><th>Missed: small / medium / large</th>{cause_heads}</tr>{rows}</table>
{''.join(sections)}"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    summary = {"split": split, "images": len(items), "threshold": threshold, "counts": dict(counts),
               "missed_by_size": {k: dict(v) for k, v in missed_by_size.items()},
               "missed_by_cause": {k: dict(v) for k, v in missed_by_cause.items()},
               "confident_false_alarms": confident_false}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
