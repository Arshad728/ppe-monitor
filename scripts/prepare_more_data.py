#!/usr/bin/env python3
"""Add the extra public datasets to the Phase 1 data, for fine-tuning the detector (datasets/ppe4).

    python scripts/prepare_more_data.py                 # ~20-30 min the first time (downloads ~1.4 GB)
    python scripts/prepare_more_data.py --sources chv gdut-hwd

Why: the Phase 1 model misses navy, brown and white helmets in an indoor workshop
(docs/real_clips.md). The extra datasets (src/ppe_monitor/data/extra_sources.py):

- CHV: people, vests, helmets in four colours;
- GDUT-HWD: helmets in four colours, many small people (no person boxes: a COCO model adds them);
- SH17: indoor industry (the images with a helmet or vest, and a sample of people without either).

Steps: download (cached in datasets/raw/, checked against pinned checksums); read; check the
labels with the current model and a COCO model, leaving out images with visible but unlabelled
people, helmets or vests (src/ppe_monitor/data/audit.py); merge with datasets/ppe3, which is
kept exactly as it is. Result: datasets/ppe4/ with report.md, and a picture sheet of kept and
left-out images per dataset in runs/data_review/ppe4/.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.data.audit import COCO_MODEL, audit_source
from ppe_monitor.data.build import SPLITS, UnknownLabels
from ppe_monitor.data.build_ext import build_extended
from ppe_monitor.data.compliance import absence_labels
from ppe_monitor.data.extra_sources import EXTRA_SOURCES, fetch_extra
from ppe_monitor.data.labels import ClassMap
from ppe_monitor.data.readers import read_source
from ppe_monitor.data.sources import RAW_DIR, SourceError
from ppe_monitor.device import best_device
from ppe_monitor.vision.detector import MODELS_DIR, load_model

BASE = PROJECT_ROOT / "datasets" / "ppe3"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "ppe4"
REVIEW = PROJECT_ROOT / "runs" / "data_review"


def sheet(samples, dropped_names, root_images, path: Path, class_map: ClassMap, n: int = 12, seed: int = 0) -> None:
    """Kept images with their boxes (top rows), and left-out ones with the reason (bottom rows)."""
    colour = {"person": (0, 200, 0), "helmet": (0, 220, 255), "vest": (0, 140, 255)}
    rnd = random.Random(seed)
    tiles = []
    for s in rnd.sample(samples, min(n, len(samples))):
        im = cv2.imread(str(s.image))
        h, w = im.shape[:2]
        for b in s.boxes:
            d = class_map.decide(b.label)
            c = colour.get(d.class_name) if d.kind == "class" else (200, 0, 200)
            cv2.rectangle(im, (int(b.x1 * w), int(b.y1 * h)), (int(b.x2 * w), int(b.y2 * h)), c, max(2, w // 400))
        tiles.append(cv2.resize(im, (400, 300)))
    for name, reason in rnd.sample(dropped_names, min(n // 2, len(dropped_names))):
        im = cv2.imread(str(root_images[name])) if name in root_images else None
        if im is None:
            continue
        im = cv2.resize(im, (400, 300))
        cv2.rectangle(im, (0, 0), (399, 22), (0, 0, 160), -1)
        cv2.putText(im, "LEFT OUT: " + reason.split(" (")[0][:48], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(tiles[-1] * 0)
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(EXTRA_SOURCES), choices=list(EXTRA_SOURCES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--raw", default=str(RAW_DIR), help="download cache (default datasets/raw)")
    parser.add_argument("--sh17-negatives", type=int, default=700,
                        help="SH17 images of people with no helmet and no vest to add (default 700)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--coco-model", default=COCO_MODEL, help=argparse.SUPPRESS)   # a smaller one for quick tests
    args = parser.parse_args()

    if not (BASE / "data.yaml").is_file():
        print("No datasets/ppe3. Phase 1 comes first: python scripts/prepare_data.py")
        return 1
    start = time.monotonic()
    device = best_device() if args.device == "auto" else args.device
    class_map = ClassMap.from_yaml()
    absent_helmet = {k for k, item in absence_labels().items() if item == "helmet"}

    print("[1/4] Getting the extra datasets")
    roots = {}
    try:
        for name in args.sources:
            src = EXTRA_SOURCES[name]
            kw = {"negatives": args.sh17_negatives} if src.fmt == "sh17" else {}
            roots[name] = fetch_extra(src, Path(args.raw), **kw)
    except (SourceError, OSError) as exc:
        print(f"\nCould not get the datasets: {exc}")
        return 1

    print("\n[2/4] Reading them")
    read = {}
    for name in args.sources:
        src = EXTRA_SOURCES[name]
        samples, stats = read_source(src.fmt, roots[name], src.name, src.splits)
        read[name] = (samples, stats)
        print(f"  {name}: {stats.images} images, {sum(stats.labels.values())} boxes "
              f"({', '.join(f'{k} {v}' for k, v in stats.labels.most_common())})")

    unknown = {f"{name}: {label!r}" for name, (_, stats) in read.items() for label in stats.labels
               if class_map.decide(label).kind == "unknown"}
    if unknown:                     # before the slow label check, not after it
        print("\nThese labels are neither mapped nor ignored in configs/classes.yaml:\n  " + "\n  ".join(sorted(unknown)) +
              "\nAdd each one under `aliases` (keep it) or `ignore` (drop it), then run again.")
        return 1

    print(f"\n[3/4] Checking their labels on {device} (a missing label teaches the model the wrong thing)")
    weights = max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    ppe_model, coco_model = load_model(weights), load_model(args.coco_model)
    print(f"  current model: {weights.name}; people: {args.coco_model}")
    extras = []
    for name in args.sources:
        src = EXTRA_SOURCES[name]
        samples, stats = read[name]
        kept, audit = audit_source(src, samples, class_map, ppe_model=ppe_model, coco_model=coco_model,
                                   absent_helmet=absent_helmet, device=device)
        extras.append((src, kept, stats, audit))
        why = ", ".join(f"{n} {reason}" for reason, n in audit.dropped.most_common()) or "nothing"
        print(f"  {name}: kept {audit.kept} of {audit.checked}" +
              (f", {audit.pseudo_people} people added" if src.persons == "pseudo" else "") + f"; left out: {why}")
        by_name = {s.image.name: s.image for s in samples}
        sheet([s for s in kept if s.split == "train"], audit.dropped_images, by_name,
              REVIEW / Path(args.out).name / f"{name}.jpg", class_map)

    print("\n[4/4] Merging with datasets/ppe3")
    try:
        report = build_extended(BASE, extras, Path(args.out), class_map)
    except UnknownLabels as exc:
        print(f"\n{exc}")
        return 1
    sources = sorted({src for s in SPLITS for src in report.images[s]})
    print(f"\n{'split':6} " + " ".join(f"{s:>18}" for s in sources) + f" {'total':>7}")
    for s in SPLITS:
        row = report.images[s]
        print(f"{s:6} " + " ".join(f"{row.get(src, 0):18d}" for src in sources) + f" {sum(row.values()):7d}")
    print(f"\n{'split':6} " + " ".join(f"{c:>7}" for c in report.classes))
    for s in SPLITS:
        print(f"{s:6} " + " ".join(f"{report.boxes[s][c]:7d}" for c in report.classes))
    print(f"\nLeft out for looking like a ppe3 val/test image: {report.left_out_near_ppe3_test or 'none'}")
    print(f"Wrote {args.out} (details: {args.out}/report.md; pictures: runs/data_review/{Path(args.out).name}/) "
          f"in {(time.monotonic() - start) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
