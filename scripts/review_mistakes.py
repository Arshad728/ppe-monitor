#!/usr/bin/env python3
"""Build a browsable gallery of the detector's mistakes.

    python scripts/review_mistakes.py                          # newest model, test split
    python scripts/review_mistakes.py --split val
    python scripts/review_mistakes.py --weights models/x.pt --data datasets/heldout/data.yaml

Writes runs/review/<model>_<data>_<split>/index.html: each error type (missed helmet, false vest, ...)
with annotated examples, and why each object was missed. The working threshold is read from the model card
(models/<model>.json) unless --threshold is given.
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.device import best_device
from ppe_monitor.vision.detector import MODELS_DIR, load_model
from ppe_monitor.vision.review import MISS_CAUSES, review


def newest_model() -> Path | None:
    models = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="model file (default: the newest in models/)")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, help="confidence threshold (default: from the model card)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else newest_model()
    if not weights or not weights.is_file():
        print("No trained model found. Train one first: python scripts/train_detector.py")
        return 1
    threshold = args.threshold
    card = weights.with_suffix(".json")
    if threshold is None and card.is_file():
        threshold = json.loads(card.read_text())["val"]["threshold"]
    if threshold is None:
        threshold = 0.25
        print("No model card with a threshold; using 0.25")

    out = PROJECT_ROOT / "runs" / "review" / f"{weights.stem}_{Path(args.data).parent.name}_{args.split}"
    device = best_device() if args.device == "auto" else args.device
    print(f"Reviewing {weights.name} on {args.split} (threshold {threshold:.3f}, {device}) ...")
    summary = review(load_model(weights), args.data, args.split, threshold, out, imgsz=args.imgsz, device=device)
    print(f"\nErrors on {summary['images']} images:")
    for kind, n in sorted(summary["counts"].items(), key=lambda kv: -kv[1]):
        sizes = summary["missed_by_size"].get(kind)
        print(f"  {kind:32s} {n:5d}" + (f"   (small {sizes.get('small', 0)}, medium {sizes.get('medium', 0)}, "
                                        f"large {sizes.get('large', 0)})" if sizes else ""))
    print("\nWhy objects were missed:")
    for kind, causes in sorted(summary["missed_by_cause"].items()):
        print(f"  {kind:16s} " + ", ".join(f"{c} {causes.get(c, 0)}" for c in MISS_CAUSES))
    print(f"  confident false alarms (worth checking for missing labels): {summary['confident_false_alarms']}")
    print(f"\nGallery: {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
