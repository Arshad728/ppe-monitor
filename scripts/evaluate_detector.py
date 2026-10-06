#!/usr/bin/env python3
"""Measure a trained detector on any labelled YOLO dataset split.

    python scripts/evaluate_detector.py                               # newest model, ppe3 test split
    python scripts/evaluate_detector.py --data datasets/heldout/data.yaml --split test
    python scripts/evaluate_detector.py --threshold 0.3               # override the working threshold

Prints per-class AP@50, precision and recall at the working threshold, and recall by object
size, and saves the same table to runs/eval/.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.device import best_device
from ppe_monitor.vision.detector import MODELS_DIR, load_model
from ppe_monitor.vision.evaluate import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="model file (default: newest in models/)")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, help="default: the model card's (best F1 on ppe3 val)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights or not weights.is_file():
        print("No trained model found. Train one first: python scripts/train_detector.py")
        return 1
    threshold = args.threshold
    if threshold is None and weights.with_suffix(".json").is_file():
        threshold = json.loads(weights.with_suffix(".json").read_text())["val"]["threshold"]
    device = best_device() if args.device == "auto" else args.device

    result = evaluate(load_model(weights), args.data, args.split, threshold=threshold, imgsz=args.imgsz, device=device)
    text = (f"# {weights.name} on {args.data} ({args.split}, {result.images} images)\n\n"
            f"{datetime.now():%Y-%m-%d %H:%M}, device {device}, threshold {result.threshold:.3f}\n\n{result.table()}\n"
            f"Ultralytics cross-check: mAP@50 {result.map50_ultralytics:.3f}, mAP@50-95 {result.map50_95_ultralytics:.3f} "
            "(its validator lets a box carry several classes; the table scores one class per box, as the live "
            "detector outputs, so the two differ a little)\n")
    print("\n" + text)
    out = PROJECT_ROOT / "runs" / "eval" / f"{weights.stem}__{Path(args.data).parent.name}_{args.split}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Saved {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
