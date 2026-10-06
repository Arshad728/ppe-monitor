#!/usr/bin/env python3
"""Download the public PPE datasets and merge them into one training dataset.

    python scripts/prepare_data.py

Result: datasets/ppe3/ (YOLO format; classes person, helmet, vest), with report.md explaining
every decision: how each source label was treated, which near-duplicate images were moved out of
val/test, and how many boxes of each class and size there are. Downloads are cached in
datasets/raw/ and checked against pinned SHA-256 checksums.
"""

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.data.build import SPLITS, UnknownLabels, build_dataset
from ppe_monitor.data.labels import ClassMap
from ppe_monitor.data.sources import RAW_DIR, SOURCES, SourceError, fetch

DEFAULT_OUT = PROJECT_ROOT / "datasets" / "ppe3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
                        help="which datasets to use (default: all)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (default datasets/ppe3)")
    parser.add_argument("--max-distance", type=int, default=6,
                        help="fingerprint bits that may differ for two images to count as near-duplicates (default 6)")
    args = parser.parse_args()

    start = time.monotonic()
    print("[1/2] Getting the source datasets")
    try:
        roots = [(SOURCES[name], fetch(SOURCES[name], RAW_DIR)) for name in args.sources]
    except (SourceError, OSError) as exc:
        print(f"\nCould not get the datasets: {exc}")
        return 1

    print("\n[2/2] Merging into one dataset")
    from pathlib import Path
    try:
        report = build_dataset(roots, Path(args.out), ClassMap.from_yaml(), max_distance=args.max_distance)
    except UnknownLabels as exc:
        print(f"\n{exc}")
        return 1

    print("\nHow each label was treated:")
    for src, decisions in report.label_decisions.items():
        kept = ", ".join(f"{lbl} -> {d['class']}" for lbl, d in decisions.items() if d["decision"] == "class")
        dropped = ", ".join(lbl for lbl, d in decisions.items() if d["decision"] == "ignore")
        print(f"  {src}\n    kept:    {kept}\n    dropped: {dropped}")
    moved = sum(report.moved_to_train.values())
    print(f"\nNear-duplicates: {report.near_duplicate_pairs} pairs; {moved} val/test images moved into train "
          f"({report.moved_to_train or 'none'}); {report.exact_duplicates_dropped} exact duplicates dropped.")
    print(f"\n{'split':6} {'images':>7} " + " ".join(f"{c:>7}" for c in report.classes))
    for s in SPLITS:
        print(f"{s:6} {report.images[s]:7d} " + " ".join(f"{report.boxes[s][c]:7d}" for c in report.classes))
    print(f"\nWrote {args.out}  (details: {args.out}/report.md)  in {time.monotonic() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
