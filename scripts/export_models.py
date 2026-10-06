#!/usr/bin/env python3
"""Export the detector and the keypoint model for the faster backends (Phase 4).

    python scripts/export_models.py                    # every backend this machine can use
    python scripts/export_models.py --backend coreml   # just one

Writes models/exported/<model>/<backend>/, which scripts/benchmark.py and scripts/run_cameras.py
load. Core ML exports need macOS to *run*, but can be made anywhere. INT8 ONNX is calibrated on
the validation split's images (never the test split, which measures it). Each export takes from a
few seconds to about a minute.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.pipeline import Settings
from ppe_monitor.vision.backends import SPECS, default_pose_weights, export, exported_path, is_macos
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", action="append", choices=[n for n, s in SPECS.items() if s.fmt != "pt"],
                        help="repeat for several (default: all that make sense on this machine)")
    parser.add_argument("--weights", help="detector (default: newest in models/)")
    parser.add_argument("--force", action="store_true", help="export again even if the files exist")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights:
        print("No trained model in models/.")
        return 1
    pose = PRETRAINED_DIR / Path(Settings.load().pose["model"]).name
    if not pose.is_file():
        load_model(pose.name)                     # downloads it into models/pretrained/
    backends = args.backend or [n for n, s in SPECS.items() if s.fmt != "pt" and (is_macos() or not s.macos_only)]
    print(f"Exporting {weights.name} and {pose.name} for: {', '.join(backends)}")
    failed = []
    for name in backends:
        for model, is_pose in ((weights, False), (pose, True)):
            target = exported_path(model, name)
            if target.exists() and not args.force:
                print(f"  {name:12s} {model.name:28s} already there: {target.relative_to(MODELS_DIR.parent)}")
                continue
            t = time.time()
            try:
                out = export(model, name, pose=is_pose)
                print(f"  {name:12s} {model.name:28s} -> {out.relative_to(MODELS_DIR.parent)} ({time.time() - t:.0f} s)")
            except Exception as exc:          # keep going: one failed format shouldn't stop the others
                failed.append(f"{name} / {model.name}: {exc}")
                print(f"  {name:12s} {model.name:28s} FAILED: {exc}")
    if failed:
        print("\nSome exports failed:\n  " + "\n  ".join(failed))
        return 1
    print("\nDone. Next: python scripts/benchmark.py")
    return 0


if __name__ == "__main__":
    code = main()
    # Exit without Python's teardown: after Core ML and ONNX Runtime have both run in one process,
    # macOS can abort while their threads are torn down ("recursive_mutex lock failed", 29 Sep 2026),
    # turning a finished export into a failed step. Everything is written by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
