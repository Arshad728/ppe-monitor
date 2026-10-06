#!/usr/bin/env python3
"""Create the synthetic clips that the simulated cameras in configs/cameras.yaml stream.

    python scripts/make_test_clips.py              # make any that are missing
    python scripts/make_test_clips.py --force      # remake all of them

Only clips whose file name starts with "synthetic_" are generated; your own clips are
never touched. See data/README.md for collecting real footage.
"""

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT, load_cameras
from ppe_monitor.sim.synthetic import make_synthetic_clip


def rel(path):
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cameras.yaml"))
    parser.add_argument("--force", action="store_true", help="regenerate clips that already exist")
    parser.add_argument("--seconds", type=float, default=20.0, help="clip length (default 20)")
    parser.add_argument("--fps", type=float, default=15.0, help="frame rate (default 15, typical for CCTV)")
    parser.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT (default 1280x720)")
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    cameras = [c for c in load_cameras(args.config) if c.sim_source and c.sim_source.name.startswith("synthetic_")]
    if not cameras:
        print("No camera in the config uses a synthetic_* clip; nothing to do.")
        return 0

    for seed, cam in enumerate(cameras):
        if cam.sim_source.exists() and not args.force:
            print(f"{cam.id}: {rel(cam.sim_source)} already exists")
            continue
        start = time.monotonic()
        info = make_synthetic_clip(cam.sim_source, label=cam.name.upper(), seconds=args.seconds,
                                   fps=args.fps, size=(width, height), seed=seed)
        codec = "H.264" if info.h264 else "MPEG-4 (ffmpeg not found or no H.264 encoder)"
        print(f"{cam.id}: wrote {rel(info.path)}  "
              f"{info.width}x{info.height} @ {info.fps:g} fps, {info.frames} frames, {codec}  "
              f"({time.monotonic() - start:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
