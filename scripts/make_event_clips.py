#!/usr/bin/env python3
"""Make the Phase 2 test clips: 10-second videos from test photos whose people have a known
helmet / vest status (see src/ppe_monitor/sim/photo_clips.py for how and why).

    python scripts/make_event_clips.py                 # -> datasets/event_clips/*.mp4 + clips.json
    python scripts/make_event_clips.py --motion sway   # -> datasets/event_clips_sway/: people moving
    python scripts/make_event_clips.py --motion pan    # -> datasets/event_clips_zone/: walking into a zone

Uses only the ppe3 TEST split, which the detector never trained on and the Phase 2 settings were
not tuned on. Two kinds of photo are used:
  - "violation" photos: at least one fully visible person known to be without a helmet or vest;
  - "compliant" photos: every person with a known status wears both.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.rules.check import load_cases
from ppe_monitor.sim.photo_clips import make_clip

DATA = PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"
OUT = PROJECT_ROOT / "datasets" / "event_clips"


def fully_visible(t) -> bool:
    return t.box[1] > 0.01 and t.box[3] < 0.99 and t.box[3] - t.box[1] > 0.2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-kind", type=int, default=43, help="clips of each kind (violation / compliant)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--motion", choices=["drift", "sway", "pan"], default="drift",
                        help="drift: slow pan and zoom; sway: people move side to side at walking speed; "
                             "pan: people move steadily into a restricted zone (Phase 3)")
    parser.add_argument("--out", help="folder (default datasets/event_clips, _sway or _zone)")
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else OUT.with_name(
        {"drift": "event_clips", "sway": "event_clips_sway", "pan": "event_clips_zone"}[args.motion])
    if not DATA.is_file():
        print("No ppe3 dataset. Run Phase 1's data step first: python scripts/prepare_data.py")
        return 1

    print("Reading the test split's labels ...")
    cases = load_cases(DATA, "test")
    violation = [c for c in cases if any(fully_visible(t) and (t.helmet is False or t.vest is False) for t in c.truths)]
    compliant = [c for c in cases if c not in violation and all(t.helmet and t.vest for t in c.truths)]
    rng = random.Random(0)
    rng.shuffle(violation)
    rng.shuffle(compliant)
    chosen = [("violation", c) for c in violation[:args.per_kind]] + [("compliant", c) for c in compliant[:args.per_kind]]

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.mp4"):
        old.unlink()
    clips = []
    for n, (kind, case) in enumerate(chosen):
        out = out_dir / f"{n:03d}_{kind}.mp4"
        truth = make_clip(case, out, seconds=args.seconds, fps=args.fps, seed=n, motion=args.motion)
        truth["kind"] = kind
        clips.append(truth)
        print(f"  {out.name}  {len(truth['people'])} people with a known status  ({case.image.name[:48]})")
    (out_dir / "clips.json").write_text(json.dumps({"seconds": args.seconds, "fps": args.fps, "motion": args.motion,
                                                    "clips": clips}), encoding="utf-8")
    print(f"\n{len(clips)} clips ({sum(k == 'violation' for k, _ in chosen)} with violations) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
