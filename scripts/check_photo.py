#!/usr/bin/env python3
"""Check photos: who is wearing a helmet (and a hi-vis vest), as the live system would judge them.

    python scripts/check_photo.py photo.jpg                   # -> photo_checked.jpg next to it
    python scripts/check_photo.py site/*.jpg --out runs/photos # several photos, results in one folder
    bash scripts/ppe.sh photo photo.jpg                       # the same, on the Mac

The model in use (the newest .pt in models/) finds the people, helmets and vests; the PPE rule of
configs/ppe.yaml decides, for each person, whether the helmet and the vest are worn (the same rule,
thresholds and body keypoints as the cameras use). One photo is one frame: the live system also
needs a violation to last its dwell time (configs/rules.yaml) before it alerts, so a photo shows
what a single frame would say.

Colours: red = no helmet, green = helmet worn, grey = can't judge (too small, cut off, or head
out of sight). The label also gives the vest ("vest ok / NO / ?"), whether or not the site's vest
rule is on. Each person's verdict is printed too.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.pipeline import PPEMonitor, Settings
from ppe_monitor.rules.ppe import MISSING, UNKNOWN, WORN
from ppe_monitor.vision.detector import MODELS_DIR, load_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORD = {WORN: "worn", MISSING: "MISSING", UNKNOWN: "can't judge"}
MARK = {WORN: "ok", MISSING: "NO", UNKNOWN: "?"}
RED, GREEN, GREY = (40, 40, 230), (60, 190, 60), (160, 160, 160)


def check(monitor: PPEMonitor, path: Path, out: Path) -> list[str]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"can't read {path} as an image")
    monitor.tracker = type(monitor.tracker)(1.0, monitor.s.tracking)   # each photo on its own:
    monitor._since_pose, monitor._pose_memory = None, {}                 # no people or keypoints carried over
    r = monitor.process(frame, 0.0)
    img = frame.copy()
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 500))
    lines = []
    for n, tr in enumerate(sorted(r.tracks, key=lambda t: t.box[0]), 1):
        v = r.verdicts[tr.track_id]
        colour = RED if v.helmet == MISSING else GREEN if v.helmet == WORN else GREY
        x1, y1, x2, y2 = (int(tr.box[0] * w), int(tr.box[1] * h), int(tr.box[2] * w), int(tr.box[3] * h))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
        label = f"{n}: helmet {MARK[v.helmet]}  vest {MARK[v.vest]}"
        scale = max(0.5, max(h, w) / 1600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(th + 6, y1)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), colour, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        posture = f", {v.posture}" if v.posture and v.posture != "upright" else ""
        lines.append(f"  person {n}: helmet {WORD[v.helmet]}, vest {WORD[v.vest]}{posture}")
    cv2.imwrite(str(out), img)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="+", help="photos, or folders of photos")
    parser.add_argument("--out", help="folder for the checked photos (default: next to each photo)")
    parser.add_argument("--weights", help="model file (default: the model in use, the newest .pt in models/)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    paths = []
    for p in map(Path, args.photos):
        if p.is_dir():
            paths += sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES and "_checked" not in x.stem)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Not found: {p}")
            return 1
    if not paths:
        print("No photos found.")
        return 1
    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    settings = Settings.load()
    device = None if args.device == "auto" else args.device
    monitor = PPEMonitor(load_model(weights), "photo", 1.0, settings, device=device)
    monitor.pose_every = 1                                               # keypoints on every photo
    print(f"Model: {weights.stem}. Red = no helmet, green = helmet worn, grey = can't judge.")
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        out = (out_dir or path.parent) / f"{path.stem}_checked.jpg"
        try:
            lines = check(monitor, path, out)
        except ValueError as exc:
            print(f"{path.name}: {exc}")
            continue
        print(f"{path.name}: {len(lines)} person(s) -> {out}")
        print("\n".join(lines) if lines else "  nobody found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
