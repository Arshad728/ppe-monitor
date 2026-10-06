#!/usr/bin/env python3
"""Check the PPE rules against people whose helmet / vest status is known from the labels.

    python scripts/check_ppe_rules.py                  # test split, settings from configs/ppe.yaml
    python scripts/check_ppe_rules.py --split val --sweep   # try other helmet / vest thresholds

For every labelled person whose status is known (a helmet or a bare-head label on them, a vest or
a no-vest label), the rule decides "worn / missing / unknown" twice:
  - from the labelled boxes, which tests the geometry alone;
  - from the detector's boxes in that one image, which is what a single video frame would give.
The second gives the per-frame error rates the event logic has to smooth over:
  false alarm   wearing it, but judged missing  (pushes towards a false alert)
  missed        not wearing it, but judged worn (hides a violation in that frame)
Tune on --split val; report on --split test.
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.device import best_device
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.check import (POSTURES, SIZE_BUCKETS, attach_poses, check_detections, check_geometry,
                                     load_cases, summary)
from ppe_monitor.vision.detector import MODELS_DIR, load_model, predict

DATA = PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"


def table(s: dict) -> list[str]:
    def pct(v: float) -> str:
        return f"{v:.1%}" if v == v else "-"   # nan (nobody to judge) -> "-"

    out = ["| | helmet | vest |", "|---|---|---|",
           "| People wearing it / not wearing it | " + " | ".join(
               f"{s[i]['people_wearing']} / {s[i]['people_bare']}" for i in ("helmet", "vest")) + " |"]
    for label, key in (("**False alarm**: wearing it, judged missing", "false_alarm"),
                       ("**Missed**: not wearing it, judged worn", "missed"),
                       ("Unknown (too small / cut off), people wearing it", "unknown_wearing"),
                       ("Unknown, people not wearing it", "unknown_bare")):
        out.append(f"| {label} | " + " | ".join(pct(s[i][key]) for i in ("helmet", "vest")) + " |")
    return out


def by_size(s: dict) -> list[str]:
    out = ["| Person height at the detector's input | helmet false alarm | helmet missed | vest false alarm | vest missed |",
           "|---|---|---|---|---|"]
    for _, _, name in SIZE_BUCKETS:
        cells = []
        for item in ("helmet", "vest"):
            v = s[item]["by_size"][name]
            cells.append(f"{v['false_alarm']:.1%} (of {v['judged_wearing']})" if v["judged_wearing"] else "-")
            cells.append(f"{v['missed']:.1%} (of {v['judged_bare']})" if v["judged_bare"] else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def by_posture(with_kp: dict, without_kp: dict) -> list[str]:
    """Helmet answers by posture, with and without the keypoint model."""
    def cell(v: dict, key: str, judged: str) -> str:
        return f"{v[key]:.1%}" if v[judged] else "-"

    def unk(v: dict, key: str) -> str:
        return f"{v[key]:.0%}" if v[key] == v[key] else "-"

    out = ["| Posture (from keypoints) | people wearing / not | false alarm, box rule | false alarm, keypoints | "
           "missed, box rule | missed, keypoints | unknown (wearing / not), keypoints |",
           "|---|---|---|---|---|---|---|"]
    for name in POSTURES:
        a, b = without_kp["helmet"]["by_posture"][name], with_kp["helmet"]["by_posture"][name]
        if not b["people_wearing"] + b["people_bare"]:
            continue
        out.append(f"| {name} | {b['people_wearing']} / {b['people_bare']} | {cell(a, 'false_alarm', 'judged_wearing')} | "
                   f"{cell(b, 'false_alarm', 'judged_wearing')} | {cell(a, 'missed', 'judged_bare')} | "
                   f"{cell(b, 'missed', 'judged_bare')} | {unk(b, 'unknown_wearing')} / {unk(b, 'unknown_bare')} |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", help="model file (default: newest in models/)")
    parser.add_argument("--config", default=str(PPE_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sweep", action="store_true", help="also try other helmet / vest thresholds")
    parser.add_argument("--no-pose", action="store_true", help="don't use the keypoint model")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights or not DATA.is_file():
        print("Needs a trained model (models/) and the ppe3 dataset (Phase 1).")
        return 1
    settings = Settings.load(args.config)
    device = best_device() if args.device == "auto" else args.device

    print(f"Reading the {args.split} split's labels ...")
    cases = load_cases(DATA, args.split)
    people = sum(len(c.truths) for c in cases)
    print(f"  {people} people with a known helmet or vest status in {len(cases)} images")
    print(f"Detecting with {weights.name} on {device} ...")
    detections = predict(load_model(weights), [c.image for c in cases], conf=0.01, imgsz=args.imgsz, device=device)
    names = ["person", "helmet", "vest"]
    use_pose = settings.pose["enabled"] and not args.no_pose
    if use_pose:
        print(f"Finding body keypoints with {Path(settings.pose['model']).name} ...")
        attach_poses(cases, load_model(settings.pose["model"]), imgsz=args.imgsz, device=device,
                     keypoint_conf=settings.pose["keypoint_conf"])

    geo = summary(check_geometry(cases, settings.rules))
    det = summary(check_detections(cases, detections, settings.rules, settings.thresholds, names))
    det_box = summary(check_detections(cases, detections, settings.rules, settings.thresholds, names, use_poses=False))
    lines = [f"# PPE rules check: {weights.stem}, {args.split} split", "",
             f"{people} people with a known status in {len(cases)} images. Settings: {Path(args.config).name} "
             f"(thresholds {settings.thresholds}).", "",
             "## The geometry alone (labelled boxes)", ""] + table(geo) + [
             "", "## One frame of the detector (what the event logic has to smooth over)", ""] + table(det) + [
             "", "People the detector didn't find at all aren't in these rates: helmet "
             f"{det['helmet']['people_wearing'] + det['helmet']['people_bare'] - det['helmet']['found_wearing'] - det['helmet']['found_bare']}"
             f", vest {det['vest']['people_wearing'] + det['vest']['people_bare'] - det['vest']['found_wearing'] - det['vest']['found_bare']}"
             " people.", "", "### By person size", ""] + by_size(det)
    if use_pose:
        lines += ["", "### Helmet, by posture", "",
                  "Posture comes from the keypoints matched to each labelled person. \"Box rule\" is the rule without "
                  "keypoints (head assumed at the top of the box); \"keypoints\" finds the head wherever it is, and "
                  "treats a bent, crouching or lying person whose head is out of sight as unknown rather than missing.", ""
                  ] + by_posture(det, det_box)

    if args.sweep:
        lines += ["", "## Other thresholds", "", "| helmet | vest | helmet false alarm | helmet missed | vest false alarm | vest missed |",
                  "|---|---|---|---|---|---|"]
        for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
            th = {**settings.thresholds, "helmet": t, "vest": t}
            s = summary(check_detections(cases, detections, settings.rules, th, names))
            lines.append(f"| {t:.2f} | {t:.2f} | {s['helmet']['false_alarm']:.1%} | {s['helmet']['missed']:.1%} | "
                         f"{s['vest']['false_alarm']:.1%} | {s['vest']['missed']:.1%} |")

    text = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / "runs" / "ppe_rules" / f"{weights.stem}_{args.split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"geometry": geo, "detector": det, "detector_box_rule": det_box}, indent=1, default=str), encoding="utf-8")
    print("\n" + text)
    print(f"Saved {out.with_suffix('.md').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
