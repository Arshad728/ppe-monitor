#!/usr/bin/env python3
"""Check the rules and zones files, and show what applies to each camera (Phase 3).

    python scripts/check_rules.py                 # configs/rules.yaml + configs/zones.yaml
    python scripts/check_rules.py --at "2026-09-24 20:30"   # which rules would be on at that time

Prints, per camera, every rule that applies to it (type, zone, severity, dwell, cooldown, hours)
and whether it is on right now, and saves a picture of each camera's zones on the frame they were
drawn on to runs/zones/<camera>.jpg. Problems stop with a message saying what to fix; things that
are allowed but probably unintended (a zone no rule uses, a rule for a camera that isn't in
configs/cameras.yaml) are listed as warnings. Exit code 0 = no problems.
"""

import argparse
import sys
from datetime import datetime

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT, ConfigError, load_cameras
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.engine import RULES_FILE, RuleSet
from ppe_monitor.rules.zone_editor import ZoneEditor
from ppe_monitor.rules.zones import ZONES_FILE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", default=str(RULES_FILE))
    parser.add_argument("--zones", default=str(ZONES_FILE))
    parser.add_argument("--at", help='check active hours at this time instead of now, e.g. "2026-09-24 20:30"')
    args = parser.parse_args()

    try:
        rs = RuleSet.load(args.rules, args.zones)
        defaults = Settings.load(PPE_CONFIG).events
    except (ConfigError, ValueError) as exc:
        print(f"PROBLEM: {exc}")
        return 1
    try:
        now = datetime.fromisoformat(args.at).astimezone() if args.at else datetime.now().astimezone()
    except ValueError:
        print(f'--at must look like "2026-09-24 20:30", not {args.at!r}')
        return 1
    try:
        cameras = {c.id: c for c in load_cameras()}
    except ConfigError as exc:
        print(f"Note: {exc}")
        cameras = {}

    warnings = []
    named = {c for r in rs.rules if r.cameras for c in r.cameras}
    for cam in sorted(named | set(rs.zones)):
        if cameras and cam not in cameras:
            warnings.append(f"camera '{cam}' has rules or zones but isn't in configs/cameras.yaml")
    for cam, cz in rs.zones.items():
        for z in cz.zones:
            if not any(r.zone == z.id and r.applies_to(cam) for r in rs.rules):
                warnings.append(f"zone '{z.id}' of camera {cam} isn't used by any rule")

    print(f"Rules: {rs.source} ({len(rs.rules)} rules). Zones: {args.zones}. "
          f"Time: {now:%Y-%m-%d %H:%M} (UTC{now:%z})" + (f" (rules use {rs.timezone})" if rs.timezone else "") + "\n")
    lines = ["| Camera | Rule | Type | Zone | Severity | Dwell | Cooldown | Active hours | On now |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cam in sorted(set(cameras) | named | set(rs.zones)) or ["(any camera)"]:
        cr = rs.for_camera(cam)
        on = cr.active(now)
        for r in cr.rules:
            zone = cr.zone(r.zone)
            lines.append(f"| {cam} | {r.id} | {r.type} | {zone.name if zone else 'anywhere'} | {r.severity} | "
                         f"{r.dwell if r.dwell is not None else defaults.dwell:g} s | "
                         f"{r.cooldown if r.cooldown is not None else defaults.cooldown:g} s | "
                         f"{r.schedule.text if r.schedule else 'always'} | {'yes' if r.id in on else 'no'} |")
    print("\n".join(lines))

    out_dir = PROJECT_ROOT / "runs" / "zones"
    out_dir.mkdir(parents=True, exist_ok=True)
    pictures = []
    for cam, cz in sorted(rs.zones.items()):
        ref = PROJECT_ROOT / cz.drawn_on if cz.drawn_on else None
        frame = cv2.imread(str(ref)) if ref and ref.is_file() else None
        if frame is None:
            warnings.append(f"camera {cam}: the frame its zones were drawn on ({cz.drawn_on}) is missing")
            continue
        h, w = frame.shape[:2]
        cv2.imwrite(str(out_dir / f"{cam}.jpg"), ZoneEditor(cam, w, h, list(cz.zones)).render(frame, show_help=False))
        pictures.append(cam)
    if pictures:
        print(f"\nZone pictures: {out_dir} ({', '.join(pictures)})")
    for w in warnings:
        print(f"Warning: {w}")
    print("\nNo problems found." if not warnings else f"\nNo problems; {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
