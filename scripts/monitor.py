#!/usr/bin/env python3
"""Watch a camera or a video for people without a helmet or hi-vis vest, and people in restricted
zones (Phases 2 and 3).

    python scripts/monitor.py --camera cam1                 # a camera from configs/cameras.yaml
    python scripts/monitor.py --source my_clip.mp4           # a video file, at its own speed
    python scripts/monitor.py --source my_clip.mp4 --as-camera cam3 --start-time "2026-09-24 18:59:30"
    python scripts/monitor.py --source my_clip.mp4 --headless --save out.mp4

Each person gets a track ID and a coloured box: green = all fine, amber = a violation is being
confirmed, red = confirmed violation, grey = too small or cut off to judge. The camera's zones are
drawn too; a ring marks where each person stands. The rules (configs/rules.yaml) decide what counts:
a violation that lasts its rule's dwell time becomes ONE event. It is printed and appended to
events.jsonl, with a snapshot image, in runs/monitor/<camera>_<time>/. Phase 5 will send these as
alerts.

--as-camera applies a camera's zones and rules to a video file (e.g. a recording from that camera).
--start-time says what time the video's first frame was, so rules with active hours behave as
they would have then (default: now).

Press q in the window to stop.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT, ConfigError, get_camera
from ppe_monitor.device import best_device
from ppe_monitor.draw import draw, snapshot
from ppe_monitor.ingestion.latest import LatestFrame
from ppe_monitor.ingestion.rtsp import open_capture
from ppe_monitor.pipeline import PPE_CONFIG, PPEMonitor, Settings
from ppe_monitor.rules.engine import RULES_FILE, RuleSet
from ppe_monitor.rules.zones import ZONES_FILE
from ppe_monitor.vision.detector import MODELS_DIR, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", metavar="ID", help="camera id from configs/cameras.yaml")
    src.add_argument("--source", help="RTSP URL or video file")
    parser.add_argument("--weights", help="model file (default: newest in models/)")
    parser.add_argument("--config", default=str(PPE_CONFIG), help="rules / tracking / event settings")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true", help="no window")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--save", help="also write the annotated video to this file")
    parser.add_argument("--out", help="folder for events.jsonl and snapshots (default: runs/monitor/<camera>_<time>)")
    parser.add_argument("--as-camera", metavar="ID", help="with --source: use this camera's zones and rules")
    parser.add_argument("--rules", default=str(RULES_FILE))
    parser.add_argument("--zones", default=str(ZONES_FILE))
    parser.add_argument("--start-time", help='with --source: the time of the first frame, e.g. "2026-09-24 18:59:30"')
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights:
        print("No trained model in models/. Train one first (Phase 1): python scripts/train_detector.py")
        return 1
    settings = Settings.load(args.config)
    if settings.model and settings.model != weights.stem:
        print(f"Note: {args.config} was calibrated for {settings.model}, but the model is {weights.stem}. "
              "Re-check it with scripts/check_ppe_rules.py.")

    if args.camera:
        try:
            cam = get_camera(args.camera)
        except ConfigError as exc:
            print(exc)
            return 1
        source, transport, camera_id = cam.url, cam.transport, cam.id
    else:
        source, transport, camera_id = args.source, "tcp", args.as_camera or Path(args.source).stem[:40]
    live = source.lower().startswith(("rtsp://", "rtsps://"))
    try:
        rules = RuleSet.load(args.rules, args.zones).for_camera(camera_id)
    except ConfigError as exc:
        print(f"Rules or zones config problem: {exc}")
        return 1
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.fromisoformat(args.start_time)
        except ValueError:
            print(f'--start-time must look like "2026-09-24 18:59:30", not {args.start_time!r}')
            return 1

    cap = open_capture(source, transport=transport)
    if not cap.isOpened():
        print(f"Could not open {source}." + (" For a fake camera, start the network first: "
                                              "python scripts/fake_cameras.py --with-server" if args.camera else ""))
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    device = best_device() if args.device == "auto" else args.device
    monitor = PPEMonitor(load_model(weights), camera_id, fps, settings, imgsz=args.imgsz, device=device, rules=rules,
                         start_time=start_time)
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "runs" / "monitor" / f"{camera_id}_{datetime.now():%Y%m%d_%H%M%S}"
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    events_file = (out_dir / "events.jsonl").open("a", encoding="utf-8")

    reader = LatestFrame(cap) if live else None
    writer = None
    if not args.headless:
        cv2.namedWindow(camera_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print(f"Monitoring {source} as camera {camera_id} with {weights.name} on {device}. Events -> {out_dir}.")
    on_now = rules.active(monitor.start_time)
    for rule in rules.rules:
        zone = rules.zone(rule.zone)
        print(f"  rule {rule.id:22s} {rule.type:15s} {rule.severity:8s}" + (f" zone: {zone.name}" if zone else "")
              + (f"  hours: {rule.schedule.text}" if rule.schedule else "")
              + ("" if rule.id in on_now else "  (off at the start)"))
    print("Press q in the window to stop.")

    seq, frames, busy, skipped, n_events, start = 0, 0, 0.0, 0, 0, time.monotonic()
    try:
        while args.duration is None or time.monotonic() - start < args.duration:
            if reader:
                ok, frame, new_seq, arrived = reader.get(seq)
                skipped += max(0, new_seq - seq - 1)
                seq = new_seq
                t = arrived - start          # live: the time the frame arrived
            else:
                ok, frame = cap.read()
                t = frames / fps             # file: the video's own clock
            if not ok:
                break
            t0 = time.perf_counter()
            r = monitor.process(frame, t)
            busy += time.perf_counter() - t0
            frames += 1
            for ev in r.events:
                n_events += 1
                snap = out_dir / "snapshots" / f"{ev.event_id}.jpg"
                cv2.imwrite(str(snap), snapshot(frame, r, ev), [cv2.IMWRITE_JPEG_QUALITY, 90])
                ev.snapshot = str(snap.relative_to(out_dir))
                events_file.write(json.dumps({**ev.to_dict(), "source": source}) + "\n")
                events_file.flush()
                print(f"  EVENT {ev.kind:14s} {ev.severity:8s} track #{ev.track_id:<4d} t={ev.confirmed:7.1f}s "
                      f"(began {ev.started:.1f}s)" + (f"  zone {ev.zone}" if ev.zone else "") + f"  {snap.name}")
            footer = f"{1000 * busy / frames:.0f} ms/frame on {device}   people {len(r.tracks)}   events {n_events}" \
                     + (f"   skipped {skipped}" if live else "")
            if args.save or not args.headless:
                shown = draw(frame, r, footer)
                if args.save:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, shown.shape[1::-1])
                    writer.write(shown)
                if not args.headless:
                    cv2.imshow(camera_id, shown)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            reader.stop()
        cap.release()
        if writer:
            writer.release()
        events_file.close()
        if not args.headless:
            cv2.destroyAllWindows()

    if frames:
        print(f"\n{frames} frames, {n_events} events. Processing {1000 * busy / frames:.0f} ms/frame "
              f"(max ~{frames / busy:.0f} fps) on {device}." + (f" {skipped} stale frames skipped to stay live." if live else ""))
        print(f"Events: {out_dir / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
