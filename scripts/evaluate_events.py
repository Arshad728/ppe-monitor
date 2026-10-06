#!/usr/bin/env python3
"""Run the full Phase 2 chain on the test clips and check the events it raises.

    python scripts/make_event_clips.py        # once: build the clips from test photos
    python scripts/evaluate_events.py         # detector -> tracker -> PPE rules -> events, scored

The goal (book, Chapter 19): every person really without a helmet or vest gets exactly one event,
and nobody wearing theirs gets any, despite frame-to-frame flicker and a passing occluder. The
report goes to runs/events/<model>/report.md, with the snapshot of every event and annotated videos
of the clips that went wrong.

It always scores both PPE rules. A rule switched off in configs/rules.yaml (a site where hi-vis
isn't required, say) is scored anyway with the default settings, so results stay comparable between
sites and models; the report says so.
"""

import argparse
import hashlib
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.device import best_device
from ppe_monitor.draw import draw, snapshot
from ppe_monitor.pipeline import PPE_CONFIG, REUSE_POSES, PPEMonitor, Settings
from ppe_monitor.rules.engine import RULES_FILE, Rule, RuleSet
from ppe_monitor.rules.scoring import ClipScore, ZoneScore, score_clip, score_zone
from ppe_monitor.rules.zones import CameraZones, Zone
from ppe_monitor.vision.detector import MODELS_DIR, load_model

CLIPS = PROJECT_ROOT / "datasets" / "event_clips"


def pct(a, b) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", default=str(CLIPS))
    parser.add_argument("--weights", help="model file (default: newest in models/)")
    parser.add_argument("--config", default=str(PPE_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--videos", type=int, default=6, help="save annotated videos of up to this many clips that went wrong")
    parser.add_argument("--stride", type=int, default=1,
                        help="process every Nth frame only, like a camera whose frames are skipped to keep up")
    parser.add_argument("--no-pose", action="store_true", help="don't use the keypoint model")
    parser.add_argument("--pose-every", type=int,
                        help="run the keypoint model on every Nth frame with people (default: pose.every in configs/ppe.yaml)")
    parser.add_argument("--out-suffix", default="", help=argparse.SUPPRESS)   # keep a comparison run's report apart
    parser.add_argument("--rules", default=str(RULES_FILE),
                        help="the PPE rules for every camera are taken from here (dwell, cooldown)")
    parser.add_argument("--zone-dwell", type=float, default=2.0, help="dwell of the zone rule on zone clips (seconds)")
    parser.add_argument("--reuse", action="store_true",
                        help="keep the detections and keypoints of every frame, and reuse them on the next --reuse "
                             "run: only the tracker, the rules and the events run again (for trying settings)")
    args = parser.parse_args()

    clips_dir = Path(args.clips)
    meta_file = clips_dir / "clips.json"
    if not meta_file.is_file():
        print(f"No clips in {clips_dir}. Make them first: python scripts/make_event_clips.py")
        return 1
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights:
        print("No trained model in models/.")
        return 1
    settings = Settings.load(args.config)
    device = best_device() if args.device == "auto" else args.device
    model = load_model(weights)
    pose_model = False if args.no_pose or not settings.pose["enabled"] else load_model(settings.pose["model"])
    if args.pose_every is not None:
        if args.pose_every < 1:
            parser.error("--pose-every must be 1 or more")
        settings.pose["every"] = args.pose_every
    tag = (f"{clips_dir.name}" + (f"_every{args.stride}" if args.stride > 1 else "") + ("_nopose" if not pose_model else "")
           + args.out_suffix)
    out = PROJECT_ROOT / "runs" / "events" / weights.stem / tag
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    for old in list((out / "snapshots").glob("*.jpg")) + list((out / "videos").glob("*.mp4")):
        old.unlink()

    cache_file = out.parent / f"{clips_dir.name}_detections.pkl"
    clips_id = hashlib.sha256(meta_file.read_bytes()).hexdigest()   # remade clips: start again
    cache = {}
    if args.reuse and cache_file.is_file():
        saved = pickle.loads(cache_file.read_bytes())
        if saved.get("clips_id") == clips_id and saved.get("weights") == weights.name:
            cache = saved["frames"]
            print(f"Reusing detections from {cache_file.relative_to(PROJECT_ROOT)}")
    reused = 0

    base_rules = [r for r in RuleSet.load(args.rules).rules if r.cameras is None]   # rules for every camera
    switched_off = [t for t in ("no_helmet", "no_vest") if not any(r.type == t for r in base_rules)]
    base_rules += [Rule(t, t) for t in switched_off]      # a benchmark of the model, not of this site's policy
    if switched_off:
        print(f"Note: {', '.join(switched_off)} is off in {Path(args.rules).name}; scored here anyway, with default settings.")
    total, ztotal = ClipScore(), ZoneScore()
    per_clip, frames_done, busy, videos_saved = [], 0, 0.0, 0
    print(f"Running {len(meta['clips'])} clips through {weights.name} on {device} ...")
    for truth in meta["clips"]:
        cap = cv2.VideoCapture(str(clips_dir / truth["clip"]))
        fps = truth["fps"]
        camera = truth["clip"].rsplit(".", 1)[0]
        rules = RuleSet(list(base_rules))
        if "zone" in truth:     # zone clips: plus "nobody in the test zone"
            z = truth["zone"]
            rules.rules.append(Rule("zone", "zone_intrusion", (camera,), z["id"], "high", dwell=args.zone_dwell))
            rules.zones[camera] = CameraZones(camera, [Zone(z["id"], z["name"], tuple(map(tuple, z["polygon"])))])
        monitor = PPEMonitor(model, camera, fps, settings, imgsz=args.imgsz, device=device, pose_model=pose_model,
                             rules=rules)
        results, tracks_per_frame, events = [], {}, []
        k, processed = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if k % args.stride:
                k += 1
                continue
            t0 = time.perf_counter()
            if args.reuse:
                saved = cache.setdefault(truth["clip"], {}).get(k)
                if saved is None or (pose_model and saved[1] is None):
                    dets = monitor.detect(frame) if saved is None else saved[0]
                    poses = monitor.detect_poses(frame) if pose_model and any(d.cls == 0 for d in dets) else None
                    saved = cache[truth["clip"]][k] = (dets, [] if pose_model and poses is None else poses)
                else:
                    reused += 1
                poses = None
                if pose_model:          # the keypoint model's turn (pose.every), or carry the last keypoints
                    people = any(d.cls == 0 and d.score >= settings.tracking.low for d in saved[0])
                    poses = REUSE_POSES if people and not monitor.pose_due() else saved[1]
                r = monitor.process(frame, k / fps, detections=saved[0], poses=poses, frame_index=k)
            else:
                r = monitor.process(frame, k / fps, frame_index=k)
            busy += time.perf_counter() - t0
            processed += 1
            tracks_per_frame[k] = {tr.track_id: tr.box for tr in r.tracks}
            for ev in r.events:
                events.append(ev)
                cv2.imwrite(str(out / "snapshots" / f"{ev.event_id}.jpg"), snapshot(frame, r, ev),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            results.append(r)
            k += 1
        cap.release()
        frames_done += processed
        truth["frames"] = min(truth["frames"], k)
        sc = score_clip(truth, events, tracks_per_frame)
        total.add(sc)
        zs = score_zone(truth, events, tracks_per_frame, args.zone_dwell)
        ztotal.add(zs)
        sc.details += zs.details
        total.details += zs.details
        went_wrong = bool(sc.details) or sc.id_switches
        per_clip.append({"clip": truth["clip"], "image": truth["image"], "kind": truth["kind"],
                         "events": [e.to_dict() for e in events], "id_switches": sc.id_switches, "notes": sc.details})
        flag = "  <- " + "; ".join(n.split(": ", 1)[1] for n in sc.details) if sc.details else ""
        print(f"  {truth['clip']:22s} events {len(events):2d}  id switches {sc.id_switches}{flag}")
        if went_wrong and videos_saved < args.videos:  # read the clip again and draw what the chain saw
            cap = cv2.VideoCapture(str(clips_dir / truth["clip"]))
            vw = None
            by_frame = {r.frame_index: r for r in results}
            last = None
            for k2 in range(truth["frames"]):
                ok, frame = cap.read()
                if not ok:
                    break
                r = by_frame.get(k2, last)
                last = r
                if r is None:
                    continue
                if vw is None:
                    h, w = frame.shape[:2]
                    vw = cv2.VideoWriter(str(out / "videos" / truth["clip"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                vw.write(draw(frame, r, f"{truth['clip']}  t={r.t:4.1f}s"))
            cap.release()
            if vw is not None:
                vw.release()
            videos_saved += 1

    items = ("helmet", "vest")
    rows = []
    for item in items:
        bare = total.caught_once[item] + total.caught_more[item] + total.missed[item]
        rows.append((item, bare, total.caught_once[item], total.caught_more[item], total.missed[item],
                     total.compliant_people[item], total.false_events[item]))
    delays = sorted(total.delays)
    motion = meta.get("motion", "drift")
    lines = [f"# Event check: {weights.stem}, {clips_dir.name}", "",
             f"{len(meta['clips'])} clips of {meta['seconds']:.0f} s at {meta['fps']:.0f} fps, made from ppe3 test photos "
             f"(scripts/make_event_clips.py, motion: {motion}), "
             + (f"1 frame in {args.stride} processed ({meta['fps'] / args.stride:.0f} fps). " if args.stride > 1 else "all frames processed. ")
             + ("Keypoints: off. " if not pose_model else "Keypoints: on, every frame with people. "
                if settings.pose["every"] == 1 else
                f"Keypoints: on, on 1 frame in {settings.pose['every']} with people (the frames between carry "
                "each person's last keypoints). ")
             + f"Settings: configs/ppe.yaml and {Path(args.rules).name} "
             f"(PPE dwell {next((r.dwell for r in base_rules if r.dwell is not None), settings.events.dwell):.1f} s, "
             f"vote window {settings.events.window:.1f} s). "
             + (f"{' and '.join(switched_off)} is off in {Path(args.rules).name} and scored here with default "
                f"settings. " if switched_off else "")
             + (f"Detections reused from an earlier run for {reused} of {frames_done} frames." if reused else
                f"Device {device}, {1000 * busy / max(1, frames_done):.0f} ms per processed frame for the whole chain."),
             "",
             "| | helmet | vest |", "|---|---|---|"]
    lines.append("| **People without it** (each should get exactly one event) | " + " | ".join(str(r[1]) for r in rows) + " |")
    lines.append("| ... exactly one event | " + " | ".join(pct(r[2], r[1]) for r in rows) + " |")
    lines.append("| ... more than one event | " + " | ".join(pct(r[3], r[1]) for r in rows) + " |")
    lines.append("| ... no event (missed) | " + " | ".join(pct(r[4], r[1]) for r in rows) + " |")
    lines.append("| **People wearing it** (should get none) | " + " | ".join(str(r[5]) for r in rows) + " |")
    lines.append("| ... false events | " + " | ".join(str(r[6]) for r in rows) + " |")
    lines += ["",
              f"Events on people whose status isn't labelled (can't be scored): {total.unscored_events}.",
              f"Time from the start of a clip to the event: median {delays[len(delays) // 2]:.1f} s, "
              f"90% within {delays[int(len(delays) * 0.9)]:.1f} s." if delays else "No events were caught.",
              f"Tracking: {total.people_tracked} people; {total.id_switches} ID switches; each person's track found in "
              f"{100 * statistics.mean(total.track_coverage):.0f}% of the frames where they were visible (median "
              f"{100 * statistics.median(total.track_coverage):.0f}%)." if total.track_coverage else ""]
    if any("zone" in c for c in meta["clips"]):
        zd = sorted(ztotal.delays)
        lines += ["", f"## Restricted zone (rule: nobody inside for {args.zone_dwell:g} s)", "",
                  "| | people |", "|---|---|",
                  f"| **Walked into the zone and stayed** (each should get exactly one event) | {ztotal.should_alert} |",
                  f"| ... exactly one event | {pct(ztotal.caught_once, ztotal.should_alert)} |",
                  f"| ... more than one event | {pct(ztotal.caught_more, ztotal.should_alert)} |",
                  f"| ... no event (missed) | {pct(ztotal.missed, ztotal.should_alert)} |",
                  f"| **Never inside** (should get none) | {ztotal.never_inside} |",
                  f"| ... false events | {ztotal.false_events} |", "",
                  f"Not scored: {ztotal.unscored_people} people (entered too late in the clip to be confirmed, or feet "
                  f"out of view), {ztotal.unscored_events} zone events on people nobody labelled.",
                  (f"Time from the feet entering the zone to the event: median {zd[len(zd) // 2]:.1f} s, "
                   f"90% within {zd[int(len(zd) * 0.9)]:.1f} s." if zd else "")]
    lines += ["", "## What went wrong", ""] + [f"- {d}" for d in total.details] + ["",
              f"Snapshots of every event: `{(out / 'snapshots').relative_to(PROJECT_ROOT)}`. Annotated videos of up to "
              f"{args.videos} clips that went wrong: `{(out / 'videos').relative_to(PROJECT_ROOT)}`."]
    report = "\n".join(lines) + "\n"
    if args.reuse:
        cache_file.write_bytes(pickle.dumps({"clips_id": clips_id, "weights": weights.name, "frames": cache}))
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"clips": per_clip, "summary": {
        "rows": rows, "unscored_events": total.unscored_events, "id_switches": total.id_switches,
        "zone": {k: getattr(ztotal, k) for k in ("should_alert", "caught_once", "caught_more", "missed",
                                                 "never_inside", "false_events", "unscored_people", "unscored_events",
                                                 "delays")},
        "people_tracked": total.people_tracked, "delays": delays, "ms_per_frame": 1000 * busy / max(1, frames_done)}},
        indent=1), encoding="utf-8")
    print("\n" + report)
    print(f"Report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
