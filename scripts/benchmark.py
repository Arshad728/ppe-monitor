#!/usr/bin/env python3
"""Speed against accuracy, for every backend and 1, 2, 4 and 8 cameras (Phase 4, the book's "done when").

    python scripts/export_models.py        # once: make the Core ML / ONNX versions of the models
    python scripts/benchmark.py            # ~15 min on the Mac: speed, then accuracy, per format
    python scripts/benchmark.py --backends pytorch,coreml --streams 1,4 --seconds 15

For each backend (vision/backends.py: PyTorch FP32/FP16, Core ML FP16/INT8, ONNX FP32/INT8):

- **Accuracy**, on the test split:
  - the detector's mAP@50 (all classes, and per class);
  - the Phase 2 rule check on one frame: how often a wearer is judged "missing", and how often a
    person without the item is judged "worn", with this backend's detector and keypoints;
  - how long one frame takes through the detector and through the keypoint model.
- **Throughput:** N video files play as N live cameras (in real time, looping). The whole chain
  (readers, batched detector and keypoints, trackers, rules, events) runs for `--seconds` after a
  warm-up, aiming at `--fps` frames per second per camera. Measured:
  - the frames per second each camera actually got;
  - the lag (newest frame's age when its analysis finished), median and 95th percentile;
  - this process's CPU use, as a share of all cores;
  - its memory.
  GPU and Neural Engine use can't be read without administrator rights on macOS
  (`sudo powermetrics`), so they aren't in the table.

The report goes to runs/benchmark/<time>/report.md (+ results.json). Run it plugged in: a MacBook
Air has no fan and slows down when hot, so the later rows can be a little pessimistic.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2
import numpy as np
import psutil

from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.check import attach_poses, check_detections, load_cases, summary
from ppe_monitor.rules.engine import RuleSet
from ppe_monitor.service import CameraSource, MultiCameraMonitor
from ppe_monitor.vision.backends import SPECS, available, load
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR, predict
from ppe_monitor.vision.evaluate import evaluate

DATA = PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"


def model_latency(backend, frame: np.ndarray, settings: Settings, runs: int = 30) -> tuple[float, float]:
    """Median ms for one frame through the detector, and through the keypoint model."""
    out = []
    for fn in (lambda: backend.detect([frame], conf=0.1),
               lambda: backend.keypoints([frame], conf=settings.pose["conf"], keypoint_conf=0.5)):
        for _ in range(5):
            fn()
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        out.append(1000 * statistics.median(times))
    return out[0], out[1]


def accuracy(backend, settings: Settings, cases) -> dict:
    ev = evaluate(backend.detector, DATA, "test", imgsz=640, device=backend.device, with_ultralytics=False)
    per_class = {c.name: c.ap50 for c in ev.classes}
    dets = predict(backend.detector, [c.image for c in cases], conf=0.01)
    if backend.pose is not None and settings.pose["enabled"]:
        attach_poses(cases, backend.pose, keypoint_conf=settings.pose["keypoint_conf"])
    rule = summary(check_detections(cases, dets, settings.rules, settings.thresholds, ["person", "helmet", "vest"]))
    for c in cases:
        c.poses = None
    return {"map50": ev.map50, **{f"ap50_{k}": v for k, v in per_class.items()},
            "helmet_false_alarm": rule["helmet"]["false_alarm"], "helmet_missed": rule["helmet"]["missed"],
            "vest_false_alarm": rule["vest"]["false_alarm"], "vest_missed": rule["vest"]["missed"]}


def throughput(backend, settings: Settings, clips: list[Path], n: int, fps: float, seconds: float,
               warmup: float = 6.0) -> dict:
    sources = [CameraSource(f"cam{i + 1}", str(clips[i % len(clips)])) for i in range(n)]
    mon = MultiCameraMonitor(sources, backend, settings, RuleSet.default(), process_fps=fps, out_dir=None)
    proc = psutil.Process()
    mon.start()
    try:
        mon.run(warmup)
        start_counts = {c: mon.stats[c].processed for c in mon.ids}
        for st in mon.stats.values():
            st.latencies.clear()
        mon.batch_sizes.clear()
        mon.step_times.clear()
        proc.cpu_percent(None)
        t0 = time.monotonic()
        mon.run(seconds)
        elapsed = time.monotonic() - t0
        cpu = proc.cpu_percent(None) / psutil.cpu_count()
        per_cam = [(mon.stats[c].processed - start_counts[c]) / elapsed for c in mon.ids]
        lat = sorted(x for st in mon.stats.values() for x in st.latencies)
        return {"streams": n, "fps_mean": statistics.mean(per_cam), "fps_min": min(per_cam),
                "lag_ms_median": 1000 * lat[len(lat) // 2] if lat else float("nan"),
                "lag_ms_p95": 1000 * lat[int(0.95 * len(lat))] if lat else float("nan"),
                "cpu_percent": cpu, "rss_mb": proc.memory_info().rss / 2 ** 20,
                "batch_mean": statistics.mean(mon.batch_sizes) if mon.batch_sizes else 0,
                "keeps_up": min(per_cam) >= 0.9 * fps}
    finally:
        mon.stop()


def fmt(v, spec=".3f"):
    return "-" if v is None or v != v else format(v, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", help=f"comma-separated, default: all available ({', '.join(SPECS)})")
    parser.add_argument("--streams", default="1,2,4,8")
    parser.add_argument("--seconds", type=float, default=20.0, help="measured time per run, after a 6 s warm-up")
    parser.add_argument("--fps", type=float, default=10.0, help="target frames per second per camera")
    parser.add_argument("--clips", default=str(PROJECT_ROOT / "datasets" / "event_clips_sway"),
                        help="video files played as cameras (default: the Phase 2 moving clips)")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--one", help=argparse.SUPPRESS)          # internal: measure one backend ...
    parser.add_argument("--result", help=argparse.SUPPRESS)       # ... and write its row here
    args = parser.parse_args()

    weights = max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    settings = Settings.load(PPE_CONFIG)
    pose = PRETRAINED_DIR / Path(settings.pose["model"]).name
    status = available(weights, pose)
    wanted = args.backends.split(",") if args.backends else list(SPECS)
    backends = [b for b in wanted if not status.get(b)]
    for b in wanted:
        if status.get(b) and not args.one:
            print(f"Skipping {b}: {status[b]}")
    clips = sorted(Path(args.clips).glob("*_violation.mp4"))[:8] or sorted(Path(args.clips).glob("*.mp4"))[:8]
    if not clips and not args.skip_throughput:
        print(f"No clips in {args.clips}. Make them: python scripts/make_event_clips.py --motion sway")
        return 1
    streams = [int(x) for x in args.streams.split(",")]
    out = PROJECT_ROOT / "runs" / "benchmark" / f"{datetime.now():%Y%m%d_%H%M%S}"
    if not args.one:
        out.mkdir(parents=True, exist_ok=True)
    frame = cv2.imread(str(clips[0])) if clips and str(clips[0]).endswith(".jpg") else None
    if clips:
        cap = cv2.VideoCapture(str(clips[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        frame = cap.read()[1]
        cap.release()
    cases = None if (args.skip_accuracy or not args.one) else load_cases(DATA, "test")

    machine = f"{platform.platform()}, {platform.processor() or platform.machine()}, {psutil.cpu_count()} cores"
    if args.one:                                   # a child process: one backend, one row
        row = measure(args.one, weights, pose, settings, frame, cases, clips, streams, args)
        Path(args.result).write_text(json.dumps(row), encoding="utf-8")
        return 0

    # Each backend is measured in a fresh process, so memory, threads and caches left by one
    # backend (PyTorch, ONNX Runtime and Core ML each keep their own) can't bias the next.
    results = {}
    print(f"Benchmark on {machine}: {', '.join(backends)}; streams {streams}; target {args.fps:g} fps per camera")
    for name in backends:
        print(f"\n=== {name} ({SPECS[name].precision}, {SPECS[name].runs_on})", flush=True)
        row_file = out / f"{name}.json"
        cmd = [sys.executable, __file__, "--one", name, "--result", str(row_file), "--streams", args.streams,
               "--seconds", str(args.seconds), "--fps", str(args.fps), "--clips", args.clips]
        cmd += ["--skip-accuracy"] * args.skip_accuracy + ["--skip-throughput"] * args.skip_throughput
        code = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
        if code != 0 or not row_file.is_file():
            print(f"  {name} failed (exit code {code}); left out of the report")
            continue
        results[name] = json.loads(row_file.read_text(encoding="utf-8"))
        row_file.unlink()
        (out / "results.json").write_text(json.dumps({"machine": machine, "fps_target": args.fps,
                                                      "seconds": args.seconds, "results": results}, indent=1),
                                          encoding="utf-8")

    report = make_report(results, machine, args, streams, settings.pose["every"] if settings.pose["enabled"] else 0)
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved {out / 'report.md'}")
    return 0


def measure(name: str, weights: Path, pose: Path, settings: Settings, frame, cases, clips, streams, args) -> dict:
    t = time.time()
    backend = load(name, weights, pose)
    row = {"precision": SPECS[name].precision, "runs_on": SPECS[name].runs_on,
           "load_s": time.time() - t, "files": [str(f.relative_to(PROJECT_ROOT)) if f else None for f in backend.files]}
    if frame is not None:
        row["det_ms"], row["pose_ms"] = model_latency(backend, frame, settings)
        print(f"  one frame: detector {row['det_ms']:.1f} ms, keypoints {row['pose_ms']:.1f} ms")
    if not args.skip_throughput:
        row["throughput"] = []
        for n in streams:
            r = throughput(backend, settings, clips, n, args.fps, args.seconds)
            row["throughput"].append(r)
            print(f"  {n} camera(s): {r['fps_mean']:.1f} fps each (min {r['fps_min']:.1f}), lag median "
                  f"{r['lag_ms_median']:.0f} ms, p95 {r['lag_ms_p95']:.0f} ms, CPU {r['cpu_percent']:.0f}%, "
                  f"{r['rss_mb']:.0f} MB" + ("" if r["keeps_up"] else "  <- can't keep up"), flush=True)
    # accuracy last: it runs batches of 16 photos, and ONNX Runtime keeps the memory it took for
    # them, which would otherwise count in the live chain's memory figure
    if cases is not None:
        row["accuracy"] = accuracy(backend, settings, cases)
        a = row["accuracy"]
        print(f"  mAP@50 {a['map50']:.3f}; helmet false alarm {a['helmet_false_alarm']:.1%}, missed "
              f"{a['helmet_missed']:.1%}; vest false alarm {a['vest_false_alarm']:.1%}, missed {a['vest_missed']:.1%}")
    return row


def make_report(results: dict, machine: str, args, streams: list[int], pose_every: int = 1) -> str:
    base = results.get("pytorch", {}).get("accuracy")
    kp = ("off" if not pose_every else "on every frame with people" if pose_every == 1 else
          f"on 1 frame in {pose_every} with people, per camera (configs/ppe.yaml, pose.every)")
    lines = [f"# Benchmark: {datetime.now():%Y-%m-%d %H:%M}", "", f"Machine: {machine}. Keypoint model: {kp}.", ""]
    if any("accuracy" in r for r in results.values()) and any("throughput" in r for r in results.values()):
        lines += ["## Speed against accuracy", "",
                  f"Frames analysed per second per camera (target {args.fps:g}) with 1, 2, 4 ... cameras, next to "
                  "the detector's mAP@50 on the test split (change against PyTorch FP32 in brackets). **Bold** = "
                  "every camera got at least 90% of the target.", "",
                  "| Backend | Precision | Runs on | mAP@50 | " +
                  " | ".join(f"{n} camera{'s' if n > 1 else ''}" for n in streams) + " | Cameras at target |",
                  "|---|---|---|---|" + "---|" * (len(streams) + 1)]
        for name, r in results.items():
            a = r.get("accuracy", {})
            d = f" ({a['map50'] - base['map50']:+.3f})" if base and a and name != "pytorch" else ""
            cells = [f"**{t['fps_mean']:.1f}**" if t["keeps_up"] else f"{t['fps_mean']:.1f}" for t in r.get("throughput", [])]
            ok = [t["streams"] for t in r.get("throughput", []) if t["keeps_up"]]
            lines.append(f"| {name} | {r['precision']} | {r['runs_on']} | {fmt(a.get('map50'))}{d} | " +
                         " | ".join(cells) + f" | {max(ok) if ok else 0} |")
        lines.append("")
    if any("accuracy" in r for r in results.values()):
        lines += ["## Accuracy (test split)", "",
                  "| Backend | Precision | Runs on | mAP@50 | person | helmet | vest | Helmet: wearer judged missing | "
                  "Helmet: bare judged worn | Vest: wearer judged missing | Vest: bare judged worn | "
                  "Detector ms/frame | Keypoints ms/frame |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            a = r.get("accuracy", {})
            d = f" ({a['map50'] - base['map50']:+.3f})" if base and a and name != "pytorch" else ""
            lines.append(f"| {name} | {r['precision']} | {r['runs_on']} | {fmt(a.get('map50'))}{d} | "
                         f"{fmt(a.get('ap50_person'))} | {fmt(a.get('ap50_helmet'))} | {fmt(a.get('ap50_vest'))} | "
                         f"{fmt(a.get('helmet_false_alarm'), '.1%')} | {fmt(a.get('helmet_missed'), '.1%')} | "
                         f"{fmt(a.get('vest_false_alarm'), '.1%')} | {fmt(a.get('vest_missed'), '.1%')} | "
                         f"{fmt(r.get('det_ms'), '.1f')} | {fmt(r.get('pose_ms'), '.1f')} |")
    if any("throughput" in r for r in results.values()):
        lines += ["", f"## Throughput: frames analysed per second per camera (target {args.fps:g}), "
                  f"with the 95th-percentile lag", "",
                  "Whole chain: readers, detector and keypoints (batched across cameras where the backend "
                  "allows), trackers, rules, events. **Bold** = every camera got at least 90% of the target.", "",
                  "| Backend | " + " | ".join(f"{n} camera{'s' if n > 1 else ''}" for n in streams) + " | CPU (of all cores) at "
                  f"{streams[-1]} |", "|---|" + "---|" * (len(streams) + 1)]
        for name, r in results.items():
            cells = []
            for t in r.get("throughput", []):
                cell = f"{t['fps_mean']:.1f} fps, {t['lag_ms_p95']:.0f} ms"
                cells.append(f"**{cell}**" if t["keeps_up"] else cell)
            cpu = r["throughput"][-1]["cpu_percent"] if r.get("throughput") else float("nan")
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {fmt(cpu, '.0f')}% |")
        lines += ["", "Cameras each backend can watch at the target rate (largest measured count that kept up):", ""]
        for name, r in results.items():
            ok = [t["streams"] for t in r.get("throughput", []) if t["keeps_up"]]
            total = max((t["fps_mean"] * t["streams"] for t in r.get("throughput", [])), default=0)
            lines.append(f"- {name}: {max(ok) if ok else 0} camera(s); at most {total:.0f} frames analysed per second "
                         "in total")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
