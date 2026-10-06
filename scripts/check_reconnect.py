#!/usr/bin/env python3
"""A camera drops out and comes back; the others must not notice (Phase 4, the book's "done when").

    python scripts/check_reconnect.py                 # 4 fake RTSP cameras, one unplugged for 15 s
    python scripts/check_reconnect.py --cameras 8 --down 30

Starts a private fake camera network (MediaMTX on a free port, one ffmpeg per camera, looping the
Phase 0 clips), watches every camera with the multi-camera engine, then:

    ~10 s   camera 2's publisher is killed: the camera is gone
    +down   it is started again: the camera is back

It passes when:
  1. nothing crashed;
  2. the other cameras kept being analysed while camera 2 was gone: at least 90 % of their rate
     before, and no gap between two analysed frames longer than 1 s (or 3 frame periods, when
     the machine analyses fewer than 3 frames a second per camera). A reader that blocked the loop while reconnecting would show up here as
     a gap of several seconds (the stream open timeout);
  3. camera 2 was reported as reconnecting, came back by itself and was analysed again.

The report (with a second-by-second timeline) goes to runs/reconnect/<time>/report.md.
"""

import argparse
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT, Camera
from ppe_monitor.executables import find_ffmpeg, find_mediamtx
from ppe_monitor.ingestion.reader import LIVE
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.engine import RuleSet
from ppe_monitor.service import CameraSource, MultiCameraMonitor, StreamSettings, choose_backend
from ppe_monitor.sim.fake_cameras import FakeCameraNetwork, write_server_config
from ppe_monitor.vision.backends import SPECS, load
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)   # show events and status as they happen, also through tee
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--down", type=float, default=15.0, help="seconds camera 2 stays unplugged")
    parser.add_argument("--fps", type=float, help="frames analysed per second per camera (default: streams.yaml)")
    parser.add_argument("--backend", choices=["auto", *SPECS])
    args = parser.parse_args()

    if not (find_ffmpeg() and find_mediamtx()):
        print("Needs ffmpeg and MediaMTX (Phase 0): python scripts/get_mediamtx.py")
        return 1
    clips = sorted((PROJECT_ROOT / "data" / "clips").glob("synthetic_cam*.mp4"))
    if not clips:
        print("No simulated camera clips. Make them: python scripts/make_test_clips.py")
        return 1
    streams = StreamSettings.load()
    settings = Settings.load(PPE_CONFIG)
    weights = max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    pose = PRETRAINED_DIR / Path(settings.pose["model"]).name if settings.pose["enabled"] else None
    name = choose_backend(args.backend or streams.backend, weights, pose)
    backend = load(name, weights, pose)
    fps = args.fps or streams.process_fps
    out = PROJECT_ROOT / "runs" / "reconnect" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    port = free_port()
    tmp = Path(tempfile.mkdtemp(prefix="reconnect_"))
    cams = [Camera(id=f"rc{i + 1}", name=f"rc{i + 1}", url=f"rtsp://127.0.0.1:{port}/rc{i + 1}",
                   sim_source=clips[i % len(clips)]) for i in range(args.cameras)]
    net = FakeCameraNetwork(cams, start_server=True, log_dir=out / "fake_camera_logs",
                            server_config=write_server_config(tmp / "mediamtx.yml", port=port, tcp_only=True))
    print(f"Starting {args.cameras} fake cameras on port {port} ...")
    net.start(ready_timeout=30)
    victim = cams[1].id
    mon = MultiCameraMonitor([CameraSource(c.id, c.url) for c in cams], backend, settings, RuleSet.default(),
                             process_fps=fps, out_dir=out, snapshots=False, reader_options=streams.reader_options())
    done: dict[str, list[float]] = {c.id: [] for c in cams}
    marks = {}
    crashed = None
    print(f"Watching them at {fps:g} fps each with {name}. Camera {victim} goes down at ~10 s for {args.down:g} s.")
    mon.start()
    t0 = time.monotonic()
    try:
        while not all(r.stats.state == LIVE for r in mon.readers.values()) and time.monotonic() - t0 < 20:
            mon.step()
        t0 = time.monotonic()
        phase = "before"
        while True:
            now = time.monotonic() - t0
            if phase == "before" and now >= 10:
                net.pause(victim)
                marks["down"], phase = now, "down"
                print(f"  {now:5.1f} s  {victim} unplugged")
            elif phase == "down" and now >= 10 + args.down:
                net.resume(victim)
                marks["up"], phase = now, "up"
                print(f"  {now:5.1f} s  {victim} plugged back in")
            elif phase == "up" and (done[victim] and done[victim][-1] > marks["up"] + 3 or now > marks["up"] + 40):
                break
            for cid, _ in mon.step():
                done[cid].append(time.monotonic() - t0)
            if victim in [c for c in done] and phase == "up" and "back" not in marks and done[victim] \
                    and done[victim][-1] > marks["up"]:
                marks["back"] = done[victim][-1]
                print(f"  {marks['back']:5.1f} s  {victim} analysed again ({marks['back'] - marks['up']:.1f} s after it came back)")
    except Exception as exc:          # the whole point is that this doesn't happen
        crashed = repr(exc)
    finally:
        reconnects = mon.readers[victim].stats.reconnects
        mon.stop()
        net.stop()

    end = max((v[-1] for v in done.values() if v), default=0)
    others = [c.id for c in cams if c.id != victim]
    down, up = marks.get("down", 10.0), marks.get("up", end)
    gaps = {}
    for cid in others:
        during = [t for t in done[cid] if down - 1 <= t <= up + 1]
        gaps[cid] = max((b - a for a, b in zip(during, during[1:])), default=float("inf"))
    settle = 2.0                          # the first seconds after all cameras came up are not counted
    rate = {cid: (len([t for t in done[cid] if settle <= t < down]) / max(1e-6, down - settle),
                  len([t for t in done[cid] if down <= t <= up]) / max(1e-6, up - down)) for cid in others}
    # 1 s, or 3 frame periods when the machine analyses fewer than 3 frames a second per camera
    slowest = min([fps] + [a for a, _ in rate.values() if a > 0])
    longest_ok = max(1.0, 3.0 / slowest)
    kept_up = all(g < longest_ok and b >= 0.9 * a for (g, (a, b)) in zip(gaps.values(), rate.values()))
    checks = [
        ("nothing crashed", crashed is None, crashed or "no exception"),
        ("the other cameras never stalled", kept_up,
         ", ".join(f"{c}: {rate[c][1]:.1f} fps (before {rate[c][0]:.1f}), longest gap {g:.2f} s"
                   for c, g in gaps.items()) + f"; allowed {longest_ok:.1f} s"),
        (f"{victim} reconnected by itself", "back" in marks and reconnects >= 1,
         f"analysed again {marks['back'] - marks['up']:.1f} s after it came back" if "back" in marks else "never came back"),
    ]
    lines = [f"# Reconnection check: {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{args.cameras} fake RTSP cameras, analysed at {fps:g} fps each with {name}. {victim} was unplugged at "
             f"{marks.get('down', float('nan')):.1f} s and plugged back in at {marks.get('up', float('nan')):.1f} s.", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {what} | {'PASS' if ok else 'FAIL'} | {detail} |" for what, ok, detail in checks]
    lines += ["", "Frames analysed per second, before and while it was down:", ""]
    lines += [f"- {cid}: {a:.1f} before, {b:.1f} during" for cid, (a, b) in rate.items()]
    lines += ["", "## Timeline (frames analysed in each second)", "",
              "| second | " + " | ".join(c.id for c in cams) + " |", "|---|" + "---|" * len(cams)]
    for s in range(int(end) + 1):
        note = " (down)" if marks.get("down", 1e9) <= s < marks.get("up", 1e9) else ""
        lines.append(f"| {s}{note} | " + " | ".join(str(sum(1 for t in done[c.id] if s <= t < s + 1)) for c in cams) + " |")
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + "\n".join(lines[:12 + len(others)]))
    print(f"\nFull report: {out / 'report.md'}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
