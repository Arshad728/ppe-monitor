#!/usr/bin/env python3
"""Watch several cameras at once (Phase 4).

    python scripts/run_cameras.py                          # every enabled camera in configs/cameras.yaml
    python scripts/run_cameras.py --cameras cam1,cam3 --show
    python scripts/run_cameras.py --backend coreml --fps 10 --duration 300
    python scripts/run_cameras.py --files datasets/event_clips_sway --count 4   # video files as cameras

Every camera gets its own reader thread, which keeps only the newest frame and reconnects by
itself when the camera drops out. The analysis takes the newest frame of each camera
`process_fps` times a second, in one batch, and runs each camera's own tracker, rules and events.

- **Events and snapshots** go to runs/live/<time>/events.jsonl and snapshots/.
- **Camera connections** go to runs/live/<time>/cameras.jsonl: each time one drops out or comes back.
- **Status:** every 10 s a table shows, per camera, the analysed frames per second, the lag
  (how old the newest frame was when its analysis finished), events and reconnections.
- **--show** opens one window with all cameras, redrawn a few times a second apart from the analysis.
- **--db** (Phase 5) also stores every event, with its evidence image, in the database and queues its
  alerts (configs/server.yaml); the alert worker sends them. It never makes the video wait.
- **--live** (Phase 5) writes each camera's newest annotated frame about once a second, for the
  dashboard's camera tiles.

Settings: configs/streams.yaml (rate, backend, reconnection). Press Ctrl+C (or q in the window) to stop.
"""

import argparse
import platform
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2

from ppe_monitor.config import PROJECT_ROOT, ConfigError, load_cameras
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.engine import RuleSet
from ppe_monitor.service import CameraSource, MultiCameraMonitor, StreamSettings, choose_backend
from ppe_monitor.vision.backends import SPECS, load
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR


def sources_from(args) -> list[CameraSource]:
    if args.files:
        clips = sorted(Path(args.files).glob("*.mp4")) if Path(args.files).is_dir() else [Path(args.files)]
        clips = clips[:args.count] if args.count else clips
        return [CameraSource(f"file{i + 1}", str(c)) for i, c in enumerate(clips)]
    cams = load_cameras()
    if args.cameras != "all":
        wanted = args.cameras.split(",")
        unknown = set(wanted) - {c.id for c in cams}
        if unknown:
            raise ConfigError(f"not in configs/cameras.yaml: {', '.join(sorted(unknown))}")
        cams = [c for c in cams if c.id in wanted]
    return [CameraSource(c.id, c.url, c.transport) for c in cams]


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)   # show events and status as they happen, also through tee
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", default="all", help="comma-separated camera ids, or all")
    parser.add_argument("--files", help="use video files as cameras instead: a file or a folder of .mp4")
    parser.add_argument("--count", type=int, help="with --files: how many")
    parser.add_argument("--backend", choices=["auto", *SPECS], help="default: configs/streams.yaml")
    parser.add_argument("--fps", type=float, help="frames analysed per second per camera (default: streams.yaml)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--show", action="store_true", help="a window with all cameras")
    parser.add_argument("--out", help="folder for events and snapshots (default runs/live/<time>)")
    parser.add_argument("--db", action="store_true", help="store events in the database and queue their alerts (Phase 5)")
    parser.add_argument("--live", action="store_true", help="write live pictures for the dashboard (Phase 5)")
    args = parser.parse_args()

    try:
        streams = StreamSettings.load()
        settings = Settings.load(PPE_CONFIG)
        rules = RuleSet.load()
        sources = sources_from(args)
    except (ConfigError, ValueError) as exc:
        print(f"Config problem: {exc}")
        return 1
    if not sources:
        print("No cameras to watch.")
        return 1
    weights = max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if not weights:
        print("No trained model in models/.")
        return 1
    pose = PRETRAINED_DIR / Path(settings.pose["model"]).name if settings.pose["enabled"] else None
    name = choose_backend(args.backend or streams.backend, weights, pose)
    try:
        backend = load(name, weights, pose)
    except RuntimeError as exc:
        print(exc)
        return 1
    fps = args.fps or streams.process_fps
    out = Path(args.out) if args.out else PROJECT_ROOT / "runs" / "live" / f"{datetime.now():%Y%m%d_%H%M%S}"
    db_sink = server = None
    if args.db or args.live:
        from ppe_monitor.backend.settings import ServerSettings
        server = ServerSettings.load()
    if args.db:
        from ppe_monitor.backend.sink import DbSink
        from ppe_monitor.backend.store import EventStore
        try:
            store = EventStore(server.database_url, server.storage_dir)
            names = {c.id: c.name for c in load_cameras()} if not args.files else {}
            run_id = store.start_run(host=platform.node(), backend=name, model=weights.stem,
                                     camera_ids=[s.id for s in sources])
            store.sync_config({s.id: names.get(s.id, Path(s.source).name if args.files else s.id) for s in sources},
                              rules)
        except Exception as exc:
            print(f"Can't open the database {server.safe_database_url}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0]}\nStart it with: python scripts/db.py start (or run without --db)")
            return 1
        db_sink = DbSink(store, server, run_id)
        print(f"Events also go to the database ({server.safe_database_url}); alerts queued for: "
              f"{', '.join(db_sink.channels) or 'no channel (none enabled in configs/server.yaml)'}")
    monitor = MultiCameraMonitor(sources, backend, settings, rules, process_fps=fps, imgsz=streams.imgsz,
                                 out_dir=out, keep_frames=args.show or args.live, reader_options=streams.reader_options(),
                                 sinks=[db_sink] if db_sink else None)
    tiles = None
    if args.live:
        from ppe_monitor.backend.sink import LiveTiles
        tiles = LiveTiles(monitor, server.live_dir, server.live_every, server.live_width)
    print(f"Watching {len(sources)} camera(s) at {fps:g} frames/s each, backend {name} "
          f"({backend.spec.precision}, {backend.spec.runs_on}). Events -> {out}")
    stopping = []
    signal.signal(signal.SIGINT, lambda *a: stopping.append(1))
    signal.signal(signal.SIGTERM, lambda *a: stopping.append(1))
    if hasattr(signal, "SIGHUP"):                 # its Terminal window closed: stop cleanly too
        signal.signal(signal.SIGHUP, lambda *a: stopping.append(1))
    monitor.start()
    start = last_print = last_show = last_report = time.monotonic()
    title = "cameras (q to stop)"
    try:
        while not stopping and (args.duration is None or time.monotonic() - start < args.duration):
            for cid, r in monitor.step():
                for ev in r.events:
                    print(f"  EVENT {cid:8s} {ev.kind:15s} {ev.severity:8s} track #{ev.track_id:<4d} "
                          f"{ev.time[11:19]}" + (f"  zone {ev.zone}" if ev.zone else ""))
            now = time.monotonic()
            if db_sink is not None and now - last_report >= 5:
                last_report = now
                db_sink.stats(monitor.status_rows(), {"backend": name, "cameras": len(sources), "fps_target": fps})
            if now - last_print >= 10:
                last_print = now
                print(f"\n[{datetime.now():%H:%M:%S}]\n{monitor.status_text()}")
            if args.show and now - last_show >= 1 / streams.display_fps:
                last_show = now
                img = monitor.mosaic()
                if img is not None:
                    cv2.imshow(title, img)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        monitor.stop()
        if tiles is not None:
            tiles.close()
        if db_sink is not None:
            db_sink.stats(monitor.status_rows(), {"backend": name, "cameras": len(sources), "stopped": True})
            db_sink.close()
            print(f"Database: {db_sink.stored} event(s) stored" + (f", {db_sink.dropped} dropped" if db_sink.dropped else ""))
        if args.show:
            cv2.destroyAllWindows()
    print(f"\nStopped after {time.monotonic() - start:.0f} s.\n{monitor.status_text()}\nEvents: {out / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
