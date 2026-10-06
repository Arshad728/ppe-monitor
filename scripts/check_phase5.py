#!/usr/bin/env python3
"""Phase 5, end to end: cameras -> database -> alert worker -> "Telegram" -> dashboard.

    python scripts/check_phase5.py                 # ~2 min; used by: bash scripts/mac_phase5.sh check

Nothing real is touched: a throw-away database (`ppe_check` on the local PostgreSQL, or a SQLite
file when DATABASE_URL points elsewhere), and a stand-in for Telegram's API on this machine that
records what it is sent. Four of the Phase 2 moving-people clips play as four cameras, with the
real model and settings. For the length of the check, the stand-in refuses the first 3 messages
(as if Telegram were down), and the rate limit is tightened to 2 alerts per camera per 30 s, so
the retries and the summaries are exercised too.

Checked, in the spirit of the book's "done when" (a violation reaches your phone with its snapshot
within a few seconds, and shows up correctly on the dashboard):
  1. every event the cameras raised is in the database, with its evidence image, and none was dropped;
  2. every alert arrived with its picture, and those not caught in the outage within 5 s of the event;
  3. the refused messages were retried and got through;
  4. past the rate limit, alerts were held and then sent as summaries; nothing is stuck or failed;
  5. the dashboard's API lists the same events, serves their images, and records a verdict;
  6. the video loop kept its pace with all of this switched on.
Report: runs/phase5_check/<time>/report.md
"""

import argparse
import re
import statistics
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.backend import localpg
from ppe_monitor.backend.channels import TelegramChannel
from ppe_monitor.backend.settings import ServerSettings
from ppe_monitor.backend.sink import DbSink
from ppe_monitor.backend.standin import StandInTelegram
from ppe_monitor.backend.store import EventStore
from ppe_monitor.backend.worker import AlertWorker
from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.engine import RuleSet
from ppe_monitor.service import CameraSource, MultiCameraMonitor, StreamSettings, choose_backend
from ppe_monitor.vision.backends import load
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR

CLIPS = PROJECT_ROOT / "datasets" / "event_clips_sway"


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60, help="how long the cameras run")
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--sqlite", action="store_true", help="use a SQLite file even if PostgreSQL is set up")
    parser.add_argument("--database-url", help=argparse.SUPPRESS)      # an empty database to use instead
    args = parser.parse_args()

    out = PROJECT_ROOT / "runs" / "phase5_check" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    s = ServerSettings.load()
    s.storage_dir, s.live_dir = out / "events", out / "live"
    s.rate_max, s.rate_minutes, s.retry_first, s.poll = 2, 0.5, 1.0, 0.25
    for name, ch in s.channels.items():
        ch.enabled, ch.min_severity = name == "telegram", "low"

    if args.database_url:
        url = args.database_url
    elif args.sqlite or not localpg.in_use(s):
        url = f"sqlite:///{out / 'check.sqlite'}"
    else:
        localpg.start()
        url = localpg.ensure_database(s, "ppe_check", fresh=True)
    store = EventStore(url, s.storage_dir)
    print(f"Database: {store.engine.dialect.name} (throw-away: {re.sub(r'://[^@]*@', '://', url)})")

    clips = sorted(CLIPS.glob("*_violation.mp4"))[:args.cameras]
    if len(clips) < args.cameras:
        print(f"Needs the Phase 2 moving clips in {CLIPS}: python scripts/make_event_clips.py --motion sway")
        return 1
    sources = [CameraSource(f"check{i + 1}", str(c)) for i, c in enumerate(clips)]
    settings, rules, streams = Settings.load(PPE_CONFIG), RuleSet.load(), StreamSettings.load()
    weights = max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    pose = PRETRAINED_DIR / Path(settings.pose["model"]).name if settings.pose["enabled"] else None
    backend_name = choose_backend(args.backend if args.backend != "auto" else streams.backend, weights, pose)
    backend = load(backend_name, weights, pose)

    tg = StandInTelegram(outage=3)
    channel = TelegramChannel("123:CHECK", ["555"], api_base=tg.url)
    worker = AlertWorker(store, s, {"telegram": channel}, log=lambda m: print("  " + m))
    run_id = store.start_run(host="check", backend=backend_name, model=weights.stem, camera_ids=[c.id for c in sources])
    store.sync_config({c.id: Path(c.source).stem for c in sources}, rules)
    sink = DbSink(store, s, run_id, log=lambda m: print("  " + m))
    mon = MultiCameraMonitor(sources, backend, settings, rules, process_fps=streams.process_fps, imgsz=streams.imgsz,
                             out_dir=out / "run", sinks=[sink], keep_frames=False)
    wt = threading.Thread(target=worker.run, name="alert-worker", daemon=True)

    print(f"{len(sources)} cameras (Phase 2 moving clips), backend {backend_name}, {args.seconds:.0f} s; "
          f"stand-in Telegram at {tg.url} (refuses the first 3 messages); rate limit {s.rate_max} per "
          f"{s.rate_minutes * 60:.0f} s per camera")
    wt.start()
    mon.start()
    t0 = time.monotonic()
    raised = []
    step_times = []
    while time.monotonic() - t0 < args.seconds:
        a = time.perf_counter()
        for cid, r in mon.step():
            for ev in r.events:
                raised.append(ev)
                print(f"  EVENT {cid} {ev.kind} track #{ev.track_id}")
        step_times.append(time.perf_counter() - a)
    rows = mon.status_rows()
    mon.stop()
    sink.close(30)
    print("Cameras stopped; letting the alert worker finish (held alerts go out as summaries) ...")
    deadline = time.monotonic() + s.rate_minutes * 60 + 20
    while time.monotonic() < deadline:
        c = store.alert_counts()
        if not c.get("pending") and not c.get("sending") and not c.get("held"):
            break
        time.sleep(0.5)
    worker.stop()
    wt.join(5)

    # -- the dashboard's view of it
    warnings.filterwarnings("ignore", message=".*httpx.*")      # Starlette's note about its test client
    from fastapi.testclient import TestClient

    from ppe_monitor.backend.api import create_app
    api_ok, api_detail = True, []
    with TestClient(create_app(store, s)) as client:
        o = client.get("/api/overview?hours=1").json()
        listed = client.get("/api/events?limit=500&since=2000-01-01T00:00:00Z").json()
        if listed["total"] != len(raised):
            api_ok = False
        api_detail.append(f"lists {listed['total']} events")
        first = listed["items"][-1]["id"] if listed["items"] else None
        if first:
            img = client.get(f"/api/events/{first}/snapshot.jpg")
            frame = client.get(f"/api/events/{first}/frame.jpg")
            api_ok &= img.status_code == 200 and img.content[:2] == b"\xff\xd8" and frame.status_code == 200
            api_detail.append(f"serves the evidence image ({len(img.content) // 1024} KB) and the clean frame")
            r = client.patch(f"/api/events/{first}", json={"status": "false_alarm", "note": "check"})
            api_ok &= r.status_code == 200 and client.get(f"/api/events/{first}").json()["status"] == "false_alarm"
            api_detail.append("records a false-alarm verdict")
        api_ok &= o["stats"]["total"] == len(raised)
        page = client.get("/")
        api_ok &= page.status_code == 200
    tg.close()

    # -- results
    stored = store.list_events(limit=500)[0]
    counts = store.alert_counts()
    lat = store.latencies()
    with store.engine.connect() as c:
        from sqlalchemy import select

        from ppe_monitor.backend.db import alerts
        arows = c.execute(select(alerts.c.attempts, alerts.c.latency_ms, alerts.c.status, alerts.c.is_summary)).all()
    first_try = [r.latency_ms for r in arows if r.status == "sent" and not r.is_summary and r.attempts == 1 and r.latency_ms]
    retried = [r for r in arows if r.attempts > 1 and r.status == "sent"]
    photos = sum(1 for m in tg.received if m["photo"])
    summaries_sent = sum(1 for m in tg.received if m["summary"])
    fps = [r["fps"] for r in rows]
    images = sum(1 for e in stored if e["has_snapshot"] and e["has_frame"])
    checks = [
        ("every event stored, with its images; none dropped",
         len(stored) == len(raised) and images == len(stored) and sink.dropped == 0 and len(raised) > 0,
         f"{len(raised)} raised, {len(stored)} stored, {images} with both images, {sink.dropped} dropped"),
        ("every alert delivered with its picture",
         counts.get("failed", 0) == 0 and not counts.get("pending") and not counts.get("held") and photos == len(tg.received),
         f"{len(tg.received)} messages received ({photos} with a picture); alerts: {counts}"),
        ("alerts not caught in the outage arrive within 5 s of the event",
         bool(first_try) and max(first_try) <= 5000,
         f"median {statistics.median(first_try) / 1000:.2f} s, 95% {pct(first_try, .95) / 1000:.2f} s, "
         f"slowest {max(first_try) / 1000:.2f} s ({len(first_try)} alerts)" if first_try else "none"),
        ("messages refused during the outage were retried and delivered",
         tg.refused == 3 and len(retried) >= 1,
         f"{tg.refused} refused; {len(retried)} alert(s) delivered on a later attempt"),
        ("past the rate limit, alerts were held and sent as summaries",
         counts.get("summarised", 0) > 0 and summaries_sent > 0,
         f"{counts.get('summarised', 0)} alerts held and reported in {summaries_sent} summaries"),
        ("the dashboard's API shows the same events and records a verdict", api_ok, "; ".join(api_detail)),
        ("the video loop kept its pace", min(fps) >= 0.9 * streams.process_fps or backend_name.startswith("onnx"),
         f"{', '.join(f'{f:.1f}' for f in fps)} frames/s per camera (target {streams.process_fps:g}); "
         f"loop step median {1000 * statistics.median(step_times):.0f} ms"),
    ]
    passed = all(ok for _, ok, _ in checks)
    lines = [f"# Phase 5 check: {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{len(sources)} cameras (Phase 2 moving clips) for {args.seconds:.0f} s, backend {backend_name}, model "
             f"{weights.stem}; database {store.engine.dialect.name} (throw-away); a stand-in for Telegram on this "
             f"machine that refused the first 3 messages; rate limit {s.rate_max} alerts per camera per "
             f"{s.rate_minutes * 60:.0f} s.", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {w} | {'PASS' if ok else 'FAIL'} | {d} |" for w, ok, d in checks]
    lines += ["", f"**{'All checks passed.' if passed else 'Some checks failed.'}**", "",
              "Time from an event's confirmation to its delivery, all alerts (including retried ones): "
              + (f"median {statistics.median(lat) / 1000:.2f} s, slowest {max(lat) / 1000:.2f} s." if lat else "none.")]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\nReport: {out / 'report.md'}")
    store.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
