#!/usr/bin/env python3
"""Send the alerts the camera service queues (Phase 5): Telegram and email, with the snapshot.

    python scripts/alert_worker.py              # runs until Ctrl+C
    python scripts/alert_worker.py --once       # send what is queued now, then stop

It runs apart from the camera service on purpose: a slow or failing Telegram or mail server can
never hold up the video. Channels, severity thresholds, retries and the per-camera rate limit
are in configs/server.yaml; tokens and passwords in configs/secrets.env. How it decides:
src/ppe_monitor/backend/worker.py.
"""

import argparse
import signal
import sys

import _bootstrap  # noqa: F401

from ppe_monitor.backend.settings import ServerSettings
from ppe_monitor.backend.store import EventStore
from ppe_monitor.backend.worker import AlertWorker
from ppe_monitor.config import ConfigError


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="send what is due now, then stop")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    args = parser.parse_args()
    try:
        s = ServerSettings.load()
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 1
    try:
        store = EventStore(s.database_url, s.storage_dir)
    except Exception as exc:
        print(f"Can't open the database {s.safe_database_url}: {type(exc).__name__}: {str(exc).splitlines()[0]}\n"
              "Start it with: python scripts/db.py start")
        return 1
    worker = AlertWorker(store, s)
    enabled = [n for n, c in s.channels.items() if c.enabled]
    for n in enabled:
        why = worker.channels[n].missing()
        print(f"  {n}: {'ready' if not why else 'NOT SET UP - ' + why} (alerts of severity {s.channels[n].min_severity} and above)")
    if not enabled:
        print("  No channel is enabled in configs/server.yaml (alerts.channels): events are stored, nothing is sent.")
    print(f"Alert worker: database {s.safe_database_url}; at most {s.rate_max} alerts per camera and channel "
          f"every {s.rate_minutes:g} min, the rest summarised.")
    signal.signal(signal.SIGTERM, lambda *a: worker.stop())
    try:
        if args.once:
            worker.send_summaries()
            while worker.process():
                pass
            worker.housekeeping(force=True)
        else:
            worker.run(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        store.close()
    print(f"Alert worker stopped: {worker.sent} sent, {worker.summaries} summaries, {worker.held} held, "
          f"{worker.failed} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
