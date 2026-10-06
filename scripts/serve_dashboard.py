#!/usr/bin/env python3
"""The safety officer's dashboard (Phase 5): http://127.0.0.1:8080

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --host 0.0.0.0     # reachable from a phone on the same Wi-Fi
                                                         # (needs DASHBOARD_PASSWORD in configs/secrets.env)

Live camera pictures, the event feed with each event's evidence image, counts by hour and by
camera, and a confirm / false-alarm verdict on each event. Settings: configs/server.yaml
(dashboard). What it serves: src/ppe_monitor/backend/api.py.
"""

import argparse
import socket
import sys

import _bootstrap  # noqa: F401

from ppe_monitor.backend.api import create_app
from ppe_monitor.backend.settings import ServerSettings
from ppe_monitor.backend.store import EventStore
from ppe_monitor.config import ConfigError

LOCAL = {"127.0.0.1", "localhost", "::1"}


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))           # no packet is sent; this only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "this-computer"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)     # the address shows at once, also in a log file
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="default: dashboard.host in configs/server.yaml")
    parser.add_argument("--port", type=int, help="default: dashboard.port")
    args = parser.parse_args()
    try:
        s = ServerSettings.load()
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 1
    host, port = args.host or s.dashboard_host, args.port or s.dashboard_port
    if host not in LOCAL and not s.secrets.get("DASHBOARD_PASSWORD"):
        print(f"Refusing to listen on {host}: anyone on the network could see the cameras. Set DASHBOARD_PASSWORD "
              "in configs/secrets.env first (and DASHBOARD_USER, default 'safety').")
        return 1
    try:
        store = EventStore(s.database_url, s.storage_dir)
    except Exception as exc:
        print(f"Can't open the database {s.safe_database_url}: {type(exc).__name__}: {str(exc).splitlines()[0]}\n"
              "Start it with: python scripts/db.py start")
        return 1
    import uvicorn

    app = create_app(store, s)
    shown = "127.0.0.1" if host in LOCAL or host == "0.0.0.0" else host
    print(f"Dashboard: http://{shown}:{port}" + (f"  (on the Wi-Fi: http://{lan_address()}:{port})" if host == "0.0.0.0" else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
