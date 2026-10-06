#!/usr/bin/env python3
"""The project's own PostgreSQL server (Phase 5): events, cameras, rules and the alert queue.

    python scripts/db.py init      # once: create data/postgres, start it, create the database and tables
    python scripts/db.py start     # start it (e.g. after a restart of the Mac)
    python scripts/db.py stop
    python scripts/db.py status    # running? how many events and alerts?

It listens on 127.0.0.1:5433 only (configs/server.yaml), with a password kept in
configs/secrets.env. To use SQLite instead (no server), put DATABASE_URL=sqlite:///data/ppe.sqlite
in configs/secrets.env; then only `init` (which creates the tables) and `status` apply.
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from ppe_monitor.backend import localpg
from ppe_monitor.backend.settings import ServerSettings
from ppe_monitor.backend.store import EventStore
from ppe_monitor.config import ConfigError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["init", "start", "stop", "status"])
    args = parser.parse_args()
    try:
        s = ServerSettings.load()
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 1
    external = bool(s.secrets.get("DATABASE_URL") or s.secrets.get("PPE_DB_HOST"))
    try:
        if args.action == "init":
            if not external:
                made = localpg.init(s)
                localpg.start()
                localpg.ensure_database(s)
                if not made:
                    print(f"Database server already set up in {localpg.DATA_DIR}")
            store = EventStore(s.database_url, s.storage_dir)       # creates the tables
            print(f"Database ready: {s.safe_database_url}")
            store.close()
        elif args.action == "start":
            if external:
                print(f"DATABASE_URL is set ({s.safe_database_url}): nothing to start here.")
            elif not localpg.start():
                print("Database server already running.")
        elif args.action == "stop":
            if external or not localpg.stop():
                print("Database server not running.")
        else:
            if not external:
                print(f"Server files: {localpg.DATA_DIR} ({'set up' if localpg.initialised() else 'not set up yet'}); "
                      f"{'running' if localpg.running() else 'not running'}")
            store = EventStore(s.database_url, s.storage_dir, create=False)
            _, n = store.list_events(limit=1)
            print(f"Database {s.safe_database_url}: {n} event(s); alerts {store.alert_counts() or 'none'}")
            store.close()
    except localpg.PgError as exc:
        print(f"Problem: {exc}")
        return 1
    except Exception as exc:
        print(f"Can't use the database {s.safe_database_url}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
