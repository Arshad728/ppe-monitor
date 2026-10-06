#!/usr/bin/env python3
"""Small jobs for the Docker Compose stack (Phase 6). docker-compose.yml calls these; you rarely need to.

    python scripts/docker_helper.py secrets           # first start: make the database and dashboard passwords
    python scripts/docker_helper.py show-login        # print the dashboard's user name and password
    python scripts/docker_helper.py health SERVICE    # healthcheck: dashboard | camera_service | alert_worker
    python scripts/docker_helper.py demo-cameras      # the simulated cameras of configs/cameras.yaml, over RTSP
    python scripts/docker_helper.py stand-in-telegram # a stand-in for api.telegram.org (for checks)

Passwords are made once, at random, into the `ppe-secrets` volume (/run/ppe-secrets), readable only
by the containers that mount it. Values in configs/secrets.env win over them: that file is handed to
every service (env_file), and an environment variable beats a *_FILE secret.
"""

import os
import secrets as pysecrets
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

SECRETS = Path(os.environ.get("PPE_SECRETS_DIR", "/run/ppe-secrets"))
POSTGRES_UID = 70            # the postgres user of the postgres:*-alpine image, which must read its password


def make_secrets() -> int:
    SECRETS.mkdir(parents=True, exist_ok=True)
    made = []
    for name, owner in (("db_password", POSTGRES_UID), ("dashboard_password", 0)):
        p = SECRETS / name
        if not p.is_file() or not p.read_text().strip():
            p.write_text(pysecrets.token_urlsafe(18) + "\n")
            made.append(name)
        p.chmod(0o600)
        try:
            os.chown(p, owner, owner)
        except (PermissionError, OSError):
            pass
    print("Secrets: " + (f"made {', '.join(made)}" if made else "already there") + f" in {SECRETS}")
    return show_login()


def show_login() -> int:
    user = os.environ.get("DASHBOARD_USER") or "safety"
    if os.environ.get("DASHBOARD_PASSWORD"):
        print(f"Dashboard login: {user} / the DASHBOARD_PASSWORD in configs/secrets.env")
    else:
        p = SECRETS / "dashboard_password"
        pw = p.read_text().strip() if p.is_file() else "(not made yet)"
        print(f"Dashboard login: user {user}, password {pw}\n"
              f"  (see it again: docker compose logs secrets; or set DASHBOARD_PASSWORD in configs/secrets.env)")
    return 0


def health(what: str) -> int:
    """0 when healthy. The dashboard: its /api/health answers. A worker: its heartbeat in the
    database is less than 30 s old."""
    from ppe_monitor.backend.settings import ServerSettings

    s = ServerSettings.load()
    if what == "dashboard":
        import base64
        import urllib.request

        req = urllib.request.Request(f"http://127.0.0.1:{s.dashboard_port}/api/health")
        pw = s.secrets.get("DASHBOARD_PASSWORD")
        if pw:
            token = base64.b64encode(f"{s.secrets.get('DASHBOARD_USER', 'safety')}:{pw}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return 0 if r.status == 200 else 1
        except Exception as exc:
            print(f"dashboard: {exc}")
            return 1
    from ppe_monitor.backend.store import EventStore, seconds_since

    try:
        store = EventStore(s.database_url, s.storage_dir, create=False)
        svc = store.services().get(what)
        store.close()
    except Exception as exc:
        print(f"{what}: database: {type(exc).__name__}")
        return 1
    age = seconds_since(svc["last_seen"]) if svc else None
    if age is None or age > 30:
        print(f"{what}: no heartbeat" + (f" for {age:.0f} s" if age is not None else " yet"))
        return 1
    return 0


def demo_cameras() -> int:
    """Stream the simulated cameras' clips (configs/cameras.yaml, sim_source) into the RTSP server,
    making the clips first if they are missing. Idles when no camera is simulated."""
    from ppe_monitor.config import ConfigError, load_cameras

    try:
        cams = [c for c in load_cameras() if c.is_simulated]
    except ConfigError as exc:
        print(f"configs/cameras.yaml: {exc}")
        return 1
    if not cams:
        print("No simulated cameras in configs/cameras.yaml: nothing to stream. (This service can be left out.)")
        while True:
            time.sleep(3600)
    if not os.environ.get("FFMPEG_BIN"):
        import imageio_ffmpeg

        os.environ["FFMPEG_BIN"] = imageio_ffmpeg.get_ffmpeg_exe()
    if any(not c.sim_source.is_file() for c in cams):
        print("Making the simulated camera clips (scripts/make_test_clips.py) ...")
        subprocess.run([sys.executable, "scripts/make_test_clips.py"], check=True)
    os.execv(sys.executable, [sys.executable, "scripts/fake_cameras.py"])
    return 0


def stand_in_telegram() -> int:
    from ppe_monitor.backend.standin import StandInTelegram

    port = int(os.environ.get("PORT", "8081"))
    tg = StandInTelegram(host="0.0.0.0", port=port)
    print(f"Stand-in for api.telegram.org on port {port}", flush=True)
    seen = 0
    while True:
        time.sleep(1)
        while seen < len(tg.received):
            m = tg.received[seen]
            seen += 1
            print(f"received {m['method']} for chat {m['chat']}: {m['bytes']} bytes"
                  + (", with a picture" if m["photo"] else ""), flush=True)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    what = sys.argv[1]
    if what == "secrets":
        return make_secrets()
    if what == "show-login":
        return show_login()
    if what == "health" and len(sys.argv) == 3:
        return health(sys.argv[2])
    if what == "demo-cameras":
        return demo_cameras()
    if what == "stand-in-telegram":
        return stand_in_telegram()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
