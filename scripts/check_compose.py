#!/usr/bin/env python3
"""Phase 6's "done when": `docker compose up` brings up the whole stack. This checks it.

    python scripts/check_compose.py              # ~10 min the first time (builds the image), ~2 min after
    python scripts/check_compose.py --keep       # leave the stack running afterwards, to look at it
    bash scripts/mac_phase6.sh docker            # the same, on the Mac (needs Docker Desktop, OrbStack or Colima)

It starts the project's docker-compose.yml as a separate Compose project ("ppe-check": its own
database volume, its own network), with three changes for the check only:
  - a stand-in for Telegram, so alerts are delivered without reaching anyone's phone;
  - the dashboard on 127.0.0.1:18080, so it can't clash with one already running on 8080;
  - evidence images and live pictures in runs/compose_check/<time>/, not in data/ and runs/.
Then it waits for every service to be healthy and checks, through the dashboard's API with its
password, that the three simulated cameras are live, that violations become events, and that their
alerts are delivered, with pictures. Finally it stops the stack and removes its volumes.
Report: runs/compose_check/<time>/report.md
"""

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "ppe-check"
PORT = 18080


def compose(*args, check=True, capture=True, override: Path | None = None, timeout=None):
    cmd = ["docker", "compose", "-p", PROJECT, "-f", str(ROOT / "docker-compose.yml")]
    if override:
        cmd += ["-f", str(override)]
    r = subprocess.run(cmd + list(args), cwd=ROOT, capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed:\n{(r.stderr or r.stdout)[-2000:]}")
    return r


def services(override) -> dict[str, dict]:
    out = compose("ps", "-a", "--format", "json", override=override).stdout.strip()
    rows = json.loads(out) if out.startswith("[") else [json.loads(line) for line in out.splitlines() if line.strip()]
    return {r["Service"]: r for r in rows}


def api(path: str, password: str | None, user: str = "safety"):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    if password:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, None


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    parser.add_argument("--no-build", action="store_true", help="use the image already built")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for everything to be healthy")
    args = parser.parse_args()
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("Docker isn't running (or isn't installed). On a Mac: install Docker Desktop, OrbStack or Colima, start it, "
              "and run this again.")
        return 1

    out = ROOT / "runs" / "compose_check" / f"{datetime.now():%Y%m%d_%H%M%S}"
    (out / "events").mkdir(parents=True)
    (out / "live").mkdir()
    override = out / "override.yml"
    override.write_text(f"""# written by scripts/check_compose.py for the check only
services:
  stand-in-telegram:
    image: ppe-monitor:latest
    command: ["python", "scripts/docker_helper.py", "stand-in-telegram"]
    depends_on:
      secrets: {{condition: service_completed_successfully}}
  alerts:
    environment:
      TELEGRAM_API_BASE: http://stand-in-telegram:8081
      TELEGRAM_BOT_TOKEN: "000:COMPOSE-CHECK"
      TELEGRAM_CHAT_ID: "1"
    volumes:
      - {out / 'events'}:/app/data/events
      - {out / 'live'}:/app/runs/live_tiles
  cameras:
    volumes:
      - {out / 'events'}:/app/data/events
      - {out / 'live'}:/app/runs/live_tiles
  dashboard:
    ports: !override
      - "127.0.0.1:{PORT}:8080"
    volumes:
      - {out / 'events'}:/app/data/events
      - {out / 'live'}:/app/runs/live_tiles
""", encoding="utf-8")

    t0 = time.monotonic()
    checks = []
    try:
        print(f"Starting the stack as Compose project '{PROJECT}'" + ("" if args.no_build else " (building the image first)") + " ...")
        compose("up", "-d", *([] if args.no_build else ["--build"]), override=override, capture=False, timeout=args.timeout + 600)
        need = ("cameras", "alerts", "dashboard")
        once = ("secrets", "db-init")
        state = {}
        while time.monotonic() - t0 < args.timeout:
            state = services(override)
            bad_once = [s for s in once if state.get(s, {}).get("State") == "exited" and state[s].get("ExitCode", 0) != 0]
            if bad_once:
                raise RuntimeError(f"{', '.join(bad_once)} failed:\n" + compose("logs", *bad_once, override=override).stdout[-2000:])
            if all(state.get(s, {}).get("Health") == "healthy" for s in need) and \
                    all(state.get(s, {}).get("State") == "exited" for s in once):
                break
            time.sleep(5)
        up_s = time.monotonic() - t0
        health = {s: f"{state.get(s, {}).get('State', '?')}/{state.get(s, {}).get('Health') or '-'}" for s in state}
        ok = all(state.get(s, {}).get("Health") == "healthy" for s in need)
        checks.append(("every service up; cameras, alerts and dashboard healthy", ok,
                       ", ".join(f"{k} {v}" for k, v in sorted(health.items())) + f" ({up_s:.0f} s)"))
        if not ok:
            raise RuntimeError("not healthy in time:\n" + compose("logs", "--tail", "40", override=override).stdout[-4000:])

        password = compose("exec", "-T", "dashboard", "cat", "/run/ppe-secrets/dashboard_password",
                           override=override).stdout.strip()
        code, _ = api("/api/health", None)
        checks.append(("the dashboard asks for a password", code == 401, f"without it: HTTP {code}"))
        code, h = api("/api/health", password)
        checks.append(("the dashboard answers with the password; database reachable", code == 200 and bool(h and h.get("ok")),
                       f"HTTP {code}, database {h and h.get('database')}"))

        deadline = time.monotonic() + 150
        ov = {}
        while time.monotonic() < deadline:
            _, ov = api("/api/overview?hours=1", password)
            live = [c for c in (ov or {}).get("cameras", []) if c.get("state") == "live"]
            if ov and len(live) >= 3 and ov["stats"]["total"] >= 1 and (ov["alerts"] or {}).get("sent", 0) >= 1:
                break
            time.sleep(5)
        cams = (ov or {}).get("cameras", [])
        checks.append(("the three simulated cameras are live over RTSP", sum(c.get("state") == "live" for c in cams) >= 3,
                       ", ".join(f"{c['id']} {c['state']} {c.get('fps') or 0:.1f} fps" for c in cams)))
        total = (ov or {}).get("stats", {}).get("total", 0)
        checks.append(("violations on the cameras become events in PostgreSQL", total >= 1,
                       f"{total} event(s): " + json.dumps((ov or {}).get("stats", {}).get("by_kind", {}))))
        sent = ((ov or {}).get("alerts") or {}).get("sent", 0)
        for _ in range(10):                       # the stand-in prints each message within a second
            photos = compose("logs", "stand-in-telegram", override=override).stdout.count("with a picture")
            if photos >= sent:
                break
            time.sleep(1)
        checks.append(("their alerts are delivered, with the evidence picture", sent >= 1 and photos >= 1,
                       f"{sent} alert(s) sent; the stand-in Telegram received {photos} with a picture"))
        _, items = api("/api/events?limit=1&since=2000-01-01T00:00:00Z", password)
        first = (items or {}).get("items", [{}])
        eid = first[0].get("id") if first else None
        if eid:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/events/{eid}/snapshot.jpg")
            req.add_header("Authorization", "Basic " + base64.b64encode(f"safety:{password}".encode()).decode())
            with urllib.request.urlopen(req, timeout=10) as r:
                img = r.read()
            checks.append(("the dashboard serves the evidence image", img[:2] == b"\xff\xd8", f"{len(img) // 1024} KB JPEG"))
    except Exception as exc:
        checks.append(("the stack ran", False, str(exc)[:1500]))
    finally:
        if args.keep:
            print(f"\nLeft running. Dashboard: http://127.0.0.1:{PORT} (user safety). Stop it with:\n"
                  f"  docker compose -p {PROJECT} -f docker-compose.yml -f {override} down -v")
        else:
            compose("down", "-v", "--remove-orphans", override=override, check=False, timeout=180)

    passed = bool(checks) and all(ok for _, ok, _ in checks)
    lines = [f"# Docker Compose check: {datetime.now():%Y-%m-%d %H:%M}", "",
             "`docker compose up` of the project's docker-compose.yml (as project `ppe-check`, with a stand-in for Telegram), "
             f"then the dashboard's API. Machine: {subprocess.run(['docker', 'version', '--format', '{{.Server.Os}}/{{.Server.Arch}} Docker {{.Server.Version}}'], capture_output=True, text=True).stdout.strip()}.",
             "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {w} | {'PASS' if ok else 'FAIL'} | {d.replace('|', '/').replace(chr(10), ' ')} |" for w, ok, d in checks]
    lines += ["", f"**{'All checks passed.' if passed else 'Some checks failed.'}**"]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\nReport: {out / 'report.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
