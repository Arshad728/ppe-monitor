"""A PostgreSQL server for this project alone, in data/postgres (used by scripts/db.py).

It listens only on 127.0.0.1, on port 5433 (configs/server.yaml), with a password that
`init` makes up and keeps in configs/secrets.env. Nothing is installed system-wide: on the Mac
the programs come from the project's conda environment (conda-forge's `postgresql`).
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..config import PROJECT_ROOT
from .settings import SECRETS_FILE, ServerSettings, write_env_value

DATA_DIR = PROJECT_ROOT / "data" / "postgres"
LOG_FILE = PROJECT_ROOT / "logs" / "postgres.log"
EXAMPLE = PROJECT_ROOT / "configs" / "secrets.env.example"


class PgError(RuntimeError):
    pass


def find_bin(name: str) -> str:
    """pg_ctl, initdb ...: from PATH, this Python's environment, or the usual install places."""
    cands = [shutil.which(name), str(Path(sys.prefix) / "bin" / name)]
    cands += sorted((str(p) for p in Path("/usr/lib/postgresql").glob(f"*/bin/{name}")), reverse=True)
    cands += [f"/opt/homebrew/opt/postgresql@16/bin/{name}", f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}",
              f"/Applications/Postgres.app/Contents/Versions/latest/bin/{name}"]
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise PgError(f"'{name}' not found. On the Mac: bash scripts/mac_phase5.sh setup (installs PostgreSQL "
                  "into the project environment from conda-forge).")


def _run(args: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if check and r.returncode != 0:
        raise PgError(f"{Path(args[0]).name} failed: {(r.stderr or r.stdout).strip()[-800:]}")
    return r


def initialised(data: Path = DATA_DIR) -> bool:
    return (data / "PG_VERSION").is_file()


def in_use(settings, data: Path = DATA_DIR) -> bool:
    """Whether the database is this project's own server in data/postgres (the Mac), rather than one
    given by DATABASE_URL or PPE_DB_HOST (Docker Compose, or a server elsewhere)."""
    return initialised(data) and not settings.secrets.get("DATABASE_URL") and not settings.secrets.get("PPE_DB_HOST")


def running(data: Path = DATA_DIR) -> bool:
    if not initialised(data):
        return False
    return _run([find_bin("pg_ctl"), "-D", str(data), "status"], check=False).returncode == 0


def init(settings: ServerSettings, data: Path = DATA_DIR, log=print) -> bool:
    """Create the server's files once. Returns False if they already existed."""
    if os.name != "nt" and os.geteuid() == 0:
        raise PgError("PostgreSQL refuses to run as root. Run this as your normal user.")
    if initialised(data):
        return False
    if not SECRETS_FILE.is_file():
        SECRETS_FILE.write_text(EXAMPLE.read_text(encoding="utf-8") if EXAMPLE.is_file() else "", encoding="utf-8")
    password = settings.secrets.get("PPE_DB_PASSWORD") or secrets.token_urlsafe(24)
    data.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".pw") as f:
        f.write(password)
        pwfile = f.name
    try:
        log(f"Creating the database server's files in {data} ...")
        _run([find_bin("initdb"), "-D", str(data), "-U", settings.db_user, f"--pwfile={pwfile}",
              "--auth-local=scram-sha-256", "--auth-host=scram-sha-256", "-E", "UTF8", "--no-locale"])
    finally:
        Path(pwfile).unlink(missing_ok=True)
    with (data / "postgresql.conf").open("a", encoding="utf-8") as f:
        f.write("\n# ppe-monitor (backend/localpg.py): this machine only\n"
                f"listen_addresses = '{settings.db_host}'\nport = {settings.db_port}\n"
                f"unix_socket_directories = '{data}'\nmax_connections = 40\n"
                "timezone = 'UTC'\nlog_timezone = 'UTC'\n")      # times are stored in UTC; the dashboard shows local time
    write_env_value(SECRETS_FILE, "PPE_DB_PASSWORD", password)
    settings.secrets._file["PPE_DB_PASSWORD"] = password
    log(f"  password saved as PPE_DB_PASSWORD in {SECRETS_FILE.relative_to(PROJECT_ROOT)}")
    return True


def start(data: Path = DATA_DIR, log=print) -> bool:
    """Start the server if it isn't running. Returns True if it was started now."""
    if not initialised(data):
        raise PgError(f"No database server in {data} yet. Run: python scripts/db.py init")
    if running(data):
        return False
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _run([find_bin("pg_ctl"), "-D", str(data), "-l", str(LOG_FILE), "-w", "-t", "60", "start"])
    log(f"Database server started (log: {LOG_FILE.relative_to(PROJECT_ROOT)})")
    return True


def stop(data: Path = DATA_DIR, log=print) -> bool:
    if not running(data):
        return False
    _run([find_bin("pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"])
    log("Database server stopped")
    return True


def ensure_database(settings: ServerSettings, name: str | None = None, *, fresh: bool = False, wait: float = 15.0) -> str:
    """Create the database `name` (default: database.name) if it's missing; `fresh` drops it first.
    Returns its SQLAlchemy URL."""
    import psycopg

    name = name or settings.db_name
    password = settings.secrets.get("PPE_DB_PASSWORD")
    end = time.monotonic() + wait
    while True:
        try:
            conn = psycopg.connect(host=settings.db_host, port=settings.db_port, user=settings.db_user,
                                   password=password, dbname="postgres", autocommit=True, connect_timeout=5)
            break
        except psycopg.OperationalError as exc:
            if time.monotonic() > end:
                raise PgError(f"can't reach PostgreSQL on {settings.db_host}:{settings.db_port}: {exc}") from exc
            time.sleep(0.5)
    with conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if exists and fresh:
            conn.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
            exists = None
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')
    base = settings.database_url.rsplit("/", 1)[0]
    return f"{base}/{name}"
