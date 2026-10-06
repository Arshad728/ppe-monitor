"""Phase 5 settings: configs/server.yaml, plus secrets from configs/secrets.env and the environment.

Secrets (tokens, passwords) are never in server.yaml. They are read from configs/secrets.env, a
KEY=value file kept out of git, and environment variables of the same name override it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import yaml

from ..config import PROJECT_ROOT, ConfigError

SERVER_CONFIG = PROJECT_ROOT / "configs" / "server.yaml"
SECRETS_FILE = PROJECT_ROOT / "configs" / "secrets.env"
SEVERITIES = ("low", "medium", "high", "critical")
CHANNELS = ("telegram", "email")
_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines; blank lines and # comments ignored; optional quotes removed."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            value = m.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            out[m.group(1)] = value
    return out


def write_env_value(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a secrets file, keeping every other line (and comments) as they are."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    done = False
    for i, line in enumerate(lines):
        m = _LINE.match(line)
        if m and m.group(1) == key and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={value}"
            done = True
            break
    if not done:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)          # only this user can read it
    except OSError:
        pass


class Secrets:
    """Lookup order: environment variable, then a file named by <KEY>_FILE in the environment (how
    Docker hands over secrets without putting them in the environment), then configs/secrets.env."""

    def __init__(self, path: Path = SECRETS_FILE, environ: dict | None = None):
        self.path = path
        self._file = read_env_file(path)
        self._env = os.environ if environ is None else environ

    def get(self, key: str, default: str = "") -> str:
        v = self._env.get(key)
        if v not in (None, ""):
            return v
        f = self._env.get(f"{key}_FILE")
        if f:
            try:
                v = Path(f).read_text(encoding="utf-8").strip()
            except OSError:
                v = ""
            if v:
                return v
        v = self._file.get(key)
        return v if v not in (None, "") else default

    def list(self, key: str) -> list[str]:
        return [p.strip() for p in self.get(key).split(",") if p.strip()]


@dataclass
class ChannelSettings:
    name: str
    enabled: bool = False
    min_severity: str = "medium"
    photo: bool = True              # send the evidence image (false: text only; the image stays on the dashboard)
    ip: str = "ipv4"                # telegram: ipv4 (see channels.http_client) or any

    def wants(self, severity: str) -> bool:
        s = severity if severity in SEVERITIES else "medium"
        return self.enabled and SEVERITIES.index(s) >= SEVERITIES.index(self.min_severity)


@dataclass
class ServerSettings:
    db_host: str = "127.0.0.1"
    db_port: int = 5433
    db_name: str = "ppe"
    db_user: str = "ppe"
    storage_dir: Path = PROJECT_ROOT / "data" / "events"
    save_frame: bool = True
    keep_days: float = 30
    live_dir: Path = PROJECT_ROOT / "runs" / "live_tiles"
    live_every: float = 1.0
    live_width: int = 640
    poll: float = 0.5
    retry_attempts: int = 6
    retry_first: float = 2.0
    retry_longest: float = 120.0
    rate_max: int = 5
    rate_minutes: float = 10.0
    channels: dict[str, ChannelSettings] = field(default_factory=dict)
    dashboard_url: str = ""
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080
    camera_down_after: float = 60.0      # Phase 6: seconds a camera may be down before an alert (0: never)
    service_silent_after: float = 60.0   # ... and the camera service may go without a heartbeat
    secrets: Secrets = field(default_factory=Secrets)

    @classmethod
    def load(cls, path: str | Path = SERVER_CONFIG, secrets: Secrets | None = None) -> "ServerSettings":
        path = Path(path)
        try:
            raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path.name}: not valid YAML ({exc})") from exc
        db, st, live, al, dash = (raw.get(k) or {} for k in ("database", "storage", "live", "alerts", "dashboard"))
        retry, rate, health = al.get("retry") or {}, al.get("rate_limit") or {}, al.get("health") or {}
        secrets = secrets or Secrets()
        channels = {}
        for name in CHANNELS:
            c = (al.get("channels") or {}).get(name) or {}
            sev = str(c.get("min_severity", "medium"))
            if sev not in SEVERITIES:
                raise ConfigError(f"{path.name}: alerts.channels.{name}.min_severity must be one of {', '.join(SEVERITIES)}")
            ip = str(c.get("ip", "ipv4"))
            if ip not in ("ipv4", "any"):
                raise ConfigError(f"{path.name}: alerts.channels.{name}.ip must be ipv4 or any")
            channels[name] = ChannelSettings(name, bool(c.get("enabled", False)), sev, bool(c.get("photo", True)), ip)
        unknown = set((al.get("channels") or {})) - set(CHANNELS)
        if unknown:
            raise ConfigError(f"{path.name}: unknown alert channels {sorted(unknown)} (known: {', '.join(CHANNELS)})")

        def where(p, default):
            p = Path(p or default)
            return p if p.is_absolute() else PROJECT_ROOT / p

        # PPE_DB_HOST / PPE_DB_PORT / DASHBOARD_HOST in the environment win (Docker Compose sets them)
        s = cls(
            db_host=secrets.get("PPE_DB_HOST") or str(db.get("host", "127.0.0.1")),
            db_port=int(secrets.get("PPE_DB_PORT") or db.get("port", 5433)),
            db_name=str(db.get("name", "ppe")), db_user=str(db.get("user", "ppe")),
            storage_dir=where(st.get("dir"), "data/events"), save_frame=bool(st.get("save_frame", True)),
            keep_days=float(st.get("keep_days", 30)),
            live_dir=where(live.get("dir"), "runs/live_tiles"), live_every=float(live.get("every", 1.0)),
            live_width=int(live.get("width", 640)),
            poll=float(al.get("poll", 0.5)), retry_attempts=int(retry.get("attempts", 6)),
            retry_first=float(retry.get("first_wait", 2)), retry_longest=float(retry.get("longest_wait", 120)),
            rate_max=int(rate.get("max_alerts", 5)), rate_minutes=float(rate.get("minutes", 10)),
            channels=channels, dashboard_url=str(al.get("dashboard_url", "") or "").rstrip("/"),
            dashboard_host=secrets.get("DASHBOARD_HOST") or str(dash.get("host", "127.0.0.1")),
            dashboard_port=int(dash.get("port", 8080)),
            camera_down_after=float(health.get("camera_down_after", 60)),
            service_silent_after=float(health.get("service_silent_after", 60)),
            secrets=secrets,
        )
        if s.rate_max < 1 or s.rate_minutes <= 0 or s.retry_attempts < 1 or s.poll <= 0:
            raise ConfigError(f"{path.name}: rate_limit, retry and poll must be positive")
        return s

    @property
    def database_url(self) -> str:
        url = self.secrets.get("DATABASE_URL")
        if url:
            if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
                rel = url[len("sqlite:///"):]
                url = f"sqlite:///{PROJECT_ROOT / rel}" if rel else url   # relative to the project
            return url
        password = self.secrets.get("PPE_DB_PASSWORD")
        auth = f"{quote(self.db_user)}:{quote(password, safe='')}" if password else quote(self.db_user)
        return f"postgresql+psycopg://{auth}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def safe_database_url(self) -> str:
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:****@", self.database_url)
