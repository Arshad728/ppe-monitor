"""Loading and validating the camera registry (configs/cameras.yaml)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERAS_FILE = PROJECT_ROOT / "configs" / "cameras.yaml"
SECRETS_FILE = PROJECT_ROOT / "configs" / "secrets.env"

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_VALID_TRANSPORTS = {"tcp", "udp"}
_UNEXPANDED_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    url: str
    transport: str = "tcp"
    enabled: bool = True
    # Local video file looped into `url` by the fake camera network (development only).
    sim_source: Path | None = None

    @property
    def is_simulated(self) -> bool:
        return self.sim_source is not None

    @property
    def safe_url(self) -> str:
        """The URL with any password replaced, for printing and logs."""
        parsed = urlparse(self.url)
        if parsed.password is None:
            return self.url
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":****@", 1)
        return parsed._replace(netloc=netloc).geturl()


def _secret_values(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of configs/secrets.env (comments and blank lines skipped)."""
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _expand(url: str, values: dict[str, str]) -> str:
    """$VAR and ${VAR} replaced from `values`; unknown ones are left as they are (and then refused)."""
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                  lambda m: values.get(m.group(1) or m.group(2), m.group(0)), url)


def load_cameras(path: str | Path = DEFAULT_CAMERAS_FILE, *, include_disabled: bool = False,
                 secrets: str | Path | None = SECRETS_FILE) -> list[Camera]:
    """Read the camera registry and return validated Camera objects.

    `${VAR}` references in URLs are expanded from the environment, or else from
    configs/secrets.env, so camera passwords never have to be written in this file.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Camera config not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    values = {**(_secret_values(Path(secrets)) if secrets else {}), **os.environ}   # the environment wins
    entries = raw.get("cameras")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path} must contain a non-empty 'cameras:' list")

    cameras: list[Camera] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"{path.name}, camera #{index + 1}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: each camera must be a mapping of fields")

        cam_id = str(entry.get("id", "")).strip()
        if not _ID_PATTERN.match(cam_id):
            raise ConfigError(
                f"{where}: id {cam_id!r} must be 1-32 chars of lowercase letters, digits, '-' or '_'"
            )
        if cam_id in seen_ids:
            raise ConfigError(f"{where}: duplicate camera id {cam_id!r}")
        seen_ids.add(cam_id)

        url = _expand(str(entry.get("url", "")).strip(), values)
        if _UNEXPANDED_VAR.search(url):
            raise ConfigError(f"{where}: url uses a variable that is set neither in the environment nor in configs/secrets.env")
        scheme = urlparse(url).scheme.lower()
        if scheme not in {"rtsp", "rtsps"}:
            raise ConfigError(f"{where}: url must start with rtsp:// (got {scheme or 'nothing'!r})")

        transport = str(entry.get("transport", "tcp")).lower()
        if transport not in _VALID_TRANSPORTS:
            raise ConfigError(f"{where}: transport must be 'tcp' or 'udp' (got {transport!r})")

        sim_source = entry.get("sim_source")
        sim_path: Path | None = None
        if sim_source:
            sim_path = Path(sim_source)
            if not sim_path.is_absolute():
                sim_path = PROJECT_ROOT / sim_path
            host = (urlparse(url).hostname or "").lower()
            if host not in {"localhost", "127.0.0.1", "::1"}:
                raise ConfigError(
                    f"{where}: sim_source is set, so url must point at the local fake camera "
                    f"server (localhost), not {host!r}"
                )

        camera = Camera(
            id=cam_id,
            name=str(entry.get("name") or cam_id),
            url=url,
            transport=transport,
            enabled=bool(entry.get("enabled", True)),
            sim_source=sim_path,
        )
        if camera.enabled or include_disabled:
            cameras.append(camera)

    return cameras


def get_camera(camera_id: str, path: str | Path = DEFAULT_CAMERAS_FILE) -> Camera:
    for camera in load_cameras(path, include_disabled=True):
        if camera.id == camera_id:
            return camera
    known = ", ".join(c.id for c in load_cameras(path, include_disabled=True))
    raise ConfigError(f"No camera with id {camera_id!r} in {path}. Known ids: {known}")
