"""The rules engine: which conditions raise an event, where, when and how urgently (book, Chapter 20).

Rules are data, not code. Each entry in configs/rules.yaml says:

    - id: pit-no-entry          # short unique name; appears in every event it raises
      type: zone_intrusion      # no_helmet | no_vest | zone_intrusion
      cameras: [cam3]           # or "all"
      zone: pit                 # a zone of those cameras (configs/zones.yaml)
      severity: critical        # low | medium | high | critical
      dwell: 2.0                # seconds the condition must last (default: configs/ppe.yaml events.dwell)
      cooldown: 60.0            # seconds before the same person can raise it again (default: events.cooldown)
      active:                   # optional; without it the rule is always on
        hours: "07:00-19:00"    # one or more ranges, "22:00-06:00" runs past midnight
        days: mon-sat           # optional: mon-fri, "mon,wed,fri", all

What each type means for one person in one frame:
  no_helmet / no_vest   the PPE rule's verdict (rules/ppe.py). With `zone:`, only inside that zone
                        ("helmets required on the scaffold"): outside it the person counts as compliant.
  zone_intrusion        standing inside the zone (rules/zones.py) is the violation.
Every type then goes through the same event engine (rules/events.py): the vote, the dwell, one event
per episode, the cooldown, one incident even if the person's track ID changes. So PPE and zone events
come out in the same shape.

Outside a rule's active hours the rule is off: nothing is counted, and anyone already being
confirmed starts again from zero when it comes back on. Times are the machine's local time, or
`timezone:` at the top of the file (e.g. Asia/Kolkata).

Without configs/rules.yaml, every camera gets the two Phase 2 rules: no helmet and no vest, anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from ..config import PROJECT_ROOT, ConfigError
from .ppe import MISSING, UNKNOWN, Verdict
from .zones import ZONES_FILE, CameraZones, Zone, feet_point, load_zones

RULES_FILE = PROJECT_ROOT / "configs" / "rules.yaml"
PPE_ITEM = {"no_helmet": "helmet", "no_vest": "vest"}
TYPES = (*PPE_ITEM, "zone_intrusion")
SEVERITIES = ("low", "medium", "high", "critical")
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_RULE_KEYS = {"id", "type", "cameras", "zone", "severity", "dwell", "cooldown", "active", "enabled", "description"}


# -- active hours ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    days: frozenset          # weekday numbers, Monday = 0
    start: time
    end: time                # end < start: runs past midnight

    def contains(self, dt: datetime) -> bool:
        t, day = dt.time(), dt.weekday()
        if self.start < self.end:
            return day in self.days and self.start <= t < self.end
        if self.start == self.end:          # "00:00-00:00" = the whole day
            return day in self.days
        return (day in self.days and t >= self.start) or ((day - 1) % 7 in self.days and t < self.end)


@dataclass(frozen=True)
class Schedule:
    windows: tuple[Window, ...]
    text: str = ""

    def active(self, dt: datetime) -> bool:
        return any(w.contains(dt) for w in self.windows)


def _parse_days(text) -> frozenset:
    text = str(text or "all").strip().lower()
    if text in ("all", "daily", "every day"):
        return frozenset(range(7))
    days = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (p.strip()[:3] for p in part.split("-", 1))
            if a not in DAYS or b not in DAYS:
                raise ValueError(f"unknown day range '{part}'")
            i, j = DAYS.index(a), DAYS.index(b)
            days.update(DAYS.index(DAYS[(i + k) % 7]) for k in range((j - i) % 7 + 1))
        else:
            if part[:3] not in DAYS:
                raise ValueError(f"unknown day '{part}'")
            days.add(DAYS.index(part[:3]))
    return frozenset(days)


def _parse_time(text: str) -> time:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", text)
    if not m or int(m.group(1)) > 24 or int(m.group(2)) > 59:
        raise ValueError(f"'{text}' is not a time like 07:30")
    h, mi = int(m.group(1)), int(m.group(2))
    return time(0, 0) if h == 24 else time(h, mi)


def parse_schedule(spec) -> Schedule | None:
    """`active:` of a rule -> Schedule; None = always on. Accepts {hours, days}, a list of those, or
    "always"."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "always"):
        return None
    specs = spec if isinstance(spec, list) else [spec]
    windows = []
    for s in specs:
        if not isinstance(s, dict) or "hours" not in s or set(s) - {"hours", "days"}:
            raise ValueError("active must be {hours: \"07:00-19:00\", days: mon-sat} (days optional)")
        days = _parse_days(s.get("days"))
        hours = s["hours"] if isinstance(s["hours"], list) else str(s["hours"]).split(",")
        for h in hours:
            if "-" not in str(h):
                raise ValueError(f"hours '{h}' must be a range like 07:00-19:00")
            a, b = str(h).split("-", 1)
            windows.append(Window(days, _parse_time(a), _parse_time(b)))
    text = "; ".join(f"{s['hours']}" + (f" {s['days']}" if s.get("days") else "") for s in specs)
    return Schedule(tuple(windows), text)


# -- rules -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    type: str
    cameras: tuple[str, ...] | None = None   # None = all cameras
    zone: str | None = None
    severity: str = "medium"
    dwell: float | None = None               # None = the event settings' default
    cooldown: float | None = None
    schedule: Schedule | None = None
    description: str = ""

    def applies_to(self, camera: str) -> bool:
        return self.cameras is None or camera in self.cameras

    def active_at(self, now: datetime | None) -> bool:
        return self.schedule is None or now is None or self.schedule.active(now)


DEFAULT_RULES = (Rule("no_helmet", "no_helmet", severity="high"), Rule("no_vest", "no_vest", severity="medium"))


@dataclass
class CameraRules:
    """The rules and zones of one camera, and what they say about each person in each frame."""
    camera: str
    rules: list[Rule]
    zones: list[Zone] = field(default_factory=list)
    tz: ZoneInfo | None = None

    def zone(self, zone_id: str | None) -> Zone | None:
        return next((z for z in self.zones if z.id == zone_id), None)

    def local_time(self, now: datetime | None) -> datetime | None:
        if now is None:
            return None
        if now.tzinfo is None:
            now = now.astimezone()
        return now.astimezone(self.tz) if self.tz else now.astimezone()

    def active(self, now: datetime | None) -> set[str]:
        local = self.local_time(now)
        return {r.id for r in self.rules if r.active_at(local)}

    def votes(self, verdict: Verdict, box, pose=None, active: set[str] | None = None):
        """One person, one frame -> ({rule id: True violating / False fine / None can't tell},
        feet point, ids of the zones they stand in)."""
        feet = feet_point(box, pose) if self.zones else None
        inside = {z.id for z in self.zones if feet is not None and z.contains(feet)}
        out = {}
        for r in self.rules:
            if active is not None and r.id not in active:
                continue
            if r.zone is not None and feet is None:
                out[r.id] = None
            elif r.type == "zone_intrusion":
                out[r.id] = r.zone in inside
            elif r.zone is not None and r.zone not in inside:
                out[r.id] = False             # PPE only required inside the zone
            else:
                answer = getattr(verdict, PPE_ITEM[r.type])
                out[r.id] = None if answer == UNKNOWN else answer == MISSING
        return out, feet, sorted(inside)


@dataclass
class RuleSet:
    rules: list[Rule]
    zones: dict[str, CameraZones] = field(default_factory=dict)
    timezone: str | None = None
    source: str = "built-in defaults"

    @classmethod
    def default(cls) -> "RuleSet":
        return cls(list(DEFAULT_RULES))

    @classmethod
    def load(cls, rules_path: str | Path = RULES_FILE, zones_path: str | Path = ZONES_FILE) -> "RuleSet":
        """configs/rules.yaml + configs/zones.yaml, checked. Without a rules file: the defaults."""
        zones = load_zones(zones_path)
        rules_path = Path(rules_path)
        if not rules_path.is_file():
            return cls(list(DEFAULT_RULES), zones)
        try:
            raw = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{rules_path.name} is not valid YAML: {exc}") from exc
        unknown = set(raw) - {"timezone", "rules"}
        if unknown:
            raise ConfigError(f"{rules_path.name}: unknown top-level keys {sorted(unknown)} (allowed: timezone, rules)")
        tz = raw.get("timezone")
        if tz:
            try:
                ZoneInfo(str(tz))
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ConfigError(f"{rules_path.name}: unknown timezone '{tz}' (e.g. Asia/Kolkata)") from exc
        rules, seen = [], set()
        for k, entry in enumerate(raw.get("rules") or []):
            rule = _parse_rule(entry, f"{rules_path.name}, rule {k + 1}", zones)
            if rule is None:
                continue
            if rule.id in seen:
                raise ConfigError(f"{rules_path.name}: rule id '{rule.id}' is used twice")
            seen.add(rule.id)
            rules.append(rule)
        return cls(rules, zones, str(tz) if tz else None, str(rules_path.name))

    def for_camera(self, camera: str) -> CameraRules:
        cz = self.zones.get(camera)
        return CameraRules(camera, [r for r in self.rules if r.applies_to(camera)], list(cz.zones) if cz else [],
                           ZoneInfo(self.timezone) if self.timezone else None)


def _parse_rule(entry, where: str, zones: dict[str, CameraZones]) -> Rule | None:
    if not isinstance(entry, dict):
        raise ConfigError(f"{where}: each rule is a set of 'key: value' lines")
    unknown = set(entry) - _RULE_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)} (allowed: {', '.join(sorted(_RULE_KEYS))})")
    if entry.get("enabled", True) is False:
        return None
    rid = str(entry.get("id", ""))
    if not _ID.match(rid):
        raise ConfigError(f"{where}: id '{rid}' must be lowercase letters, digits, - or _")
    where = f"{where} ({rid})"
    rtype = entry.get("type")
    if rtype not in TYPES:
        raise ConfigError(f"{where}: type must be one of {', '.join(TYPES)}, not '{rtype}'")
    cams = entry.get("cameras", "all")
    if cams == "all":
        cameras = None
    elif isinstance(cams, str):
        cameras = (cams,)
    elif isinstance(cams, list) and cams and all(isinstance(c, str) for c in cams):
        cameras = tuple(cams)
    else:
        raise ConfigError(f"{where}: cameras must be 'all', a camera id, or a list of camera ids")
    zone = entry.get("zone")
    if rtype == "zone_intrusion" and not zone:
        raise ConfigError(f"{where}: a zone_intrusion rule needs a zone")
    if zone is not None:
        zone = str(zone)
        if cameras is None:
            raise ConfigError(f"{where}: a rule with a zone must list its cameras (zones belong to one camera)")
        for cam in cameras:
            if cam not in zones or zones[cam].get(zone) is None:
                have = ", ".join(z.id for z in zones[cam].zones) if cam in zones and zones[cam].zones else "none"
                raise ConfigError(f"{where}: camera {cam} has no zone '{zone}' in zones.yaml (it has: {have}). "
                                  f"Draw it with: python scripts/draw_zones.py --camera {cam}")
    severity = str(entry.get("severity", "medium"))
    if severity not in SEVERITIES:
        raise ConfigError(f"{where}: severity must be one of {', '.join(SEVERITIES)}")
    nums = {}
    for key in ("dwell", "cooldown"):
        if entry.get(key) is not None:
            try:
                nums[key] = float(entry[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{where}: {key} must be a number of seconds") from exc
            if nums[key] < 0:
                raise ConfigError(f"{where}: {key} can't be negative")
    try:
        schedule = parse_schedule(entry.get("active"))
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}") from exc
    return Rule(rid, rtype, cameras, zone, severity, nums.get("dwell"), nums.get("cooldown"), schedule,
                str(entry.get("description", "")))
