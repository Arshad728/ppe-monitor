"""Restricted zones: polygons drawn per camera, and "is this person standing inside?" (book, Chapter 10).

A zone is a polygon a safety officer draws once on a still frame from one camera
(scripts/draw_zones.py). Its corners are stored as fractions of the frame's width and height
(0-1), never in pixels, so a zone survives a change of stream resolution. They live in
configs/zones.yaml:

    cameras:
      cam1:
        drawn_on: configs/zones/cam1.jpg    # the frame the zones were drawn on (for re-checking)
        size: [1280, 720]                    # its size in pixels, for information only
        zones:
          - id: pit                          # short name used by rules and events
            name: Excavation pit             # shown on alerts
            polygon: [[0.04, 0.74], [0.40, 0.74], [0.46, 0.98], [0.02, 0.98]]

A person is inside a zone when the point where they stand is inside the polygon. That point is:
  - the average of their visible ankle keypoints, when the keypoint model found any: it stays
    right when the box is wrong, e.g. a hand stretched out or a second person merged in;
  - otherwise the bottom-centre of their box (the book's rule).
If the box reaches the bottom edge of the frame and no ankle is visible, the feet are out of view
and the answer is "unknown": no alert, and no "all clear" either.

The test itself is ray casting: from the point, count how many polygon edges a horizontal ray to
the right crosses; odd = inside. It works for any simple polygon, convex or not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT, ConfigError

ZONES_FILE = PROJECT_ROOT / "configs" / "zones.yaml"
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
EDGE = 0.005   # a box this close to the bottom of the frame may have its feet cut off


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    polygon: tuple[tuple[float, float], ...]   # corners, 0-1 fractions of the frame, in order

    def contains(self, point: tuple[float, float]) -> bool:
        return point_in_polygon(point, self.polygon)

    @property
    def area(self) -> float:
        return abs(_signed_area(self.polygon))


@dataclass
class CameraZones:
    camera: str
    zones: list[Zone] = field(default_factory=list)
    drawn_on: str | None = None    # reference frame, relative to the project folder
    size: tuple[int, int] | None = None

    def get(self, zone_id: str) -> Zone | None:
        return next((z for z in self.zones if z.id == zone_id), None)


def point_in_polygon(point: tuple[float, float], polygon) -> bool:
    """Ray casting: odd number of edge crossings to the right of the point = inside."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):                       # the edge spans the ray's height
            cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross_x:
                inside = not inside
    return inside


def _signed_area(polygon) -> float:
    return 0.5 * sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                     - polygon[(i + 1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)


def polygon_problem(polygon) -> str | None:
    """Why this polygon can't be a zone, or None if it is fine."""
    if len(polygon) < 3:
        return f"needs at least 3 corners, has {len(polygon)}"
    for x, y in polygon:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return (f"corner ({x}, {y}) is outside 0-1; corners are fractions of the frame's width and "
                    "height, not pixels")
    n = len(polygon)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue   # neighbouring edges share a corner
            if _segments_cross(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return f"its edges cross (edge {i + 1} and edge {j + 1}); click the corners in order around the zone"
    if abs(_signed_area(polygon)) < 1e-4:
        return "has (almost) no area"
    return None


def feet_point(box, pose=None, edge: float = EDGE) -> tuple[float, float] | None:
    """Where a person stands, as 0-1 frame fractions, or None if their feet are out of view.
    box: (x1, y1, x2, y2) fractions; pose: a vision.pose.Pose, if the keypoint model ran."""
    if pose is not None:
        ankles = pose.ankles
        if ankles is not None:
            return float(ankles[0]) / pose.aspect, float(ankles[1])
    if box[3] >= 1.0 - edge:
        return None
    return (box[0] + box[2]) / 2, box[3]


def zones_containing(zones: list[Zone], point) -> list[str]:
    return [] if point is None else [z.id for z in zones if z.contains(point)]


# -- the config file ---------------------------------------------------------------------------
def load_zones(path: str | Path = ZONES_FILE) -> dict[str, CameraZones]:
    """configs/zones.yaml -> {camera id: CameraZones}. A missing file means no zones."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    cams = raw.get("cameras") or {}
    if not isinstance(cams, dict):
        raise ConfigError(f"{path.name}: 'cameras' must map camera ids to their zones")
    out = {}
    for cam_id, entry in cams.items():
        entry = entry or {}
        zones, seen = [], set()
        for k, z in enumerate(entry.get("zones") or []):
            where = f"{path.name}, camera {cam_id}, zone {k + 1}"
            zid = str(z.get("id", ""))
            if not _ID.match(zid):
                raise ConfigError(f"{where}: id '{zid}' must be lowercase letters, digits, - or _")
            if zid in seen:
                raise ConfigError(f"{where}: id '{zid}' is used twice for this camera")
            seen.add(zid)
            try:
                poly = tuple((float(p[0]), float(p[1])) for p in z.get("polygon") or [])
            except (TypeError, ValueError, IndexError) as exc:
                raise ConfigError(f"{where} ({zid}): polygon must be a list of [x, y] pairs") from exc
            problem = polygon_problem(poly)
            if problem:
                raise ConfigError(f"{where} ({zid}): the polygon {problem}")
            zones.append(Zone(zid, str(z.get("name") or zid), poly))
        size = entry.get("size")
        out[str(cam_id)] = CameraZones(str(cam_id), zones, entry.get("drawn_on"),
                                       tuple(int(v) for v in size) if size else None)
    return out


def save_zones(all_zones: dict[str, CameraZones], path: str | Path = ZONES_FILE) -> None:
    """Write configs/zones.yaml, keeping the explanatory header."""
    path = Path(path)
    header = ("# Restricted zones, per camera. Draw or edit them with: python scripts/draw_zones.py --camera <id>\n"
              "# Corners are fractions (0-1) of the frame's width and height, so they survive a resolution change.\n"
              "# Rules that use these zones (who may not enter, when, how urgent) are in configs/rules.yaml.\n\n")
    data = {"cameras": {}}
    for cam_id, cz in sorted(all_zones.items()):
        entry = {}
        if cz.drawn_on:
            entry["drawn_on"] = cz.drawn_on
        if cz.size:
            entry["size"] = _Flow(cz.size)
        entry["zones"] = [{"id": z.id, "name": z.name,
                           "polygon": _Flow([round(x, 4), round(y, 4)] for x, y in z.polygon)} for z in cz.zones]
        data["cameras"][cam_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.dump(data, Dumper=_Dumper, sort_keys=False, width=200), encoding="utf-8")


class _Flow(list):
    """A list written on one line: polygon: [[0.1, 0.2], [0.3, 0.4], ...]"""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Flow, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
