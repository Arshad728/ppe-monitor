"""The state behind scripts/draw_zones.py: corners being clicked, zones finished, what to save.

Kept apart from the OpenCV window so it can be tested without a screen. Clicks arrive in pixels
of the frame on screen; zones are stored as fractions of that frame (rules/zones.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .zones import _ID, CameraZones, Zone, polygon_problem

CYAN, YELLOW, WHITE, BLACK, RED = (230, 200, 0), (0, 220, 255), (255, 255, 255), (0, 0, 0), (40, 40, 230)
HELP = ["click: add a corner    right-click / u: remove the last corner",
        "Enter: finish this zone (name it in the Terminal)    d: delete a zone",
        "s: save    q: quit    h: hide this help"]


@dataclass
class ZoneEditor:
    camera: str
    width: int
    height: int
    zones: list[Zone] = field(default_factory=list)
    corners: list[tuple[int, int]] = field(default_factory=list)   # the zone being drawn, in pixels
    changed: bool = False

    def add(self, x: int, y: int) -> None:
        self.corners.append((min(max(int(x), 0), self.width - 1), min(max(int(y), 0), self.height - 1)))

    def undo(self) -> None:
        if self.corners:
            self.corners.pop()

    def polygon(self) -> tuple[tuple[float, float], ...]:
        """The corners so far as 0-1 fractions of the frame."""
        return tuple((round(x / self.width, 4), round(y / self.height, 4)) for x, y in self.corners)

    def problem(self) -> str | None:
        return polygon_problem(self.polygon())

    def finish(self, zone_id: str, name: str = "") -> Zone:
        """Turn the clicked corners into a zone. Raises ValueError with a message for the user."""
        zone_id = zone_id.strip().lower()
        if not _ID.match(zone_id):
            raise ValueError("the id must be lowercase letters, digits, - or _ (e.g. pit, crane-2)")
        if any(z.id == zone_id for z in self.zones):
            raise ValueError(f"camera {self.camera} already has a zone '{zone_id}'")
        problem = self.problem()
        if problem:
            raise ValueError(f"this zone {problem}")
        zone = Zone(zone_id, name.strip() or zone_id, self.polygon())
        self.zones.append(zone)
        self.corners.clear()
        self.changed = True
        return zone

    def delete(self, zone_id: str) -> bool:
        before = len(self.zones)
        self.zones = [z for z in self.zones if z.id != zone_id.strip()]
        self.changed |= len(self.zones) != before
        return len(self.zones) != before

    def result(self, drawn_on: str | None) -> CameraZones:
        return CameraZones(self.camera, list(self.zones), drawn_on, (self.width, self.height))

    def render(self, frame: np.ndarray, show_help: bool = True) -> np.ndarray:
        img = frame.copy()
        for z in self.zones:
            pts = np.array([[int(x * self.width), int(y * self.height)] for x, y in z.polygon], np.int32)
            tint = img.copy()
            cv2.fillPoly(tint, [pts], CYAN)
            cv2.addWeighted(tint, 0.2, img, 0.8, 0, img)
            cv2.polylines(img, [pts], True, CYAN, 2, cv2.LINE_AA)
            top = pts[np.argmin(pts[:, 1])]
            _text(img, f"{z.id}: {z.name}", int(top[0]), max(18, int(top[1]) - 6), CYAN)
        if self.corners:
            pts = np.array(self.corners, np.int32)
            cv2.polylines(img, [pts], len(self.corners) > 2, YELLOW, 2, cv2.LINE_AA)
            for p in self.corners:
                cv2.circle(img, p, 5, YELLOW, -1)
            problem = self.problem() if len(self.corners) >= 3 else None
            if problem:
                _text(img, f"not a valid zone yet: {problem}", 10, self.height - 12, RED)
        if show_help:
            for k, line in enumerate([f"camera {self.camera}: {len(self.zones)} zone(s)"] + HELP):
                _text(img, line, 10, 24 + 22 * k, WHITE)
        return img


def _text(img, text, x, y, colour):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
