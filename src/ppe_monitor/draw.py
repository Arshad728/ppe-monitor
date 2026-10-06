"""Drawing tracked people, their PPE status and the camera's restricted zones on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from .pipeline import FrameResult
from .rules.events import PENDING, VIOLATION
from .rules.ppe import MISSING, UNKNOWN, WORN

# BGR
GREEN, AMBER, RED, GREY, WHITE, BLACK = (60, 190, 60), (0, 170, 255), (40, 40, 230), (160, 160, 160), (255, 255, 255), (0, 0, 0)
HELMET_C, VEST_C = (0, 215, 255), (0, 140, 255)
MARK = {WORN: "ok", MISSING: "NO", UNKNOWN: "?"}


def _box_px(box, w, h):
    return int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)


def _label(img, text, x, y, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), colour, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK if colour != RED else WHITE, 1,
                cv2.LINE_AA)


def draw_zones(img: np.ndarray, r: FrameResult) -> None:
    """Each zone of the camera: tinted red while someone stands in it, outlined in red while one of
    its rules is on, grey and marked "off" outside its rules' active hours."""
    if r.rules is None or not r.rules.zones:
        return
    h, w = img.shape[:2]
    occupied = {z for zs in r.inside.values() for z in zs}
    for zone in r.rules.zones:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], np.int32)
        on = any(rule.zone == zone.id and rule.id in r.active for rule in r.rules.rules)
        colour = RED if on else GREY
        if on and zone.id in occupied:
            tint = img.copy()
            cv2.fillPoly(tint, [pts], RED)
            cv2.addWeighted(tint, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        top = pts[np.argmin(pts[:, 1])]
        _label(img, zone.name + ("" if on else " (off now)"), int(top[0]), int(top[1]) - 2, colour)


def draw(frame: np.ndarray, r: FrameResult, footer: str = "") -> np.ndarray:
    """Person boxes coloured by status: green all fine, amber a violation being confirmed,
    red a confirmed violation, grey can't judge yet. Label: track id, helmet / vest verdicts,
    posture when not upright and the zone they stand in; a dot marks the head found by the keypoint
    model, a ring where they stand. The camera's zones are drawn underneath."""
    img = frame.copy()
    h, w = img.shape[:2]
    draw_zones(img, r)
    for d in r.helmets:
        x1, y1, x2, y2 = _box_px(d.box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), HELMET_C, 1)
    for d in r.vests:
        x1, y1, x2, y2 = _box_px(d.box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), VEST_C, 1)
    for tr in r.tracks:
        v, st = r.verdicts[tr.track_id], r.status[tr.track_id]
        phases = set(st.values())
        if VIOLATION in phases:
            colour = RED
        elif PENDING in phases:
            colour = AMBER
        elif v.helmet == UNKNOWN and v.vest == UNKNOWN:
            colour = GREY
        else:
            colour = GREEN
        x1, y1, x2, y2 = _box_px(tr.box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3 if colour == RED else 2)
        posture = f" ({v.posture})" if v.posture and v.posture != "upright" else ""
        zones = "".join(f" in {z}" for z in r.inside.get(tr.track_id, []))
        _label(img, f"#{tr.track_id} helmet {MARK[v.helmet]} vest {MARK[v.vest]}{posture}{zones}", x1, y1, colour)
        feet = r.feet.get(tr.track_id)
        if feet is not None:                              # where they stand: the point tested against zones
            cv2.circle(img, (int(feet[0] * w), int(feet[1] * h)), 5, colour, 2)
        pose = r.poses.get(tr.track_id)
        if pose is not None and pose.head is not None:  # where the keypoints put the head
            hx, hy = int(pose.head[0] / pose.aspect * w), int(pose.head[1] * h)
            cv2.circle(img, (hx, hy), 4, colour, -1)
    if footer:
        cv2.rectangle(img, (0, h - 28), (w, h), BLACK, -1)
        cv2.putText(img, footer, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return img


def snapshot(frame: np.ndarray, r: FrameResult, event) -> np.ndarray:
    """The evidence image for an event: the annotated frame with the person outlined and captioned."""
    img = draw(frame, r)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _box_px(event.box, w, h)
    cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), RED, 4)
    zone = r.rules.zone(event.zone) if r.rules is not None and event.zone else None
    text = (f"{event.kind.replace('_', ' ').upper()}" + (f" - {zone.name}" if zone else "")
            + f"  [{event.severity or '-'}]  camera {event.camera}  track #{event.track_id}  "
            + (event.time[:19].replace("T", " ") if event.time else f"t={event.confirmed:.1f}s"))
    cv2.rectangle(img, (0, 0), (w, 30), RED, -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return img
