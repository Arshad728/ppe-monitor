"""Short test videos made from labelled test photos, for checking the whole chain end to end.

Until real site clips are available, this is the closest thing to video with known answers. Each clip
is one test photo, whose people have a known helmet/vest status (data/compliance.py), turned into
10 seconds of video:
  - a slow pan and zoom, so every frame is slightly different and detections flicker as they
    would on a real camera;
  - camera noise and brightness drift, then normal video compression;
  - a dark vertical bar (like a pole, or someone walking past close to the camera) that sweeps
    across the frame from 3 s to 7 s and briefly hides each person. This tests that tracks
    survive occlusion.

The people don't move and nobody puts a helmet on or takes it off, so this tests:
  - that the chain raises one event per person who is really without PPE;
  - that it raises none for people wearing it, despite frame-to-frame flicker;
  - that track IDs survive the occlusion.
With motion="sway" the frame is 1.25x as wide as the photo and the photo slides from side to side
in it (grey on either side), so everyone moves at walking pace (up to about 0.2 frame widths a
second), back and forth, and stays in view. That tests tracking and events for moving people, especially when frames
passes behind anyone or changes their PPE. That needs real clips.

With motion="pan" (Phase 3) the frame is 1.4x as wide as the photo and the photo slides steadily
from the left to the right over the clip, while a restricted zone covers the right part of the
frame. People on the right of the photo walk into the zone at different times and stay; people on
the left never reach it. Every labelled person is included, with or without a known helmet / vest
status, since whether someone stands in the zone doesn't depend on their PPE.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from ..data.compliance import PersonTruth
from ..rules.check import Case

MAX_SIDE = 1280


SWAY_MARGIN = 1.25  # moving clips are 1.25x as wide as the photo, so it can slide from side to side
PAN_MARGIN = 1.4    # zone clips are 1.4x as wide, and the photo slides from the left edge to the right edge
ZONE = ((0.60, 0.0), (1.0, 0.0), (1.0, 1.0), (0.68, 1.0))   # the zone clips' restricted zone, frame fractions


def _pan_crop(k: int, n: int) -> tuple[float, float, float, float]:
    """The photo moving steadily from the left edge of a 1.4x wide frame to its right edge."""
    x0 = -(PAN_MARGIN - 1) * k / max(1, n - 1)
    return x0, 0.0, PAN_MARGIN, 1.0


def _sway_crop(k: int, fps: float, phase: float) -> tuple[float, float, float, float]:
    """The whole photo sliding from side to side in a frame 1.25x as wide (grey on either side):
    10% of the frame width either way, every 3 s. Everyone moves across the frame at walking pace
    and stays in view. As a crop window: 1.25x the photo's width, partly outside it."""
    x0 = -(SWAY_MARGIN - 1) / 2 + (SWAY_MARGIN - 1) / 2 * np.sin(2 * np.pi * (k / fps) / 3.0 + phase)
    return x0, 0.0, SWAY_MARGIN, 1.0


def _crop_at(k: int, n: int, focus: tuple[float, float], rng_params) -> tuple[float, float, float, float]:
    """Crop window (x0, y0, w, h) in 0-1 image fractions for frame k of n."""
    zoom_end, dx, dy = rng_params
    a = k / max(1, n - 1)
    s = 1.0 + (zoom_end - 1.0) * a                     # 1.0 -> zoom_end (e.g. 0.85 = 15% closer)
    cx = 0.5 + (focus[0] - 0.5) * (1 - s) * 2 * a + dx * a
    cy = 0.5 + (focus[1] - 0.5) * (1 - s) * 2 * a + dy * a
    x0 = min(max(cx - s / 2, 0.0), 1 - s)
    y0 = min(max(cy - s / 2, 0.0), 1 - s)
    return x0, y0, s, s


def _map_box(box, crop) -> tuple[float, float, float, float] | None:
    x0, y0, w, h = crop
    b = ((box[0] - x0) / w, (box[1] - y0) / h, (box[2] - x0) / w, (box[3] - y0) / h)
    c = (max(0.0, b[0]), max(0.0, b[1]), min(1.0, b[2]), min(1.0, b[3]))
    if c[2] <= c[0] or c[3] <= c[1]:
        return None
    return c


def make_clip(case: Case, out: Path, *, seconds: float = 10.0, fps: float = 15.0, seed: int = 0,
              motion: str = "drift") -> dict:
    """Write the clip and return its ground truth: per person, the truth and a box per frame.
    motion: "drift" (slow pan and zoom), "sway" (the whole photo slides from side to side, so
    everyone moves across the frame at about walking speed) or "pan" (the photo slides steadily to the
    right, into a restricted zone; the truth then includes the zone)."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    img = cv2.imread(str(case.image))
    h, w = img.shape[:2]
    wide = {"sway": SWAY_MARGIN, "pan": PAN_MARGIN}.get(motion, 1.0)
    scale = min(1.0, MAX_SIDE / max(h, w * wide))
    out_w, out_h = int(w * wide * scale) // 2 * 2, int(h * scale) // 2 * 2
    n = int(seconds * fps)
    known = [t for t in case.truths]
    if motion == "pan":   # everyone labelled, known PPE status or not
        others = [b for b in case.gt_boxes.get("person", []) if all(tuple(b) != tuple(t.box) for t in known)]
        known += [PersonTruth(tuple(b), None, None) for b in others]
    fx = float(np.mean([(t.box[0] + t.box[2]) / 2 for t in known]))
    fy = float(np.mean([(t.box[1] + t.box[3]) / 2 for t in known]))
    params = (rng.uniform(0.82, 0.92), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03))
    bar_w = 0.07
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    boxes = [[] for _ in known]
    for k in range(n):
        crop = (_crop_at(k, n, (fx, fy), params) if motion == "drift" else
                _sway_crop(k, fps, params[1] * 50) if motion == "sway" else _pan_crop(k, n))
        x0, y0, cw, ch = crop
        m = np.float32([[out_w / (cw * w), 0, -x0 * out_w / cw], [0, out_h / (ch * h), -y0 * out_h / ch]])
        frame = cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(90, 90, 90))
        gain = 1.0 + 0.06 * np.sin(2 * np.pi * k / (fps * 4))
        frame = np.clip(frame.astype(np.float32) * gain + nrng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
        t = k / fps
        bar = None
        if 3.0 <= t < 7.0:                              # the occluder sweeps left to right
            bx = (t - 3.0) / 4.0 * (1 + bar_w) - bar_w
            x1, x2 = int(max(0, bx) * out_w), int(min(1, bx + bar_w) * out_w)
            frame[:, x1:x2] = (frame[:, x1:x2] * 0.15 + 30).astype(np.uint8)
            bar = (bx, bx + bar_w)
        writer.write(frame)
        for p, truth in enumerate(known):
            b = _map_box(truth.box, crop)
            hidden = bar is not None and b is not None and bar[0] <= (b[0] + b[2]) / 2 <= bar[1]
            boxes[p].append(None if b is None else (*(float(v) for v in b), bool(hidden)))
    writer.release()
    truth = {"clip": out.name, "image": case.image.name, "fps": fps, "frames": n, "size": [out_w, out_h],
             "people": [{"helmet": t.helmet, "vest": t.vest, "boxes": boxes[p]} for p, t in enumerate(known)]}
    if motion == "pan":
        truth["zone"] = {"id": "zone", "name": "Test zone", "polygon": [list(c) for c in ZONE]}
    return truth
