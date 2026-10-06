"""Synthetic test clips for exercising the video plumbing before real footage exists.

These clips are cartoons: simple figures with or without a helmet and vest walking on a
stylised site, a hatched restricted zone, and a burned-in camera label, clock and frame
counter. They are deliberately NOT training or evaluation data for the detector. Their
only job is to give the fake camera network something realistic-sized to stream, with a
few scripted "violations" so later phases have predictable moments to look at:

    worker A  takes the helmet off between 6 s and 12 s and sets it on the railing
    worker B  never wears a vest
    worker C  is inside the restricted zone from about 8 s to 17 s
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from ..executables import encoder_args, find_ffmpeg, pick_encoder
from ..ingestion.rtsp import open_capture

# BGR colours
_HELMET_YELLOW = (0, 215, 255)
_HELMET_WHITE = (235, 235, 235)
_VEST_ORANGE = (0, 120, 255)
_VEST_LIME = (60, 255, 200)
_STRIPE = (210, 210, 210)
_SKIN = (120, 160, 205)
_SHIRTS = [(150, 90, 40), (60, 60, 60), (40, 110, 60), (120, 40, 110)]


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    frames: int
    fps: float
    width: int
    height: int
    h264: bool


@dataclass
class _Worker:
    label: str
    base_x: float  # fraction of width
    y: float  # feet position, fraction of height
    amp: float  # walking amplitude, fraction of width
    speed: float  # radians per second
    phase: float
    shirt: tuple
    vest_colour: tuple | None
    helmet_colour: tuple | None
    helmet_off: tuple[float, float] | None = None  # seconds


def _background(w: int, h: int, rng: np.random.Generator, zone: np.ndarray) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.38)
    # sky / far wall, then ground with a gentle vertical gradient
    img[:horizon] = (170, 150, 120)
    for y in range(horizon, h):
        shade = int(95 + 50 * (y - horizon) / (h - horizon))
        img[y] = (shade - 10, shade, shade + 5)
    # scaffolding in the background
    for i in range(6):
        x = int(w * (0.08 + i * 0.07))
        cv2.line(img, (x, int(h * 0.12)), (x, horizon + 10), (90, 90, 100), 4)
    for j in range(4):
        y = int(h * (0.14 + j * 0.07))
        cv2.line(img, (int(w * 0.08), y), (int(w * 0.43), y), (90, 90, 100), 3)
    # a railing on the right (worker A puts the helmet here)
    ry = int(h * 0.55)
    cv2.line(img, (int(w * 0.62), ry), (int(w * 0.97), ry), (70, 70, 80), 5)
    for x in np.linspace(w * 0.62, w * 0.97, 8):
        cv2.line(img, (int(x), ry), (int(x), ry + int(h * 0.12)), (70, 70, 80), 4)
    # restricted zone: hatched polygon with a red outline
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [zone], 255)
    stripes = np.zeros_like(img)
    for k in range(-h, w, 40):
        cv2.line(stripes, (k, h), (k + h, 0), (0, 200, 230), 14)
    img[mask > 0] = (0.55 * img[mask > 0] + 0.45 * stripes[mask > 0]).astype(np.uint8)
    cv2.polylines(img, [zone], True, (40, 40, 220), 3)
    cv2.putText(img, "RESTRICTED", tuple(zone[0] + np.array([10, 30])), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 220), 2, cv2.LINE_AA)
    # fine sensor-like noise so frames are never perfectly flat
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_worker(img: np.ndarray, wk: _Worker, t: float, w: int, h: int) -> None:
    x = w * (wk.base_x + wk.amp * math.sin(wk.speed * t + wk.phase))
    y = h * wk.y
    scale = 0.6 + 0.9 * (wk.y - 0.4)  # nearer the camera (lower in frame) = bigger
    s = h * 0.30 * scale  # full body height in pixels
    step = math.sin(wk.speed * t * 6 + wk.phase)
    cx, feet = int(x), int(y)
    hip = int(feet - 0.45 * s)
    shoulder = int(feet - 0.80 * s)
    head_r = max(4, int(0.075 * s))
    head_c = (cx, int(shoulder - head_r * 1.2))
    lw = max(2, int(s * 0.05))

    # legs and arms
    cv2.line(img, (cx, hip), (int(cx - 0.12 * s * step), feet), (50, 45, 40), lw)
    cv2.line(img, (cx, hip), (int(cx + 0.12 * s * step), feet), (50, 45, 40), lw)
    cv2.line(img, (cx - int(0.12 * s), shoulder + lw), (int(cx - 0.18 * s), int(hip + 0.05 * s)), wk.shirt, lw)
    cv2.line(img, (cx + int(0.12 * s), shoulder + lw), (int(cx + 0.18 * s), int(hip + 0.05 * s)), wk.shirt, lw)
    # torso, optionally with a hi-vis vest and reflective stripes
    tl, br = (cx - int(0.13 * s), shoulder), (cx + int(0.13 * s), hip)
    cv2.rectangle(img, tl, br, wk.shirt, -1)
    if wk.vest_colour:
        cv2.rectangle(img, (tl[0] + 2, tl[1] + 2), (br[0] - 2, br[1] - 2), wk.vest_colour, -1)
        for f in (0.45, 0.72):
            yy = int(shoulder + f * (hip - shoulder))
            cv2.line(img, (tl[0] + 2, yy), (br[0] - 2, yy), _STRIPE, max(2, lw // 2))
    # head, optionally with a helmet
    cv2.circle(img, head_c, head_r, _SKIN, -1)
    helmet_on = wk.helmet_colour is not None and not (
        wk.helmet_off and wk.helmet_off[0] <= t < wk.helmet_off[1]
    )
    if helmet_on:
        cv2.ellipse(img, (head_c[0], head_c[1] - head_r // 4), (int(head_r * 1.25), int(head_r * 1.1)),
                    0, 180, 360, wk.helmet_colour, -1)
        cv2.line(img, (head_c[0] - int(head_r * 1.4), head_c[1] - head_r // 4),
                 (head_c[0] + int(head_r * 1.4), head_c[1] - head_r // 4), wk.helmet_colour, max(2, lw // 2))
    elif wk.helmet_colour is not None:
        # helmet resting on the railing while it is off
        rx, ry = int(w * 0.80), int(h * 0.55)
        cv2.ellipse(img, (rx, ry - 2), (int(head_r * 1.25), int(head_r * 1.1)), 0, 180, 360, wk.helmet_colour, -1)


def _scenario(seed: int) -> list[_Worker]:
    rng = np.random.default_rng(seed)
    shirts = [_SHIRTS[i % len(_SHIRTS)] for i in rng.permutation(len(_SHIRTS))]
    return [
        _Worker("A", 0.70, 0.78, 0.10, 0.5, rng.uniform(0, 6), shirts[0], _VEST_ORANGE, _HELMET_YELLOW,
                helmet_off=(6.0, 12.0)),
        _Worker("B", 0.30, 0.66, 0.12, 0.35, rng.uniform(0, 6), shirts[1], None, _HELMET_WHITE),
        # starts at the right, is inside the zone from about 8 s to 17 s, then walks back out
        _Worker("C", 0.55, 0.90, 0.30, 0.25, math.pi / 2, shirts[2], _VEST_LIME, _HELMET_YELLOW),
    ]


def make_synthetic_clip(
    out_path: str | Path,
    label: str = "CAM 1",
    *,
    seconds: float = 20.0,
    fps: float = 15.0,
    size: tuple[int, int] = (1280, 720),
    seed: int = 0,
    start_time: datetime | None = None,
    to_h264: bool = True,
) -> ClipInfo:
    """Render a synthetic site clip. With `to_h264` (and ffmpeg available) the result is
    re-encoded to H.264 with a keyframe every second, like a typical IP camera."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    rng = np.random.default_rng(seed)
    zone = np.array([[int(w * 0.04), int(h * 0.74)], [int(w * 0.40), int(h * 0.74)],
                     [int(w * 0.46), int(h * 0.98)], [int(w * 0.02), int(h * 0.98)]], np.int32)
    bg = _background(w, h, rng, zone)
    workers = _scenario(seed)
    start_time = start_time or datetime(2026, 1, 5, 9, 30, 0)
    n_frames = int(round(seconds * fps))

    ffmpeg = find_ffmpeg() if to_h264 else None
    if ffmpeg:
        fd, tmp_name = tempfile.mkstemp(suffix=".avi")
        os.close(fd)
        raw_path = Path(tmp_name)
    else:
        raw_path = out_path
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*("MJPG" if ffmpeg else "mp4v")), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {raw_path}")
    try:
        for i in range(n_frames):
            t = i / fps
            frame = bg.copy()
            for wk in sorted(workers, key=lambda k: k.y):  # far first, near last
                _draw_worker(frame, wk, t, w, h)
            stamp = (start_time + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.") + f"{int((t % 1) * 10)}"
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}   {stamp}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"#{i:05d}", (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "SYNTHETIC - plumbing test only", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()

    h264 = False
    if ffmpeg:
        encoder = pick_encoder(ffmpeg)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path), "-an",
               "-c:v", encoder, *encoder_args(encoder, live=False), "-pix_fmt", "yuv420p",
               "-r", f"{fps:g}", "-bf", "0", "-force_key_frames", "expr:gte(t,n_forced*1)",
               "-movflags", "+faststart", str(out_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            h264 = encoder != "mpeg4"
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed to encode {out_path}: {exc.stderr.strip()}") from exc
        finally:
            Path(raw_path).unlink(missing_ok=True)

    return ClipInfo(path=out_path, frames=n_frames, fps=fps, width=w, height=h, h264=h264)


def clip_fps(path: str | Path) -> float | None:
    """Frame rate of a local video file, or None if OpenCV can't read it."""
    cap = open_capture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    finally:
        cap.release()
    return fps if 1.0 <= fps <= 240.0 else None


def clip_duration(path: str | Path) -> float | None:
    """Length of a local video file in seconds, or None if OpenCV can't read it."""
    cap = open_capture(str(path))
    try:
        if not cap.isOpened():
            return None
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return frames / fps if frames > 0 and 1.0 <= fps <= 240.0 else None


__all__ = ["ClipInfo", "make_synthetic_clip", "clip_fps", "clip_duration"]
