"""The annotation tool's logic, apart from its window (scripts/annotate_clip.py), so it can be tested.

A person marks each violation in a clip: where it begins, where it ends, and where the person is
(a box drawn at one or more moments). The episode is "sustained" (an alert is expected) when it
lasts at least the rule's dwell plus a margin; the key `n` overrides that.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .truth import KINDS, ClipTruth, Episode

KEY_KIND = {ord("h"): "no_helmet", ord("v"): "no_vest", ord("z"): "zone_intrusion"}
COLOURS = {"no_helmet": (0, 0, 255), "no_vest": (0, 165, 255), "zone_intrusion": (255, 0, 255)}
HELP = ["space play/pause   , . one frame   [ ] one second   Home: start",
        "h / v / z  begin a no-helmet / no-vest / zone episode here   e  end it here",
        "drag the mouse: the person's box now   n  sustained yes/no   x  delete the selected episode",
        "Tab  select the next episode   s  save   q  quit"]


class Annotator:
    def __init__(self, truth: ClipTruth, n_frames: int, dwell: dict[str, float] | None = None, margin: float = 0.5):
        self.truth, self.n = truth, n_frames
        self.dwell = {"no_helmet": 3.0, "no_vest": 3.0, "zone_intrusion": 2.0, **(dwell or {})}
        self.margin = margin
        self.frame = 0
        self.selected: int | None = len(truth.episodes) - 1 if truth.episodes else None
        self.open: int | None = None           # the episode begun and not yet ended
        self.manual: set[int] = set()          # episodes whose "sustained" was set by hand
        self.dirty = False
        self.top_h = 0

    # -- time ---------------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self.frame / self.truth.fps

    def seek(self, frame: int) -> None:
        self.frame = max(0, min(self.n - 1, frame))

    def step(self, frames: int) -> None:
        self.seek(self.frame + frames)

    # -- episodes -----------------------------------------------------------------------------------
    def _auto_sustained(self, i: int) -> None:
        if i in self.manual:
            return
        e = self.truth.episodes[i]
        e.sustained = (e.end - e.start) >= self.dwell[e.kind] + self.margin

    def begin(self, kind: str) -> int:
        if kind not in KINDS:
            raise ValueError(kind)
        if kind not in self.truth.checked:
            self.truth.checked = tuple(self.truth.checked) + (kind,)
        end = min(self.truth.duration, self.t + 1.0 / self.truth.fps)
        self.truth.episodes.append(Episode(kind, self.t, end, False))
        self.selected = self.open = len(self.truth.episodes) - 1
        self.dirty = True
        return self.selected

    def end(self) -> bool:
        i = self.open if self.open is not None else self.selected
        if i is None:
            return False
        e = self.truth.episodes[i]
        if self.t <= e.start:
            return False
        e.end = self.t
        self._auto_sustained(i)
        self.open = None
        self.dirty = True
        return True

    def toggle_sustained(self) -> None:
        if self.selected is None:
            return
        e = self.truth.episodes[self.selected]
        e.sustained = not e.sustained
        self.manual.add(self.selected)
        self.dirty = True

    def delete(self) -> None:
        if self.selected is None:
            return
        del self.truth.episodes[self.selected]
        self.manual = {i if i < self.selected else i - 1 for i in self.manual if i != self.selected}
        self.open = None
        self.selected = (len(self.truth.episodes) - 1) if self.truth.episodes else None
        self.dirty = True

    def next(self) -> None:
        if self.truth.episodes:
            self.selected = 0 if self.selected is None else (self.selected + 1) % len(self.truth.episodes)

    def set_box(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """The selected episode's person, at the current time (fractions of the frame)."""
        if self.selected is None:
            return False
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            return False
        e = self.truth.episodes[self.selected]
        e.boxes = sorted([b for b in e.boxes if abs(b[0] - self.t) > 1e-6] + [(self.t, (x1, y1, x2, y2))])
        self.dirty = True
        return True

    def active(self) -> list[int]:
        return [i for i, e in enumerate(self.truth.episodes) if e.start <= self.t <= e.end or i == self.open]

    # -- drawing ------------------------------------------------------------------------------------
    def render(self, frame: np.ndarray, show_help: bool = True) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        for i in self.active():
            e = self.truth.episodes[i]
            b = e.box_at(self.t)
            col = COLOURS[e.kind]
            if b is not None:
                cv2.rectangle(img, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), col,
                              3 if i == self.selected else 1)
                cv2.putText(img, f"{i + 1} {e.kind}{'' if e.sustained else ' (not sustained)'}",
                            (int(b[0] * w) + 4, max(18, int(b[1] * h) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        bar = np.zeros((34, w, 3), np.uint8)
        for i, e in enumerate(self.truth.episodes):
            x0, x1 = int(e.start / self.truth.duration * w), int(min(1.0, e.end / self.truth.duration) * w)
            cv2.rectangle(bar, (x0, 4 + (i % 3) * 9), (max(x0 + 2, x1), 11 + (i % 3) * 9), COLOURS[e.kind],
                          -1 if e.sustained else 1)
        x = int(self.frame / max(1, self.n - 1) * (w - 1))
        cv2.line(bar, (x, 0), (x, 33), (255, 255, 255), 2)
        status = (f"{self.t:6.2f} s / {self.truth.duration:.2f} s   frame {self.frame}   "
                  f"{len(self.truth.episodes)} episode(s)" + ("   (not saved)" if self.dirty else ""))
        top = np.zeros((26 + (18 * len(HELP) if show_help else 0), w, 3), np.uint8)
        self.top_h = top.shape[0]              # the window's mouse positions start this far down
        cv2.putText(top, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if show_help:
            for k, line in enumerate(HELP):
                cv2.putText(top, line, (8, 38 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        return np.vstack([top, img, bar])


def new_truth(clip_path: Path, fps: float, n_frames: int) -> ClipTruth:
    return ClipTruth(clip_path.name, clip_path, fps, n_frames / fps, [], ("no_helmet",),
                     "hand-annotated with scripts/annotate_clip.py")
