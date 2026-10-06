"""From the camera service to the database, without ever slowing the video loop.

The book's pitfall for this phase: calling the database, Telegram or email from the loop that
processes video. A slow network call there would freeze every camera. So the loop only hands
events to a queue in memory (microseconds); a writer thread draws the evidence image, saves it,
and stores the event and its alerts. If the database is down, the writer keeps retrying; the
events also stay in the run's events.jsonl, as in Phase 4.

LiveTiles writes each camera's newest annotated frame to a file every second or so, for the
dashboard's camera tiles; it also runs in its own thread.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..draw import draw, snapshot
from .settings import ServerSettings
from .store import EventStore


def jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode a JPEG")
    return buf.tobytes()


class DbSink:
    """Hand events, camera states and camera stats to the database from a background thread."""

    def __init__(self, store: EventStore, settings: ServerSettings, run_id: str | None, *,
                 max_queue: int = 1000, log=print):
        self.store, self.s, self.run_id, self.log = store, settings, run_id, log
        self.channels = [name for name, ch in settings.channels.items() if ch.enabled]
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.stored = 0
        self.failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="db-sink", daemon=True)
        self._thread.start()

    # called from the video loop: never blocks
    def _put(self, item) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if self.dropped in (1, 10, 100) or self.dropped % 1000 == 0:
                self.log(f"[db] queue full: {self.dropped} item(s) not stored (they are still in events.jsonl)")

    def event(self, ev, frame: np.ndarray, result) -> None:
        self._put(("raw_event", ev, frame.copy(), result))

    def camera(self, camera: str, state: str, detail: str, when) -> None:
        self._put(("camera", camera, state, detail, when))

    def stats(self, rows: list[dict], detail: dict | None = None) -> None:
        self._put(("stats", rows, detail or {}))

    # the writer thread
    def _handle(self, item) -> None:
        kind = item[0]
        if kind == "event":
            _, ev, event_id, snap, clean, when = item
            channels = [c for c in self.channels if self.s.channels[c].wants(ev.severity)]
            self.store.add_event(ev, run_id=self.run_id, channels=channels, snapshot=snap, frame=clean,
                                 event_id=event_id, confirmed_at=when)
            self.stored += 1
        elif kind == "camera":
            _, camera, state, detail, when = item
            self.store.camera_state(camera, state, detail, when)
        elif kind == "stats":
            _, rows, detail = item
            self.store.camera_stats(rows)
            self.store.heartbeat("camera_service", {**detail, "stored": self.stored, "dropped": self.dropped,
                                                    "queued": self.q.qsize()})

    def _run(self) -> None:
        wait = 0.5
        item = None
        while True:
            if item is None:
                try:
                    item = self.q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        return
                    continue
            if item[0] == "raw_event":        # draw and encode once, so a retry doesn't redo it
                _, ev, frame, result = item
                item = ("event", ev, str(uuid.uuid4()), jpeg(snapshot(frame, result, ev)),
                        jpeg(frame, 92) if self.s.save_frame else None, getattr(result, "now", None))
            try:
                self._handle(item)
                item, wait = None, 0.5
            except Exception as exc:          # the database is down or refused: keep the item, try again
                if item[0] == "stats":        # a newer status report will come: don't retry this one
                    item = None
                self.failures += 1
                if self.failures in (1, 5) or self.failures % 50 == 0:
                    self.log(f"[db] could not store ({type(exc).__name__}: {str(exc).splitlines()[0][:160]}); "
                             f"retrying in {wait:.0f} s")
                if self._stop.is_set() and wait >= 4:
                    return                    # shutting down and still failing: give up
                time.sleep(wait)
                wait = min(wait * 2, 10.0)

    def close(self, timeout: float = 10.0) -> None:
        """Store what is queued (up to `timeout` seconds), then stop."""
        end = time.monotonic() + timeout
        while not self.q.empty() and time.monotonic() < end:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(max(0.1, end - time.monotonic()))


class LiveTiles:
    """Every `every` seconds, the newest annotated frame of each camera -> <dir>/<camera>.jpg."""

    def __init__(self, monitor, folder: Path, every: float = 1.0, width: int = 640):
        self.monitor, self.folder, self.every, self.width = monitor, Path(folder), every, width
        self.folder.mkdir(parents=True, exist_ok=True)
        for old in self.folder.glob("*.jpg"):
            old.unlink(missing_ok=True)       # no stale tiles from an earlier run
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-tiles", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            for cid, (img, r) in list(self.monitor.latest.items()):
                try:
                    h, w = img.shape[:2]
                    small = cv2.resize(img, (self.width, int(h * self.width / w)))
                    tile = draw(small, r)
                    tmp = self.folder / f".{cid}.tmp"
                    tmp.write_bytes(jpeg(tile, 80))
                    tmp.replace(self.folder / f"{cid}.jpg")   # atomic: the dashboard never reads half a file
                except Exception:
                    continue

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
