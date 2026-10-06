"""Reading a live stream in the background and keeping only the newest frame.

If processing is slower than the camera, the processing loop skips stale frames instead of falling
further and further behind real time (the "latest-frame" idea from Chapter 16; Phase 4 builds the
multi-camera version).
"""

from __future__ import annotations

import threading
import time


class LatestFrame:
    def __init__(self, cap):
        self.cap, self.frame, self.seq, self.ok = cap, None, 0, True
        self.stamp = 0.0  # time.monotonic() when the newest frame arrived
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame, self.seq, self.stamp = frame, self.seq + 1, time.monotonic()

    def get(self, last_seq: int):
        """Wait for a frame newer than `last_seq`. Returns (ok, frame, seq, arrival time)."""
        while True:
            with self.lock:
                if not self.ok or self.seq != last_seq:
                    return self.ok, self.frame, self.seq, self.stamp
            time.sleep(0.002)

    def stop(self):
        self.ok = False
