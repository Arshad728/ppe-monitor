"""The alert worker: takes alerts off the queue (the `alerts` table) and sends them.

It runs as its own process (scripts/alert_worker.py), apart from the camera service, so a slow or
failing Telegram or mail server can never hold up the video (the book's pitfall for this phase).

For each alert, in order:
1. **Rate limit.** If this camera has already had `max_alerts` alerts on this channel in the last
   `minutes`, the alert is held. Held alerts go out together as one summary as soon as the
   camera is under its limit again, so nothing is lost and nobody's phone is flooded.
2. **Send**, with the evidence image.
3. **On failure,** try again after 2, 4, 8 ... s (up to `longest_wait`), `attempts` times in all.
   Errors that retrying can't fix (a wrong token, an unknown chat) fail at once. Telegram's "too
   many requests" is waited out for as long as it asks.

Every alert's outcome is recorded: sent (with the time from the event's confirmation to
delivery), held and summarised, or failed with the reason. The dashboard shows it per event.

It also watches the system itself (Phase 6): a camera that stays down, or a camera service that
stops reporting, is itself worth an alert. A safety system that has quietly stopped watching is
worse than none, because people believe it is watching (`watch_health`).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from .channels import Message, SendError, event_message, make_channels, summary_message
from .db import now_utc, utc
from .settings import ServerSettings
from .store import EventStore, parse_time


class AlertWorker:
    def __init__(self, store: EventStore, settings: ServerSettings, channels: dict | None = None, *, log=print):
        self.store, self.s, self.log = store, settings, log
        self.channels = channels if channels is not None else make_channels(settings)
        self.sent = self.failed = self.held = self.summaries = 0
        self._last_beat = 0.0
        self._last_cleanup = 0.0
        self._last_ping = 0.0
        self.ping_s: dict[str, float | None] = {}        # channel -> last round trip (s), None if it failed
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="send")
        self._count = threading.Lock()
        self._stop = threading.Event()
        self._down: dict[str, str] = {}          # camera -> since when it is down (alert sent for it)
        self._silent_since: str | None = None    # the camera service's last heartbeat, when we said it went silent
        self._health_out: list[list] = []        # [message, tries] not sent yet
        self.health_sent = 0

    # -- one pass ---------------------------------------------------------------------------------
    def window_start(self):
        return now_utc() - timedelta(minutes=self.s.rate_minutes)

    def over_limit(self, channel: str, camera: str) -> bool:
        return self.store.sent_recently(channel, camera, self.window_start()) >= self.s.rate_max

    def _deliver(self, channel, msg: Message, done: list[str]) -> list[str]:
        cfg = self.s.channels.get(channel.name)
        if cfg is not None and not cfg.photo:
            msg.photo = None                  # text only: the picture doesn't leave this machine
        return channel.send(msg, done)

    def send_summaries(self) -> int:
        n = 0
        for g in self.store.held_groups():
            ch = self.channels.get(g["channel"])
            if ch is None or ch.missing() or self.over_limit(g["channel"], g["camera"]):
                continue
            try:
                done = self._deliver(ch, summary_message(g, self.s.rate_minutes, self.s.dashboard_url), [])
            except SendError as exc:
                self.log(f"[alerts] summary for {g['camera']} on {g['channel']} not sent yet: {exc}")
                continue
            self.store.record_summary(g, done)
            self.summaries += 1
            n += 1
            self.log(f"[alerts] {g['channel']}: summary of {len(g['ids'])} held alert(s) for {g['camera']}")
        return n

    def _send_one(self, ch, a: dict, claimed: float) -> None:
        t0 = time.perf_counter()
        try:
            done = self._deliver(ch, event_message(a, self.s.dashboard_url), a["done"])
        except SendError as exc:
            attempts = a["attempts"] + 1
            give_up = exc.permanent or attempts >= self.s.retry_attempts
            wait = exc.retry_after if exc.retry_after else min(self.s.retry_first * 2 ** (attempts - 1),
                                                                self.s.retry_longest)
            self.store.mark_retry(a["id"], str(exc), wait, getattr(exc, "done", a["done"]), give_up)
            if give_up:
                with self._count:
                    self.failed += 1
                self.log(f"[alerts] {a['channel']}: gave up on the alert for event {a['event_id'][:8]} "
                         f"after {attempts} attempt(s): {exc}")
            else:
                self.log(f"[alerts] {a['channel']}: {exc}; trying again in {wait:.0f} s")
            return
        sending = time.perf_counter() - t0
        latency = self.store.mark_sent(a["id"], a["confirmed_at"], done)
        with self._count:
            self.sent += 1
        waited = (latency / 1000 - sending) if latency is not None else None
        self.log(f"[alerts] {a['channel']}: {a['kind']} on {a['camera']} sent"
                 + (f", {latency / 1000:.1f} s after the event ({waited:.1f} s waiting, {sending:.1f} s sending)"
                    if latency is not None else ""))

    def process(self, limit: int = 10) -> int:
        """Send what is due, up to 4 at a time. Returns how many alerts were taken off the queue."""
        batch = self.store.claim_alerts(limit)
        claimed = time.perf_counter()
        sent_now: dict[tuple, int] = {}                     # (channel, camera) -> count, rate limit
        jobs = []
        for a in batch:
            ch = self.channels.get(a["channel"])
            if ch is None:
                self.store.mark_failed(a["id"], f"unknown channel {a['channel']}")
                self.failed += 1
                continue
            why = ch.missing()
            if why:
                self.store.mark_failed(a["id"], why)
                self.failed += 1
                self.log(f"[alerts] {a['channel']}: not set up: {why}")
                continue
            key = (a["channel"], a["camera"])
            if key not in sent_now:
                sent_now[key] = self.store.sent_recently(a["channel"], a["camera"], self.window_start())
            if sent_now[key] >= self.s.rate_max:
                self.store.mark_held(a["id"])
                self.held += 1
                continue
            sent_now[key] += 1
            jobs.append((ch, a))
        # alerts of one batch go out in parallel: a slow upload doesn't hold up the next alert
        for f in [self._pool.submit(self._send_one, ch, a, claimed) for ch, a in jobs]:
            f.result()
        return len(batch)

    def keep_warm(self, every: float = 60.0) -> None:
        """Ask each configured Telegram channel who it is about once a minute: the connection stays
        open (the next alert needs no new handshake) and a broken connection shows up on the
        dashboard before an alert needs it."""
        mono = time.monotonic()
        if mono - self._last_ping < every:
            return
        self._last_ping = mono
        for name, ch in self.channels.items():
            if hasattr(ch, "ping") and not ch.missing() and self.s.channels.get(name) and self.s.channels[name].enabled:
                try:
                    self.ping_s[name] = ch.ping()
                except SendError as exc:
                    self.ping_s[name] = None
                    self.log(f"[alerts] {name}: can't reach it ({exc}); alerts will be retried")

    def housekeeping(self, force: bool = False) -> None:
        mono = time.monotonic()
        if force or mono - self._last_beat >= 5:
            self._last_beat = mono
            self.watch_health()
            self.store.heartbeat("alert_worker", {"sent": self.sent, "failed": self.failed, "held": self.held,
                                                  "summaries": self.summaries, "health_alerts": self.health_sent,
                                                  "channels": {k: (not v.missing()) for k, v in self.channels.items()},
                                                  "round_trip_s": self.ping_s})
        if force or mono - self._last_cleanup >= 3600:
            self._last_cleanup = mono
            n = self.store.cleanup_images(self.s.keep_days)
            if n:
                self.log(f"[alerts] deleted the images of {n} event(s) older than {self.s.keep_days:g} days")

    # -- watching the system itself (Phase 6) -----------------------------------------------------
    @staticmethod
    def _clock(iso_time: str | None) -> str:
        t = parse_time(iso_time)
        return t.astimezone().strftime("%H:%M:%S") if t else "?"

    def watch_health(self, now=None) -> list[str]:
        """One alert when a camera has been down (connecting or reconnecting) for longer than
        `camera_down_after` seconds, and one when it is back; the same for a camera service whose
        heartbeat stops for longer than `service_silent_after` seconds. Cameras that went down
        together share one message. A deliberate stop (the camera service says so as it ends) is
        not an outage. Returns the titles of the messages queued this time."""
        now = utc(now) or now_utc()
        queued = []
        svc = self.store.services().get("camera_service")
        if svc and svc.get("last_seen") and self.s.service_silent_after > 0:
            age = (now - parse_time(svc["last_seen"])).total_seconds()
            stopped = bool((svc.get("detail") or {}).get("stopped"))
            if not stopped and age > self.s.service_silent_after:
                if self._silent_since != svc["last_seen"]:
                    self._silent_since = svc["last_seen"]
                    queued.append(Message("\u26A0\uFE0F The camera service has stopped reporting",
                                          [f"No word from it since {self._clock(svc['last_seen'])} ({age / 60:.0f} min): "
                                           "no camera is being watched.", "Restart it, or check the machine it runs on."]))
                return self._queue(queued)             # camera states are stale while it is silent
            if self._silent_since is not None:
                self._silent_since = None
                if not stopped:
                    queued.append(Message("\u2705 The camera service is reporting again", ["The cameras are watched again."]))
        elif not svc:
            return self._queue(queued)
        if self.s.camera_down_after <= 0:
            return self._queue(queued)
        down_now, new_down, back = {}, [], []
        for cam in self.store.cameras():
            if cam["state"] in ("connecting", "reconnecting") and cam.get("state_since"):
                since = parse_time(cam["state_since"])
                if (now - since).total_seconds() > self.s.camera_down_after:
                    down_now[cam["id"]] = cam
                    if cam["id"] not in self._down:
                        self._down[cam["id"]] = cam["state_since"]
                        new_down.append(cam)
            elif cam["id"] in self._down:
                since = self._down.pop(cam["id"])
                if cam["state"] == "live":
                    back.append((cam, since))
        if new_down:
            names = ", ".join(c["name"] or c["id"] for c in new_down)
            queued.append(Message(f"\U0001F4F7 Camera{'s' if len(new_down) > 1 else ''} down: {names}",
                                  [f"{c['name'] or c['id']}: not sending since {self._clock(c['state_since'])}"
                                   + (f" ({c['detail'][:80]})" if c.get("detail") else "") for c in new_down]
                                  + ["Nothing is detected on it until it is back."]))
        if back:
            queued.append(Message(f"\u2705 Back: {', '.join(c['name'] or c['id'] for c, _ in back)}",
                                  [f"{c['name'] or c['id']}: down from {self._clock(since)} to {self._clock(c['state_since'])}"
                                   for c, since in back]))
        return self._queue(queued)

    def _queue(self, msgs: list[Message]) -> list[str]:
        for m in msgs:
            self.log(f"[health] {m.title}")
            self._health_out.append([m, 0])
        self._send_health()
        return [m.title for m in msgs]

    def _send_health(self) -> None:
        """Send what is waiting to every channel that takes high-severity alerts; keep what failed
        for the next pass (at most 10 tries)."""
        keep = []
        for item in self._health_out:
            msg, tries = item
            ok = True
            for name, ch in self.channels.items():
                cfg = self.s.channels.get(name)
                if cfg is None or not cfg.wants("high") or ch.missing():
                    continue
                try:
                    ch.send(Message(msg.title, list(msg.lines), None, self.s.dashboard_url, msg.camera), [])
                except SendError as exc:
                    ok = False
                    self.log(f"[health] {name}: not sent yet ({exc})")
            if ok:
                self.health_sent += 1
            elif tries + 1 < 10:
                keep.append([msg, tries + 1])
        self._health_out = keep

    # -- the loop ---------------------------------------------------------------------------------
    def run(self, seconds: float | None = None) -> None:
        end = None if seconds is None else time.monotonic() + seconds
        stuck = self.store.recover_stuck()
        if stuck:
            self.log(f"[alerts] {stuck} alert(s) left half-sent by an earlier run are back in the queue")
        self.housekeeping(force=True)
        self.keep_warm(0)
        for name, rt in self.ping_s.items():
            if rt is not None:
                self.log(f"[alerts] {name}: reachable, {rt:.2f} s round trip")
        wait = self.s.poll
        while not self._stop.is_set() and (end is None or time.monotonic() < end):
            try:
                self.send_summaries()
                busy = self.process()
                self.housekeeping()
                if not busy:
                    self.keep_warm()
                wait = self.s.poll
            except Exception as exc:        # the database is down: wait and try again
                busy = 0
                self.log(f"[alerts] {type(exc).__name__}: {str(exc).splitlines()[0][:160]}; retrying in {wait:.0f} s")
                wait = min(max(wait, 1.0) * 2, 30.0)
            if not busy:
                self._stop.wait(wait)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
