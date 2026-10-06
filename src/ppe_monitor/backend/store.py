"""Reading and writing events, cameras and the alert queue (one class, used by every Phase 5 process).

The camera service adds events (through backend/sink.py, off its video loop); the alert worker
takes alerts from the queue; the dashboard lists events, shows their images and records the
safety officer's verdict. Evidence images are files under storage.dir; the database holds their
paths.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import and_, func, select, update

from .db import (alerts, camera_log, cameras, create_schema, events, make_engine, now_utc, rules, runs, services,
                 utc, zones)

STATUSES = ("new", "confirmed", "false_alarm")
KINDS = ("no_helmet", "no_vest", "zone_intrusion")


def iso(dt: datetime | None) -> str | None:
    dt = utc(dt)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z") if dt else None


def parse_time(value: str | datetime | None) -> datetime | None:
    """ISO 8601 (with or without a time zone; none means UTC) -> aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return utc(value)
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    return utc(dt)


class EventStore:
    def __init__(self, url: str, storage_dir: str | Path, *, create: bool = True, echo: bool = False):
        self.url = url
        self.engine = make_engine(url, echo)
        self.storage_dir = Path(storage_dir)
        self.is_postgres = self.engine.dialect.name == "postgresql"
        if create:
            create_schema(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    # -- set-up ---------------------------------------------------------------------------------
    def start_run(self, *, host: str, backend: str, model: str, camera_ids: list[str]) -> str:
        run_id = str(uuid.uuid4())
        with self.engine.begin() as c:
            c.execute(runs.insert().values(id=run_id, started_at=now_utc(), host=host, backend=backend, model=model,
                                           cameras=camera_ids))
        return run_id

    def sync_config(self, camera_names: dict[str, str], ruleset=None) -> None:
        """Cameras (id -> name), and the rules and zones in force, from the config files."""
        now = now_utc()
        with self.engine.begin() as c:
            have = {r.id for r in c.execute(select(cameras.c.id))}
            for cid, name in camera_names.items():
                if cid in have:
                    c.execute(update(cameras).where(cameras.c.id == cid).values(name=name))
                else:
                    c.execute(cameras.insert().values(id=cid, name=name, state="connecting", state_since=now, events=0))
            if ruleset is None:
                return
            c.execute(rules.delete())
            for r in ruleset.rules:
                c.execute(rules.insert().values(
                    id=r.id, type=r.type, severity=r.severity, cameras=list(r.cameras) if r.cameras else None,
                    zone=r.zone, dwell=r.dwell, cooldown=r.cooldown,
                    active=getattr(r.schedule, "text", None) if r.schedule is not None else None,
                    description=getattr(r, "description", "") or "", updated_at=now))
            c.execute(zones.delete())
            for cid, cz in (getattr(ruleset, "zones", None) or {}).items():
                for z in cz.zones:
                    c.execute(zones.insert().values(camera=cid, zone=z.id, name=z.name,
                                                    polygon=[list(map(float, p)) for p in z.polygon], updated_at=now))

    # -- the camera service ---------------------------------------------------------------------
    def save_images(self, event_id: str, when: datetime, snapshot: bytes | None, frame: bytes | None) -> tuple[str | None, str | None]:
        day = utc(when).astimezone().strftime("%Y-%m-%d")        # folders by local day
        folder = self.storage_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for data, suffix in ((snapshot, ""), (frame, "_frame")):
            if data is None:
                out.append(None)
                continue
            path = folder / f"{event_id}{suffix}.jpg"
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
            out.append(f"{day}/{path.name}")
        return out[0], out[1]

    def add_event(self, ev, *, run_id: str | None, channels: list[str], snapshot: bytes | None = None,
                  frame: bytes | None = None, confirmed_at: datetime | None = None, event_id: str | None = None) -> str:
        """Store one event with its images, and queue one alert per channel, in one transaction.
        Pass the same `event_id` when retrying, so a retry can't store the event twice."""
        event_id = event_id or str(uuid.uuid4())
        with self.engine.connect() as c:
            if c.execute(select(events.c.id).where(events.c.id == event_id)).first() is not None:
                return event_id                   # already stored by an earlier attempt
        # the frame's own wall-clock time when given (to the microsecond); ev.time is to the second
        confirmed = utc(confirmed_at) if confirmed_at else parse_time(ev.time) if ev.time else now_utc()
        started = confirmed - timedelta(seconds=max(0.0, float(ev.confirmed) - float(ev.started)))
        snap, fr = self.save_images(event_id, confirmed, snapshot, frame)
        now = now_utc()
        with self.engine.begin() as c:
            c.execute(events.insert().values(
                id=event_id, run_id=run_id, run_event_id=ev.event_id, camera=ev.camera, rule=ev.rule or ev.kind,
                kind=ev.kind, severity=ev.severity or "medium", zone=ev.zone, track_id=int(ev.track_id),
                started_at=started, confirmed_at=confirmed, share=float(ev.share),
                box=[round(float(v), 4) for v in ev.box], snapshot=snap, frame=fr, images_deleted=False,
                status="new", created_at=now))
            for ch in channels:
                c.execute(alerts.insert().values(event_id=event_id, channel=ch, camera=ev.camera, status="pending",
                                                 attempts=0, next_try_at=now, done=[], is_summary=False,
                                                 created_at=now))
            n = c.execute(update(cameras).where(cameras.c.id == ev.camera)
                          .values(events=func.coalesce(cameras.c.events, 0) + 1)).rowcount
            if not n:
                c.execute(cameras.insert().values(id=ev.camera, name=ev.camera, events=1))
        return event_id

    def camera_state(self, camera: str, state: str, detail: str = "", when: datetime | None = None) -> None:
        when = utc(when) or now_utc()
        with self.engine.begin() as c:
            c.execute(camera_log.insert().values(camera=camera, time=when, state=state, detail=detail or None))
            n = c.execute(update(cameras).where(cameras.c.id == camera)
                          .values(state=state, state_since=when, detail=detail or None, last_seen=when)).rowcount
            if not n:
                c.execute(cameras.insert().values(id=camera, name=camera, state=state, state_since=when,
                                                  detail=detail or None, last_seen=when, events=0))

    def camera_stats(self, rows: list[dict]) -> None:
        """The camera service's status table (MultiCameraMonitor.status_rows), every few seconds."""
        now = now_utc()
        with self.engine.begin() as c:
            for r in rows:
                fps = float(r.get("fps") or 0.0)
                lag = r.get("lag_ms")
                lag = None if lag is None or lag != lag else float(lag)
                c.execute(update(cameras).where(cameras.c.id == r["camera"])
                          .values(state=r.get("state"), fps=fps, lag_ms=lag, last_seen=now))

    def heartbeat(self, name: str, detail: dict | None = None) -> None:
        now = now_utc()
        with self.engine.begin() as c:
            n = c.execute(update(services).where(services.c.name == name).values(last_seen=now, detail=detail or {})).rowcount
            if not n:
                c.execute(services.insert().values(name=name, last_seen=now, detail=detail or {}))

    # -- the dashboard --------------------------------------------------------------------------
    @staticmethod
    def _event_dict(r) -> dict:
        return {"id": r.id, "camera": r.camera, "rule": r.rule, "kind": r.kind, "severity": r.severity,
                "zone": r.zone, "track_id": r.track_id, "started_at": iso(r.started_at),
                "confirmed_at": iso(r.confirmed_at), "share": r.share, "box": r.box,
                "has_snapshot": bool(r.snapshot) and not r.images_deleted,
                "has_frame": bool(r.frame) and not r.images_deleted,
                "status": r.status, "status_at": iso(r.status_at), "note": r.note, "run_id": r.run_id}

    def _filters(self, camera=None, kind=None, status=None, since=None, until=None, severity=None):
        conds = []
        if camera:
            conds.append(events.c.camera.in_(camera if isinstance(camera, (list, tuple)) else [camera]))
        if kind:
            conds.append(events.c.kind.in_(kind if isinstance(kind, (list, tuple)) else [kind]))
        if status:
            conds.append(events.c.status.in_(status if isinstance(status, (list, tuple)) else [status]))
        if severity:
            conds.append(events.c.severity.in_(severity if isinstance(severity, (list, tuple)) else [severity]))
        if since is not None:
            conds.append(events.c.confirmed_at >= parse_time(since))
        if until is not None:
            conds.append(events.c.confirmed_at < parse_time(until))
        return and_(*conds) if conds else None

    def list_events(self, *, limit: int = 50, offset: int = 0, **filters) -> tuple[list[dict], int]:
        where = self._filters(**filters)
        q = select(events)
        cq = select(func.count()).select_from(events)
        if where is not None:
            q, cq = q.where(where), cq.where(where)
        q = q.order_by(events.c.confirmed_at.desc(), events.c.id).limit(max(1, min(limit, 500))).offset(max(0, offset))
        with self.engine.connect() as c:
            rows = [self._event_dict(r) for r in c.execute(q)]
            total = c.execute(cq).scalar_one()
            by_event: dict[str, list] = {}
            if rows:
                for a in c.execute(select(alerts.c.event_id, alerts.c.channel, alerts.c.status, alerts.c.latency_ms,
                                          alerts.c.is_summary, alerts.c.summary_id)
                                   .where(alerts.c.event_id.in_([r["id"] for r in rows]), alerts.c.is_summary.is_(False))):
                    by_event.setdefault(a.event_id, []).append(
                        {"channel": a.channel, "status": a.status, "latency_ms": a.latency_ms})
            for r in rows:
                r["alerts"] = by_event.get(r["id"], [])
        return rows, int(total)

    def get_event(self, event_id: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(select(events).where(events.c.id == event_id)).first()
            if r is None:
                return None
            d = self._event_dict(r)
            d["alerts"] = [self._alert_dict(a) for a in
                           c.execute(select(alerts).where(alerts.c.event_id == event_id).order_by(alerts.c.id))]
            cam = c.execute(select(cameras.c.name).where(cameras.c.id == r.camera)).first()
            d["camera_name"] = cam.name if cam else r.camera
        return d

    def image_path(self, event_id: str, which: str = "snapshot") -> Path | None:
        col = events.c.snapshot if which == "snapshot" else events.c.frame
        with self.engine.connect() as c:
            r = c.execute(select(col, events.c.images_deleted).where(events.c.id == event_id)).first()
        if r is None or not r[0] or r[1]:
            return None
        p = self.storage_dir / r[0]
        return p if p.is_file() else None

    def set_status(self, event_id: str, status: str, note: str | None = None) -> dict | None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        values = {"status": status, "status_at": now_utc()}
        if note is not None:
            values["note"] = note.strip()[:2000] or None
        with self.engine.begin() as c:
            n = c.execute(update(events).where(events.c.id == event_id).values(**values)).rowcount
        return self.get_event(event_id) if n else None

    def stats(self, since: datetime, until: datetime, **filters) -> dict:
        """Counts for the dashboard: per hour (per day over 3 days or more), per camera, kind, status."""
        since, until = parse_time(since), parse_time(until)
        where = self._filters(since=since, until=until, **filters)
        q = select(events.c.confirmed_at, events.c.camera, events.c.kind, events.c.status, events.c.severity).where(where)
        with self.engine.connect() as c:
            rows = list(c.execute(q))
        step = timedelta(hours=1) if until - since <= timedelta(days=3) else timedelta(days=1)
        start = since.replace(minute=0, second=0, microsecond=0)
        if step == timedelta(days=1):
            start = start.replace(hour=0)
        buckets, t = [], start
        while t < until and len(buckets) < 400:
            buckets.append(t)
            t += step
        counts = Counter()
        for r in rows:
            k = int((utc(r.confirmed_at) - start) // step)
            counts[k] += 1
        reviewed = sum(1 for r in rows if r.status != "new")
        false = sum(1 for r in rows if r.status == "false_alarm")
        return {
            "since": iso(since), "until": iso(until), "total": len(rows),
            "step": "hour" if step == timedelta(hours=1) else "day",
            "series": [{"t": iso(b), "n": counts.get(i, 0)} for i, b in enumerate(buckets)],
            "by_camera": dict(Counter(r.camera for r in rows).most_common()),
            "by_kind": dict(Counter(r.kind for r in rows).most_common()),
            "by_status": {s: sum(1 for r in rows if r.status == s) for s in STATUSES},
            "by_severity": dict(Counter(r.severity for r in rows).most_common()),
            "false_alarm_share": (false / reviewed) if reviewed else None,
        }

    def cameras(self) -> list[dict]:
        with self.engine.connect() as c:
            return [{"id": r.id, "name": r.name, "state": r.state, "state_since": iso(r.state_since), "detail": r.detail,
                     "fps": r.fps, "lag_ms": r.lag_ms, "last_seen": iso(r.last_seen), "events": r.events or 0}
                    for r in c.execute(select(cameras).order_by(cameras.c.id))]

    def services(self) -> dict[str, dict]:
        with self.engine.connect() as c:
            return {r.name: {"last_seen": iso(r.last_seen), "detail": r.detail or {}} for r in c.execute(select(services))}

    def rules(self) -> list[dict]:
        with self.engine.connect() as c:
            return [dict(r._mapping) | {"updated_at": iso(r.updated_at)} for r in c.execute(select(rules).order_by(rules.c.id))]

    # -- the alert worker -----------------------------------------------------------------------
    @staticmethod
    def _alert_dict(a) -> dict:
        return {"id": a.id, "event_id": a.event_id, "channel": a.channel, "camera": a.camera, "status": a.status,
                "attempts": a.attempts, "next_try_at": iso(a.next_try_at), "last_error": a.last_error,
                "done": a.done or [], "is_summary": a.is_summary, "summary_id": a.summary_id,
                "summary_count": a.summary_count, "created_at": iso(a.created_at), "held_at": iso(a.held_at),
                "sent_at": iso(a.sent_at), "latency_ms": a.latency_ms}

    def recover_stuck(self, older_than: float = 60.0) -> int:
        """Alerts left 'sending' by a worker that stopped mid-way go back to the queue."""
        cutoff = now_utc() - timedelta(seconds=older_than)
        with self.engine.begin() as c:
            return c.execute(update(alerts).where(alerts.c.status == "sending", alerts.c.next_try_at < cutoff)
                             .values(status="pending")).rowcount

    def claim_alerts(self, limit: int = 10) -> list[dict]:
        """Take up to `limit` due alerts off the queue (status -> sending), oldest first, with their event.
        On PostgreSQL, rows another worker is taking at the same moment are skipped, not waited for."""
        now = now_utc()
        with self.engine.begin() as c:
            q = (select(alerts.c.id).where(alerts.c.status == "pending", alerts.c.next_try_at <= now)
                 .order_by(alerts.c.id).limit(limit))
            if self.is_postgres:
                q = q.with_for_update(skip_locked=True)
            ids = [r.id for r in c.execute(q)]
            if not ids:
                return []
            c.execute(update(alerts).where(alerts.c.id.in_(ids), alerts.c.status == "pending")
                      .values(status="sending", next_try_at=now))     # `now` marks this worker's claim
            rows = c.execute(select(alerts, events.c.kind, events.c.severity, events.c.rule, events.c.zone,
                                    events.c.track_id, events.c.confirmed_at, events.c.started_at,
                                    events.c.snapshot, events.c.images_deleted, cameras.c.name.label("camera_name"))
                             .select_from(alerts.join(events, alerts.c.event_id == events.c.id)
                                          .outerjoin(cameras, cameras.c.id == alerts.c.camera))
                             .where(alerts.c.id.in_(ids), alerts.c.status == "sending", alerts.c.next_try_at == now)
                             .order_by(alerts.c.id)).all()
        out = []
        for r in rows:
            d = self._alert_dict(r)
            d.update(kind=r.kind, severity=r.severity, rule=r.rule, zone=r.zone, track_id=r.track_id,
                     confirmed_at=utc(r.confirmed_at), started_at=utc(r.started_at),
                     snapshot=None if r.images_deleted or not r.snapshot else self.storage_dir / r.snapshot,
                     camera_name=r.camera_name or r.camera)
            out.append(d)
        return out

    def sent_recently(self, channel: str, camera: str, since: datetime) -> int:
        with self.engine.connect() as c:
            return int(c.execute(select(func.count()).select_from(alerts).where(
                alerts.c.channel == channel, alerts.c.camera == camera, alerts.c.status == "sent",
                alerts.c.sent_at >= since)).scalar_one())

    def mark_sent(self, alert_id: int, confirmed_at: datetime | None, done: list[str]) -> float | None:
        now = now_utc()
        latency = (now - utc(confirmed_at)).total_seconds() * 1000 if confirmed_at else None
        with self.engine.begin() as c:
            c.execute(update(alerts).where(alerts.c.id == alert_id).values(
                status="sent", sent_at=now, latency_ms=latency, last_error=None, done=done,
                attempts=alerts.c.attempts + 1))
        return latency

    def mark_retry(self, alert_id: int, error: str, wait: float, done: list[str], give_up: bool) -> None:
        with self.engine.begin() as c:
            c.execute(update(alerts).where(alerts.c.id == alert_id).values(
                status="failed" if give_up else "pending", last_error=error[:1000], done=done,
                attempts=alerts.c.attempts + 1, next_try_at=now_utc() + timedelta(seconds=wait)))

    def mark_failed(self, alert_id: int, error: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(alerts).where(alerts.c.id == alert_id).values(status="failed", last_error=error[:1000]))

    def mark_held(self, alert_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(update(alerts).where(alerts.c.id == alert_id).values(status="held", held_at=now_utc()))

    def held_groups(self) -> list[dict]:
        """Held alerts, grouped by channel and camera: [{channel, camera, ids, kinds, newest event}]."""
        with self.engine.connect() as c:
            rows = c.execute(select(alerts.c.id, alerts.c.channel, alerts.c.camera, alerts.c.held_at, events.c.id.label("eid"),
                                    events.c.kind, events.c.confirmed_at, events.c.snapshot, events.c.images_deleted,
                                    cameras.c.name.label("camera_name"))
                             .select_from(alerts.join(events, alerts.c.event_id == events.c.id)
                                          .outerjoin(cameras, cameras.c.id == alerts.c.camera))
                             .where(alerts.c.status == "held").order_by(alerts.c.id)).all()
        groups: dict[tuple, dict] = {}
        for r in rows:
            g = groups.setdefault((r.channel, r.camera), {"channel": r.channel, "camera": r.camera,
                                                          "camera_name": r.camera_name or r.camera, "ids": [],
                                                          "kinds": Counter(), "first": utc(r.confirmed_at)})
            g["ids"].append(r.id)
            g["kinds"][r.kind] += 1
            g["newest_event"] = r.eid
            g["last"] = utc(r.confirmed_at)
            g["snapshot"] = None if r.images_deleted or not r.snapshot else self.storage_dir / r.snapshot
        return list(groups.values())

    def record_summary(self, group: dict, done: list[str]) -> int:
        now = now_utc()
        with self.engine.begin() as c:
            sid = c.execute(alerts.insert().values(
                event_id=group["newest_event"], channel=group["channel"], camera=group["camera"], status="sent",
                attempts=1, next_try_at=now, done=done, is_summary=True, summary_count=len(group["ids"]),
                created_at=now, sent_at=now)).inserted_primary_key[0]
            c.execute(update(alerts).where(alerts.c.id.in_(group["ids"]), alerts.c.status == "held")
                      .values(status="summarised", summary_id=sid))
        return int(sid)

    def alert_counts(self, since: datetime | None = None) -> dict:
        q = select(alerts.c.status, func.count()).group_by(alerts.c.status)
        if since is not None:
            q = q.where(alerts.c.created_at >= since)
        with self.engine.connect() as c:
            return {s: int(n) for s, n in c.execute(q)}

    def latencies(self, since: datetime | None = None) -> list[float]:
        q = select(alerts.c.latency_ms).where(alerts.c.status == "sent", alerts.c.is_summary.is_(False),
                                              alerts.c.latency_ms.is_not(None))
        if since is not None:
            q = q.where(alerts.c.sent_at >= since)
        with self.engine.connect() as c:
            return [float(r[0]) for r in c.execute(q)]

    # -- housekeeping ---------------------------------------------------------------------------
    def cleanup_images(self, keep_days: float, now: datetime | None = None) -> int:
        """Delete evidence images older than keep_days, except those of events marked false alarm
        (kept for retraining). The event rows stay."""
        cutoff = (utc(now) or now_utc()) - timedelta(days=keep_days)
        with self.engine.begin() as c:
            rows = c.execute(select(events.c.id, events.c.snapshot, events.c.frame).where(
                events.c.confirmed_at < cutoff, events.c.images_deleted.is_(False),
                events.c.status != "false_alarm")).all()
            for r in rows:
                for rel in (r.snapshot, r.frame):
                    if rel:
                        (self.storage_dir / rel).unlink(missing_ok=True)
            if rows:
                c.execute(update(events).where(events.c.id.in_([r.id for r in rows])).values(images_deleted=True))
        for day in self.storage_dir.glob("*"):
            if day.is_dir() and not any(day.iterdir()):
                day.rmdir()
        return len(rows)


def seconds_since(iso_time: str | None, now: datetime | None = None) -> float | None:
    if not iso_time:
        return None
    return ((now or datetime.now(timezone.utc)) - parse_time(iso_time)).total_seconds()
