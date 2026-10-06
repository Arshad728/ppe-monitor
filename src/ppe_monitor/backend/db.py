"""The database schema (book, Chapter 22): cameras, zones, rules, events, and the alert queue.

PostgreSQL in normal use; SQLite works too (tests, or one machine without a database server).
The same tables, written with SQLAlchemy Core, serve both.

    runs        one row per start of the camera service (which model, which cameras)
    cameras     each camera's name, and its live state as last reported (fps, lag, up/down)
    camera_log  every time a camera dropped out or came back
    zones       the restricted zones of each camera (copied from configs/zones.yaml at start)
    rules       the rules in force (copied from configs/rules.yaml at start)
    events      every confirmed violation: what, where, when, which person, the evidence images,
                and the safety officer's verdict (new / confirmed / false alarm)
    alerts      the alert queue: one row per event and channel (Telegram, email). The alert
                worker takes rows from here, sends them, and records how it went. Being a table,
                the queue survives restarts, and the camera service never waits for a phone.
    services    heartbeats of the camera service and the alert worker, for the dashboard
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, MetaData, String,
                        Table, Text, create_engine, event)
from sqlalchemy.engine import Engine

SCHEMA_VERSION = 1
metadata = MetaData()
TS = DateTime(timezone=True)

schema_info = Table(
    "schema_info", metadata,
    Column("version", Integer, primary_key=True),
    Column("created_at", TS),
)

runs = Table(
    "runs", metadata,
    Column("id", String(36), primary_key=True),
    Column("started_at", TS, nullable=False),
    Column("host", String(200)),
    Column("backend", String(40)),
    Column("model", String(200)),
    Column("cameras", JSON),
)

cameras = Table(
    "cameras", metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(200)),
    Column("state", String(20)),              # live / connecting / reconnecting / stopped
    Column("state_since", TS),
    Column("detail", Text),                   # the last error, when not live
    Column("fps", Float),                     # frames analysed per second, as last reported
    Column("lag_ms", Float),
    Column("last_seen", TS),                  # when the camera service last reported on it
    Column("events", Integer, default=0),
)

camera_log = Table(
    "camera_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("camera", String(64), nullable=False, index=True),
    Column("time", TS, nullable=False),
    Column("state", String(20)),
    Column("detail", Text),
)

zones = Table(
    "zones", metadata,
    Column("camera", String(64), primary_key=True),
    Column("zone", String(64), primary_key=True),
    Column("name", String(200)),
    Column("polygon", JSON),                  # [[x, y], ...] in 0-1 fractions of the frame
    Column("updated_at", TS),
)

rules = Table(
    "rules", metadata,
    Column("id", String(64), primary_key=True),
    Column("type", String(32)),
    Column("severity", String(16)),
    Column("cameras", JSON),                  # null = every camera
    Column("zone", String(64)),
    Column("dwell", Float),
    Column("cooldown", Float),
    Column("active", JSON),
    Column("description", Text),
    Column("updated_at", TS),
)

events = Table(
    "events", metadata,
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36)),
    Column("run_event_id", String(120)),      # the camera service's own id for it
    Column("camera", String(64), nullable=False),
    Column("rule", String(64)),
    Column("kind", String(32), nullable=False),   # no_helmet / no_vest / zone_intrusion
    Column("severity", String(16)),
    Column("zone", String(64)),
    Column("track_id", Integer),
    Column("started_at", TS),                 # when the violation began
    Column("confirmed_at", TS, nullable=False),   # when it became an event (after the dwell)
    Column("share", Float),                   # share of violating votes when confirmed
    Column("box", JSON),                      # the person's box, 0-1 fractions of the frame
    Column("snapshot", String(300)),          # annotated evidence image, relative to storage.dir
    Column("frame", String(300)),             # the clean frame, for labelling
    Column("images_deleted", Boolean, default=False, nullable=False),
    Column("status", String(20), default="new", nullable=False),   # new / confirmed / false_alarm
    Column("status_at", TS),
    Column("note", Text),
    Column("created_at", TS, nullable=False),
)
Index("ix_events_confirmed_at", events.c.confirmed_at)
Index("ix_events_camera_time", events.c.camera, events.c.confirmed_at)
Index("ix_events_status", events.c.status)

alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String(36), ForeignKey("events.id", ondelete="CASCADE")),
    Column("channel", String(20), nullable=False),     # telegram / email
    Column("camera", String(64), nullable=False),      # copied from the event, for the rate limit
    # pending -> sending -> sent; or held (rate limit) -> summarised; or failed
    Column("status", String(20), nullable=False, default="pending"),
    Column("attempts", Integer, nullable=False, default=0),
    Column("next_try_at", TS),
    Column("last_error", Text),
    Column("done", JSON),                     # recipients already reached (a retry skips them)
    Column("is_summary", Boolean, nullable=False, default=False),
    Column("summary_id", Integer),            # held alerts: the summary that reported them
    Column("summary_count", Integer),         # summaries: how many held alerts they report
    Column("created_at", TS, nullable=False),
    Column("held_at", TS),
    Column("sent_at", TS),
    Column("latency_ms", Float),              # from the event's confirmation to delivery
)
Index("ix_alerts_queue", alerts.c.status, alerts.c.next_try_at)
Index("ix_alerts_rate", alerts.c.channel, alerts.c.camera, alerts.c.sent_at)
Index("ix_alerts_event", alerts.c.event_id)

services = Table(
    "services", metadata,
    Column("name", String(40), primary_key=True),      # camera_service / alert_worker
    Column("last_seen", TS),
    Column("detail", JSON),
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utc(dt: datetime | None) -> datetime | None:
    """Times read back from SQLite lose their time zone; they were written in UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def make_engine(url: str, echo: bool = False) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, echo=echo, connect_args={"check_same_thread": False, "timeout": 10})

        @event.listens_for(engine, "connect")
        def _pragmas(conn, _record):   # several processes share the file: WAL, and wait on locks
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        return engine
    return create_engine(url, echo=echo, pool_pre_ping=True, pool_size=5, max_overflow=5)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)
    with engine.begin() as conn:
        if conn.execute(schema_info.select()).first() is None:
            conn.execute(schema_info.insert().values(version=SCHEMA_VERSION, created_at=now_utc()))
