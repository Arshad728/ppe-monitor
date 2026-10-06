"""The dashboard's API (FastAPI), and the dashboard page itself.

    GET   /api/overview            counts for a time range, the cameras, and the services' heartbeats
    GET   /api/events              events, newest first; filters: camera, kind, status, severity, since, until
    GET   /api/events/{id}         one event, with how each of its alerts went
    PATCH /api/events/{id}         the safety officer's verdict: {"status": "confirmed" | "false_alarm" | "new", "note": ...}
    GET   /api/events/{id}/snapshot.jpg   the evidence image (and /frame.jpg: the clean frame)
    GET   /api/cameras             every camera's state, rate and lag
    GET   /api/cameras/{id}/live.jpg      its newest annotated frame (refreshed about once a second)
    GET   /api/rules               the rules in force
    GET   /api/health              is the database reachable; when did the camera service and alert worker last report

"False alarm" is the button the book insists on: those events are kept, with their clean frames,
as hard examples for retraining (Phase 6).

When DASHBOARD_PASSWORD is set, every page and call needs it (HTTP basic authentication). It must
be set before the dashboard listens beyond this machine (scripts/serve_dashboard.py checks).
"""

from __future__ import annotations

import base64
import re
import secrets as pysecrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..dashboard import STATIC_DIR
from .settings import ServerSettings
from .store import KINDS, STATUSES, EventStore, iso, parse_time, seconds_since

_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EVENT_ID = re.compile(r"^[0-9a-f-]{36}$")


class Verdict(BaseModel):
    status: Literal["new", "confirmed", "false_alarm"]
    note: str | None = Field(default=None, max_length=2000)


def _split(value: str | None) -> list[str] | None:
    return [v for v in (value or "").split(",") if v] or None


def _range(since: str | None, until: str | None, hours: float | None) -> tuple[datetime, datetime]:
    try:
        end = parse_time(until) or datetime.now(timezone.utc)
        start = parse_time(since) or end - timedelta(hours=hours or 24)
    except ValueError as exc:
        raise HTTPException(400, f"bad time: {exc}") from exc
    if start >= end:
        raise HTTPException(400, "since must be before until")
    return start, end


def create_app(store: EventStore, settings: ServerSettings, *, static_dir: Path = STATIC_DIR) -> FastAPI:
    app = FastAPI(title="PPE monitor", version="0.5", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    user = settings.secrets.get("DASHBOARD_USER", "safety")
    password = settings.secrets.get("DASHBOARD_PASSWORD")

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        if password:
            ok = False
            header = request.headers.get("authorization", "")
            if header.lower().startswith("basic "):
                try:
                    u, _, p = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                    ok = pysecrets.compare_digest(u, user) & pysecrets.compare_digest(p, password)
                except (ValueError, UnicodeDecodeError):
                    ok = False
            if not ok:
                return Response("Sign in to the PPE dashboard", 401, {"WWW-Authenticate": 'Basic realm="PPE monitor"'})
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.get("/api/health")
    def health():
        try:
            svc = store.services()
            db_ok = True
        except Exception as exc:          # database down: say so rather than fail
            return JSONResponse({"ok": False, "database": f"unreachable: {type(exc).__name__}"}, 503)
        now = datetime.now(timezone.utc)
        return {"ok": db_ok, "database": store.engine.dialect.name, "now": iso(now),
                "services": {k: {**v, "age_s": seconds_since(v["last_seen"], now)} for k, v in svc.items()}}

    @app.get("/api/overview")
    def overview(since: str | None = None, until: str | None = None, hours: float | None = Query(None, gt=0, le=24 * 400),
                 camera: str | None = None):
        start, end = _range(since, until, hours)
        now = datetime.now(timezone.utc)
        svc = store.services()
        return {"now": iso(now), "stats": store.stats(start, end, camera=_split(camera)), "cameras": cameras(),
                "services": {k: {**v, "age_s": seconds_since(v["last_seen"], now)} for k, v in svc.items()},
                "alerts": store.alert_counts(start),
                "channels": {k: v.enabled for k, v in settings.channels.items()},
                "database": store.engine.dialect.name}

    @app.get("/api/events")
    def list_events(camera: str | None = None, kind: str | None = None, status: str | None = None,
                    severity: str | None = None, since: str | None = None, until: str | None = None,
                    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        for name, vals, allowed in (("kind", _split(kind), KINDS), ("status", _split(status), STATUSES)):
            bad = set(vals or []) - set(allowed)
            if bad:
                raise HTTPException(400, f"unknown {name}: {', '.join(sorted(bad))}")
        try:
            items, total = store.list_events(camera=_split(camera), kind=_split(kind), status=_split(status),
                                             severity=_split(severity), since=since or None, until=until or None,
                                             limit=limit, offset=offset)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def _event_or_404(event_id: str) -> dict:
        if not _EVENT_ID.match(event_id):
            raise HTTPException(404, "no such event")
        ev = store.get_event(event_id)
        if ev is None:
            raise HTTPException(404, "no such event")
        return ev

    @app.get("/api/events/{event_id}")
    def get_event(event_id: str):
        return _event_or_404(event_id)

    @app.patch("/api/events/{event_id}")
    def set_verdict(event_id: str, body: Verdict):
        _event_or_404(event_id)
        return store.set_status(event_id, body.status, body.note)

    def _image(event_id: str, which: str):
        _event_or_404(event_id)
        path = store.image_path(event_id, which)
        if path is None:
            raise HTTPException(404, "image not kept (older than storage.keep_days) or never saved")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/events/{event_id}/snapshot.jpg")
    def snapshot(event_id: str):
        return _image(event_id, "snapshot")

    @app.get("/api/events/{event_id}/frame.jpg")
    def frame(event_id: str):
        return _image(event_id, "frame")

    @app.get("/api/cameras")
    def cameras():
        out = []
        now = datetime.now(timezone.utc).timestamp()
        for c in store.cameras():
            tile = settings.live_dir / f"{c['id']}.jpg"
            age = now - tile.stat().st_mtime if tile.is_file() else None
            out.append({**c, "tile_age_s": age})
        return out

    @app.get("/api/cameras/{camera_id}/live.jpg")
    def live(camera_id: str):
        if not _CAMERA_ID.match(camera_id):
            raise HTTPException(404, "no such camera")
        tile = settings.live_dir / f"{camera_id}.jpg"
        if not tile.is_file():
            raise HTTPException(404, "no live picture (is the camera service running with --live?)")
        return Response(tile.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/rules")
    def list_rules():
        return store.rules()

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
