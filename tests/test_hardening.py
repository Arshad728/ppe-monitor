"""Phase 6 hardening: the system watches itself, and secrets can come from files (Docker)."""

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy", reason="Phase 5 libraries not installed: pip install -r requirements-server.txt")

from ppe_monitor.backend.db import create_schema
from ppe_monitor.backend.settings import Secrets, ServerSettings
from ppe_monitor.backend.store import EventStore
from ppe_monitor.backend.worker import AlertWorker
from test_backend import FakeChannel, settings


@pytest.fixture
def store(tmp_path):
    s = EventStore(f"sqlite:///{tmp_path / 'h.sqlite'}", tmp_path / "events", create=False)
    create_schema(s.engine)
    yield s
    s.close()


def at(minutes: float) -> datetime:
    return datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def worker(store, tmp_path, **kw):
    ch = FakeChannel()
    w = AlertWorker(store, settings(tmp_path, **kw), {"telegram": ch}, log=lambda m: None)
    return w, ch


def beat(store, when, stopped=False):
    store.heartbeat("camera_service", {"stopped": True} if stopped else {"cameras": 2})
    from sqlalchemy import update

    from ppe_monitor.backend.db import services
    with store.engine.begin() as c:
        c.execute(update(services).values(last_seen=when))


def test_a_camera_down_for_a_minute_raises_one_alert_and_one_when_back(store, tmp_path):
    w, ch = worker(store, tmp_path)
    store.sync_config({"cam1": "Main gate", "cam2": "Bay"})
    store.camera_state("cam1", "live", "", at(0))
    store.camera_state("cam2", "live", "", at(0))
    store.camera_state("cam1", "reconnecting", "could not open the stream", at(1))
    beat(store, at(1.5))
    assert w.watch_health(at(1.5)) == []                          # down for 30 s: not yet
    beat(store, at(2.2))
    titles = w.watch_health(at(2.2))
    assert titles == ["\U0001F4F7 Camera down: Main gate"]
    assert "could not open" in ch.sent[0].lines[0] and ch.sent[0].photo is None
    beat(store, at(3))
    assert w.watch_health(at(3)) == []                            # still down: no second alert
    store.camera_state("cam1", "live", "", at(4))
    beat(store, at(4.1))
    assert w.watch_health(at(4.1)) == ["✅ Back: Main gate"]
    assert len(ch.sent) == 2 and w.health_sent == 2


def test_cameras_that_go_down_together_share_one_message(store, tmp_path):
    w, ch = worker(store, tmp_path)
    for cam in ("a", "b", "c"):
        store.camera_state(cam, "reconnecting", "", at(0))
    beat(store, at(2))
    assert w.watch_health(at(2)) == ["\U0001F4F7 Cameras down: a, b, c"] and len(ch.sent) == 1


def test_a_silent_camera_service_raises_an_alert_but_a_deliberate_stop_does_not(store, tmp_path):
    w, ch = worker(store, tmp_path)
    store.camera_state("cam1", "live", "", at(0))
    beat(store, at(0))
    assert w.watch_health(at(0.5)) == []
    assert w.watch_health(at(1.5)) == ["⚠️ The camera service has stopped reporting"]
    assert w.watch_health(at(5)) == []                            # once per outage
    beat(store, at(6))
    assert w.watch_health(at(6)) == ["✅ The camera service is reporting again"]
    beat(store, at(7), stopped=True)
    assert w.watch_health(at(30)) == []                           # stopped on purpose
    assert len(ch.sent) == 2


def test_a_camera_stopped_on_purpose_is_not_down_and_health_alerts_can_be_switched_off(store, tmp_path):
    w, ch = worker(store, tmp_path, camera_down_after=0.0)
    store.camera_state("cam1", "reconnecting", "", at(0))
    beat(store, at(5))
    assert w.watch_health(at(5)) == []
    w2, _ = worker(store, tmp_path)
    store.camera_state("cam1", "stopped", "", at(0))
    assert w2.watch_health(at(5)) == []


def test_unsent_health_alerts_are_tried_again(store, tmp_path):
    w, ch = worker(store, tmp_path)
    ch.fail = 1
    store.camera_state("cam1", "reconnecting", "", at(0))
    beat(store, at(2))
    w.watch_health(at(2))
    assert ch.sent == [] and len(w._health_out) == 1
    beat(store, at(2.1))
    w.watch_health(at(2.1))
    assert len(ch.sent) == 1 and w._health_out == []


def test_secrets_from_files_and_docker_overrides(tmp_path):
    pw = tmp_path / "db_password"
    pw.write_text("s3cret\n")
    env = {"PPE_DB_PASSWORD_FILE": str(pw), "PPE_DB_HOST": "db", "PPE_DB_PORT": "5432", "DASHBOARD_HOST": "0.0.0.0"}
    s = ServerSettings.load(secrets=Secrets(tmp_path / "none.env", environ=env))
    assert s.database_url == "postgresql+psycopg://ppe:s3cret@db:5432/ppe"
    assert s.dashboard_host == "0.0.0.0" and s.camera_down_after == 60 and s.service_silent_after == 60
    env["PPE_DB_PASSWORD"] = "direct"                              # the variable itself wins over the file
    assert Secrets(tmp_path / "none.env", environ=env).get("PPE_DB_PASSWORD") == "direct"
    assert Secrets(tmp_path / "none.env", environ={"X_FILE": str(tmp_path / "missing")}).get("X", "d") == "d"
