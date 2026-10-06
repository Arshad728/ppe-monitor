"""Phase 5: storage, the alert queue and worker, the Telegram channel, the API.

Runs on SQLite. Set PPE_TEST_DATABASE_URL to a PostgreSQL database (it is emptied!) to run the
storage and worker tests there too; scripts/mac_phase5.sh check does, with a throw-away database.
"""

import base64
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy", reason="Phase 5 libraries not installed: pip install -r requirements-server.txt")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from ppe_monitor.backend.channels import Message, SendError, TelegramChannel, event_message
from ppe_monitor.backend.db import metadata
from ppe_monitor.backend.settings import ChannelSettings, Secrets, ServerSettings, read_env_file, write_env_value
from ppe_monitor.backend.store import EventStore
from ppe_monitor.backend.worker import AlertWorker
from ppe_monitor.rules.events import Event

PG_URL = os.environ.get("PPE_TEST_DATABASE_URL", "")


@pytest.fixture(params=["sqlite"] + (["postgresql"] if PG_URL else []))
def store(request, tmp_path):
    url = f"sqlite:///{tmp_path / 't.sqlite'}" if request.param == "sqlite" else PG_URL
    s = EventStore(url, tmp_path / "events", create=False)
    metadata.drop_all(s.engine)
    from ppe_monitor.backend.db import create_schema
    create_schema(s.engine)
    yield s
    s.close()


def settings(tmp_path, **kw) -> ServerSettings:
    s = ServerSettings(storage_dir=tmp_path / "events", live_dir=tmp_path / "live",
                       channels={"telegram": ChannelSettings("telegram", True, "medium"),
                                 "email": ChannelSettings("email", False, "high")},
                       secrets=Secrets(tmp_path / "none.env", environ={}))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def event(camera="cam1", kind="no_helmet", severity="high", when=None, track=3) -> Event:
    when = when or datetime.now(timezone.utc)
    return Event(kind, camera, track, 10.0, 13.0, (0.2, 0.2, 0.4, 0.9), 0.9, 130, event_id=f"{camera}-{kind}-{track}",
                 rule=kind, severity=severity, time=when.isoformat(timespec="seconds"))


JPEG = b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"


class FakeChannel:
    name = "telegram"

    def __init__(self, fail=0, permanent=False, retry_after=None, configured=True):
        self.sent, self.fail, self.permanent, self.retry_after, self.configured = [], fail, permanent, retry_after, configured

    def missing(self):
        return "" if self.configured else "TELEGRAM_BOT_TOKEN is not set"

    def send(self, msg, done):
        if self.fail:
            self.fail -= 1
            raise SendError("telegram: boom", permanent=self.permanent, retry_after=self.retry_after)
        self.sent.append(msg)
        return ["123"]


# -- settings ----------------------------------------------------------------------------------
def test_secrets_file_and_environment(tmp_path):
    f = tmp_path / "secrets.env"
    f.write_text("# comment\nTELEGRAM_BOT_TOKEN='abc:def'\nTELEGRAM_CHAT_ID=1, 2\nEMPTY=\n")
    assert read_env_file(f)["TELEGRAM_BOT_TOKEN"] == "abc:def"
    s = Secrets(f, environ={"TELEGRAM_CHAT_ID": "9"})
    assert s.get("TELEGRAM_BOT_TOKEN") == "abc:def" and s.list("TELEGRAM_CHAT_ID") == ["9"]   # the environment wins
    write_env_value(f, "TELEGRAM_CHAT_ID", "5")
    write_env_value(f, "NEW", "x")
    text = f.read_text()
    assert "# comment" in text and "TELEGRAM_CHAT_ID=5" in text and text.rstrip().endswith("NEW=x")
    assert oct(f.stat().st_mode & 0o777) == "0o600"


def test_server_settings_and_database_url(tmp_path):
    s = ServerSettings.load(secrets=Secrets(tmp_path / "x.env", environ={"PPE_DB_PASSWORD": "p@ss/word"}))
    assert s.database_url == "postgresql+psycopg://ppe:p%40ss%2Fword@127.0.0.1:5433/ppe"
    assert "p%40ss" not in s.safe_database_url
    assert s.channels["telegram"].wants("high") and not s.channels["telegram"].wants("low")
    assert not s.channels["email"].wants("critical")                     # disabled
    s2 = ServerSettings.load(secrets=Secrets(tmp_path / "x.env", environ={"DATABASE_URL": "sqlite:///data/x.sqlite"}))
    assert s2.database_url.endswith("/data/x.sqlite") and s2.database_url.startswith("sqlite:////")


# -- storage -----------------------------------------------------------------------------------
def test_events_are_stored_with_images_and_queued_alerts(store, tmp_path):
    store.sync_config({"cam1": "Main gate"})
    eid = store.add_event(event(), run_id="r1", channels=["telegram"], snapshot=JPEG, frame=JPEG)
    assert store.add_event(event(), run_id="r1", channels=["telegram"], event_id=eid) == eid   # a retry: no duplicate
    items, total = store.list_events()
    assert total == 1 and items[0]["kind"] == "no_helmet" and items[0]["has_snapshot"]
    assert items[0]["alerts"] == [{"channel": "telegram", "status": "pending", "latency_ms": None}]
    ev = store.get_event(eid)
    assert ev["camera_name"] == "Main gate" and ev["alerts"][0]["status"] == "pending"
    assert store.image_path(eid).read_bytes() == JPEG
    started = datetime.fromisoformat(ev["started_at"].replace("Z", "+00:00"))
    confirmed = datetime.fromisoformat(ev["confirmed_at"].replace("Z", "+00:00"))
    assert (confirmed - started).total_seconds() == pytest.approx(3.0)
    assert store.set_status(eid, "false_alarm", " reflection ")["status"] == "false_alarm"
    assert store.get_event(eid)["note"] == "reflection"
    with pytest.raises(ValueError):
        store.set_status(eid, "maybe")
    assert store.cameras()[0]["events"] == 1


def test_filters_and_stats(store):
    now = datetime.now(timezone.utc)
    for i, (cam, kind) in enumerate([("cam1", "no_helmet"), ("cam1", "zone_intrusion"), ("cam2", "no_helmet")]):
        store.add_event(event(cam, kind, when=now - timedelta(minutes=10 * i), track=i), run_id=None, channels=[])
    store.add_event(event("cam2", when=now - timedelta(days=3)), run_id=None, channels=[])
    items, total = store.list_events(camera="cam1", since=now - timedelta(hours=1))
    assert total == 2 and [e["kind"] for e in items] == ["no_helmet", "zone_intrusion"]    # newest first
    assert store.list_events(kind=["zone_intrusion"])[1] == 1
    st = store.stats(now - timedelta(hours=24), now + timedelta(minutes=1))
    assert st["total"] == 3 and st["by_camera"] == {"cam1": 2, "cam2": 1} and st["step"] == "hour"
    assert sum(b["n"] for b in st["series"]) == 3 and len(st["series"]) == 25
    assert st["false_alarm_share"] is None
    assert store.stats(now - timedelta(days=7), now)["step"] == "day"


def test_old_images_are_deleted_but_false_alarms_kept(store, tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=40)
    a = store.add_event(event(when=old, track=1), run_id=None, channels=[], snapshot=JPEG, frame=JPEG)
    b = store.add_event(event(when=old, track=2), run_id=None, channels=[], snapshot=JPEG, frame=JPEG)
    c = store.add_event(event(track=3), run_id=None, channels=[], snapshot=JPEG)
    store.set_status(b, "false_alarm")
    assert store.cleanup_images(30) == 1
    assert store.image_path(a) is None and store.image_path(b) is not None and store.image_path(c) is not None
    assert not store.get_event(a)["has_snapshot"]


# -- the alert worker --------------------------------------------------------------------------
def test_worker_sends_and_records_the_delay(store, tmp_path):
    ch = FakeChannel()
    w = AlertWorker(store, settings(tmp_path), {"telegram": ch}, log=lambda *a: None)
    store.sync_config({"cam1": "Main gate"})
    eid = store.add_event(event(when=datetime.now(timezone.utc) - timedelta(seconds=2)), run_id=None,
                          channels=["telegram"], snapshot=JPEG)
    assert w.process() == 1 and len(ch.sent) == 1
    assert "No helmet" in ch.sent[0].title and "Main gate" in ch.sent[0].title and ch.sent[0].photo == JPEG
    a = store.get_event(eid)["alerts"][0]
    assert a["status"] == "sent" and a["latency_ms"] >= 2000 and a["done"] == ["123"]
    assert w.process() == 0                                            # nothing sent twice


def test_text_only_channel_keeps_the_picture_home(store, tmp_path):
    s = settings(tmp_path)
    s.channels["telegram"].photo = False
    ch = FakeChannel()
    w = AlertWorker(store, s, {"telegram": ch}, log=lambda *a: None)
    store.add_event(event(), run_id=None, channels=["telegram"], snapshot=JPEG)
    w.process()
    assert len(ch.sent) == 1 and ch.sent[0].photo is None


def test_worker_retries_then_gives_up_or_fails_at_once(store, tmp_path):
    s = settings(tmp_path, retry_first=0.01, retry_attempts=3)
    ch = FakeChannel(fail=1)
    w = AlertWorker(store, s, {"telegram": ch}, log=lambda *a: None)
    eid = store.add_event(event(), run_id=None, channels=["telegram"])
    w.process()
    a = store.get_event(eid)["alerts"][0]
    assert a["status"] == "pending" and a["attempts"] == 1 and "boom" in a["last_error"]
    time.sleep(0.05)
    w.process()
    assert store.get_event(eid)["alerts"][0]["status"] == "sent" and len(ch.sent) == 1
    # a wrong token or chat: no point retrying
    w.channels["telegram"] = FakeChannel(fail=5, permanent=True)
    e2 = store.add_event(event(track=9), run_id=None, channels=["telegram"])
    w.process()
    assert store.get_event(e2)["alerts"][0]["status"] == "failed"
    # not set up at all
    w.channels["telegram"] = FakeChannel(configured=False)
    e3 = store.add_event(event(track=10), run_id=None, channels=["telegram"])
    w.process()
    assert "not set" in store.get_event(e3)["alerts"][0]["last_error"]


def test_rate_limit_holds_then_summarises(store, tmp_path):
    s = settings(tmp_path, rate_max=3, rate_minutes=0.02)               # 3 per 1.2 s
    ch = FakeChannel()
    w = AlertWorker(store, s, {"telegram": ch}, log=lambda *a: None)
    ids = [store.add_event(event(kind="no_helmet" if i % 2 else "zone_intrusion", track=i), run_id=None,
                           channels=["telegram"], snapshot=JPEG) for i in range(7)]
    store.add_event(event("cam2", track=99), run_id=None, channels=["telegram"])   # another camera: its own limit
    while w.process():
        pass
    statuses = Counter(store.get_event(i)["alerts"][0]["status"] for i in ids)
    assert statuses == {"sent": 3, "held": 4} and len(ch.sent) == 4          # + cam2's
    assert w.send_summaries() == 0                                           # still over the limit
    time.sleep(1.3)
    assert w.send_summaries() == 1
    assert "4 more alerts" in ch.sent[-1].title
    statuses = Counter(store.get_event(i)["alerts"][0]["status"] for i in ids)
    assert statuses == {"sent": 3, "summarised": 4}
    assert store.alert_counts()["sent"] == 5                                  # 4 alerts + 1 summary


def test_a_worker_that_stopped_mid_send_leaves_nothing_stuck(store, tmp_path):
    eid = store.add_event(event(), run_id=None, channels=["telegram"])
    assert len(store.claim_alerts()) == 1 and store.claim_alerts() == []     # taken: not given out twice
    assert store.recover_stuck(older_than=-1) == 1
    assert store.get_event(eid)["alerts"][0]["status"] == "pending"


# -- Telegram, against a stand-in for its API ----------------------------------------------------
def telegram(handler, chats=("111", "222")) -> TelegramChannel:
    return TelegramChannel("42:TOKEN", list(chats), api_base="https://tg.test",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_telegram_sends_the_photo_to_each_chat():
    calls = []

    def handler(req: httpx.Request):
        calls.append((req.url.path, req.content))
        return httpx.Response(200, json={"ok": True, "result": {}})

    done = telegram(handler).send(Message("T <b>", ["line & more"], JPEG, "http://x/#event=1"), [])
    assert done == ["111", "222"] and [c[0] for c in calls] == ["/bot42:TOKEN/sendPhoto"] * 2
    body = calls[0][1]
    assert b'name="chat_id"\r\n\r\n111' in body and b"fakejpeg" in body and b"T &lt;b&gt;" in body


def test_pictures_are_shrunk_for_sending_and_telegram_uses_ipv4():
    import cv2

    from ppe_monitor.backend.channels import http_client, shrink
    big = cv2.imencode(".jpg", cv2.GaussianBlur(np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8),
                                               (0, 0), 3))[1].tobytes()
    small = shrink(big)
    img = cv2.imdecode(np.frombuffer(small, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (720, 1280) and len(small) < len(big)
    assert shrink(JPEG) == JPEG                                              # not a picture: left alone
    pool = http_client("ipv4")._transport._pool
    assert pool._local_address == "0.0.0.0"


def test_telegram_errors_say_whether_to_retry():
    def busy(req):
        return httpx.Response(429, json={"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 7}})

    with pytest.raises(SendError) as e:
        telegram(busy).send(Message("t", []), [])
    assert not e.value.permanent and e.value.retry_after == 7

    def wrong_chat(req):
        return httpx.Response(400, json={"ok": False, "description": "Bad Request: chat not found"})

    with pytest.raises(SendError) as e:
        telegram(wrong_chat).send(Message("t", []), [])
    assert e.value.permanent

    seen = []

    def second_fails(req):
        seen.append(1)
        return httpx.Response(200, json={"ok": True}) if len(seen) == 1 else httpx.Response(502, text="bad gateway")

    with pytest.raises(SendError) as e:
        telegram(second_fails).send(Message("t", []), [])
    assert e.value.done == ["111"] and not e.value.permanent               # a retry won't message 111 again
    assert telegram(lambda r: httpx.Response(200, json={"ok": True})).send(Message("t", []), ["111"]) == ["111", "222"]


def test_event_message_reads_well(tmp_path):
    snap = tmp_path / "s.jpg"
    snap.write_bytes(JPEG)
    now = datetime.now(timezone.utc)
    m = event_message({"kind": "zone_intrusion", "zone": "pit", "severity": "critical", "camera": "cam3",
                       "camera_name": "Excavation", "track_id": 4, "rule": "pit-working-hours", "event_id": "e1",
                       "confirmed_at": now, "started_at": now - timedelta(seconds=2), "snapshot": snap},
                      "http://192.168.1.5:8080")
    assert "CRITICAL" in m.title and "Restricted zone (pit)" in m.title and "Excavation" in m.title
    assert "in the zone for 2 s" in m.lines[0] and m.link.endswith("#event=e1") and m.photo == JPEG
    assert len(m.telegram_html()) < 1024


# -- the background writer ---------------------------------------------------------------------
class FlakyStore:
    def __init__(self, fail):
        self.fail, self.events = fail, []

    def add_event(self, ev, **kw):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("database down")
        self.events.append((ev, kw))

    def camera_state(self, *a):
        pass


def test_db_sink_never_blocks_and_retries(tmp_path):
    from ppe_monitor.backend.sink import DbSink

    class R:                       # a FrameResult stand-in, enough for the snapshot
        tracks, verdicts, status, helmets, vests, poses, feet, inside, rules, active = [], {}, {}, [], [], {}, {}, {}, None, set()
        now = datetime.now(timezone.utc)

    store = FlakyStore(fail=2)
    sink = DbSink(store, settings(tmp_path), "run1", log=lambda *a: None)
    t0 = time.perf_counter()
    sink.event(event(), np.zeros((120, 160, 3), np.uint8), R())
    assert time.perf_counter() - t0 < 0.05                                  # the video loop doesn't wait
    deadline = time.time() + 10
    while not store.events and time.time() < deadline:
        time.sleep(0.05)
    sink.close()
    assert len(store.events) == 1 and sink.failures == 2
    kw = store.events[0][1]
    assert kw["channels"] == ["telegram"] and kw["snapshot"][:2] == b"\xff\xd8" and kw["confirmed_at"] == R.now


# -- the API -----------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from ppe_monitor.backend.api import create_app
    st = EventStore(f"sqlite:///{tmp_path / 'api.sqlite'}", tmp_path / "events")
    st.sync_config({"cam1": "Main gate", "cam2": "Scaffold"})
    s = settings(tmp_path)
    s.live_dir.mkdir(parents=True)
    (s.live_dir / "cam1.jpg").write_bytes(JPEG)
    ids = [st.add_event(event("cam1", track=1), run_id=None, channels=["telegram"], snapshot=JPEG, frame=JPEG),
           st.add_event(event("cam2", "zone_intrusion", "critical", track=2), run_id=None, channels=[])]
    st.heartbeat("alert_worker", {"channels": {"telegram": True}})
    with TestClient(create_app(st, s)) as c:
        c.ids, c.s = ids, s
        yield c
    st.close()


def test_api_overview_events_and_verdicts(client):
    o = client.get("/api/overview?hours=24").json()
    assert o["stats"]["total"] == 2 and {c["id"] for c in o["cameras"]} == {"cam1", "cam2"}
    assert o["services"]["alert_worker"]["age_s"] < 60 and o["alerts"] == {"pending": 1}
    r = client.get("/api/events?kind=zone_intrusion").json()
    assert r["total"] == 1 and r["items"][0]["severity"] == "critical"
    assert client.get("/api/events?kind=bogus").status_code == 400
    e = client.get(f"/api/events/{client.ids[0]}").json()
    assert e["camera_name"] == "Main gate" and e["alerts"][0]["channel"] == "telegram"
    r = client.patch(f"/api/events/{client.ids[0]}", json={"status": "false_alarm", "note": "shadow"})
    assert r.status_code == 200 and r.json()["status"] == "false_alarm"
    assert client.patch(f"/api/events/{client.ids[0]}", json={"status": "nope"}).status_code == 422
    assert client.get("/api/overview?hours=24").json()["stats"]["false_alarm_share"] == 1.0
    assert client.get(f"/api/events/{client.ids[0]}/snapshot.jpg").content == JPEG
    assert client.get(f"/api/events/{client.ids[1]}/frame.jpg").status_code == 404
    assert client.get("/api/events/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/api/events/..%2F..%2Fetc").status_code == 404


def test_api_live_tiles_and_page(client):
    assert client.get("/api/cameras/cam1/live.jpg").content == JPEG
    assert client.get("/api/cameras/cam2/live.jpg").status_code == 404
    assert client.get("/api/cameras/..%2Fx/live.jpg").status_code == 404
    page = client.get("/")
    assert page.status_code == 200 and "PPE monitor" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_api_password(tmp_path):
    from fastapi.testclient import TestClient

    from ppe_monitor.backend.api import create_app
    st = EventStore(f"sqlite:///{tmp_path / 'a.sqlite'}", tmp_path / "events")
    s = settings(tmp_path, secrets=Secrets(tmp_path / "x.env", environ={"DASHBOARD_PASSWORD": "s3cret"}))
    with TestClient(create_app(st, s)) as c:
        assert c.get("/api/events").status_code == 401 and c.get("/").status_code == 401
        good = base64.b64encode(b"safety:s3cret").decode()
        bad = base64.b64encode(b"safety:guess").decode()
        assert c.get("/api/events", headers={"Authorization": f"Basic {good}"}).status_code == 200
        assert c.get("/api/events", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    st.close()
