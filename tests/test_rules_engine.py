"""The rules engine, driven by hand-written people and tracks (book, Chapter 20: no video needed).

A 16:9 camera "yard" with one zone, "pit", covering the lower right quarter of the frame. People
are made-up detections: a box 0.3 of the frame tall whose bottom edge is where they stand.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from ppe_monitor.config import ConfigError
from ppe_monitor.pipeline import PPEMonitor, Settings
from ppe_monitor.rules.engine import CameraRules, Rule, RuleSet, parse_schedule
from ppe_monitor.rules.events import EventEngine
from ppe_monitor.rules.zones import CameraZones, Zone
from ppe_monitor.vision.matching import Det

FPS = 15
FRAME = np.zeros((720, 1280, 3), np.uint8)
PIT = Zone("pit", "Excavation pit", ((0.5, 0.5), (1.0, 0.5), (1.0, 0.95), (0.5, 0.95)))
H, W = 0.30, 0.06
MONDAY_NOON = datetime(2026, 9, 21, 12, 0, 0).astimezone()


class FakeModel:
    names = {0: "person", 1: "helmet", 2: "vest"}


def rules(*extra: Rule, zones=(PIT,), camera="yard") -> RuleSet:
    base = [Rule("no_helmet", "no_helmet", severity="high"), Rule("no_vest", "no_vest")]
    return RuleSet(base + list(extra), {camera: CameraZones(camera, list(zones))})


INTRUSION = Rule("pit-no-entry", "zone_intrusion", ("yard",), "pit", "critical", dwell=2.0)


def person(fx, fy, helmet=True, vest=True):
    """Detections for an upright worker standing at (fx, fy): feet = bottom-centre of the box."""
    x, y = fx - W / 2, fy - H
    dets = [Det(0, (x, y, x + W, fy), 0.9)]
    if helmet:
        dets.append(Det(1, (x + 0.01, y - 0.01, x + W - 0.01, y + 0.05), 0.8))
    if vest:
        dets.append(Det(2, (x + 0.005, y + 0.07, x + W - 0.005, y + 0.17), 0.8))
    return dets


def run(frames, ruleset, start=MONDAY_NOON, camera="yard"):
    """frames: one list of detections per frame at 15 fps. Returns the events."""
    m = PPEMonitor(FakeModel(), camera, FPS, Settings.load(), pose_model=False, rules=ruleset, start_time=start)
    events = []
    for k, dets in enumerate(frames):
        events += m.process(FRAME, k / FPS, detections=dets).events
    return events


def walk(path, seconds, **kw):
    """path(t) -> (fx, fy) feet position; one frame list per 1/15 s."""
    return [person(*path(k / FPS), **kw) for k in range(int(seconds * FPS))]


# -- zone intrusion ------------------------------------------------------------------------------
def test_walking_into_the_zone_and_staying_gives_one_event_after_the_dwell():
    # walks right along y = 0.8, crossing the zone edge (x = 0.5) at t = 4 s, then stops at x = 0.7
    frames = walk(lambda t: (min(0.3 + 0.05 * t, 0.7), 0.8), 20)
    events = run(frames, rules(INTRUSION))
    assert [e.kind for e in events] == ["zone_intrusion"]
    ev = events[0]
    assert (ev.rule, ev.zone, ev.severity, ev.camera) == ("pit-no-entry", "pit", "critical", "yard")
    assert 4.0 <= ev.started <= 4.7 and ev.confirmed == pytest.approx(ev.started + 2.0, abs=0.1)


def test_passing_through_a_corner_of_the_zone_gives_no_event():
    # walks right into the zone at t = 4.5 s and turns up out of it at t = 5.5 s: inside for 1 s
    frames = walk(lambda t: (0.45 if t < 4.5 else 0.55, 0.8 if t < 5.5 else 0.45), 10)
    assert run(frames, rules(INTRUSION)) == []


def test_standing_on_the_edge_does_not_alert():
    # feet flicker in and out every frame: half the votes are "inside", never more than 60%
    frames = [person(0.499 if k % 2 else 0.501, 0.8) for k in range(20 * FPS)]
    assert run(frames, rules(INTRUSION)) == []


def test_feet_out_of_view_are_not_an_intrusion():
    # box reaching the bottom edge of the frame: we can't see where they stand
    frames = [[Det(0, (0.6, 0.7, 0.66, 1.0), 0.9)] for _ in range(10 * FPS)]
    assert [e for e in run(frames, rules(INTRUSION)) if e.kind == "zone_intrusion"] == []


def test_other_cameras_are_not_affected():
    frames = walk(lambda t: (0.7, 0.8), 10)
    rs = rules(INTRUSION)
    assert [e.kind for e in run(frames, rs)] == ["zone_intrusion"]
    assert run(frames, rs, camera="gate") == []
    assert [r.id for r in rs.for_camera("gate").rules] == ["no_helmet", "no_vest"]


def test_a_short_cooldown_lets_the_same_person_alert_again():
    rule = Rule("pit-no-entry", "zone_intrusion", ("yard",), "pit", dwell=2.0, cooldown=10.0)
    # in the zone 0-6 s, out 6-10 s, in again 10-16 s: the second entry is within 10 s of the first alert
    inside = lambda t: (0.7, 0.8) if (t < 6 or 10 <= t < 16 or t >= 24) else (0.3, 0.8)
    events = run(walk(inside, 30), rules(rule))
    assert len(events) == 2 and events[1].started >= 24


# -- active hours --------------------------------------------------------------------------------
def test_active_hours_parsing():
    s = parse_schedule({"hours": "07:00-19:00", "days": "mon-sat"})
    monday = datetime(2026, 9, 21)
    assert s.active(monday.replace(hour=7)) and s.active(monday.replace(hour=18, minute=59))
    assert not s.active(monday.replace(hour=19)) and not s.active(monday.replace(hour=6, minute=59))
    assert not s.active(datetime(2026, 9, 27, 12))                       # Sunday
    night = parse_schedule({"hours": "22:00-06:00", "days": "mon"})
    assert night.active(monday.replace(hour=23)) and night.active(monday.replace(hour=3) + timedelta(days=1))
    assert not night.active(monday.replace(hour=3))                        # early Monday belongs to Sunday night
    split = parse_schedule({"hours": "06:00-12:00, 13:00-18:00"})
    assert split.active(monday.replace(hour=11)) and not split.active(monday.replace(hour=12, minute=30))
    assert parse_schedule({"hours": "00:00-24:00"}).active(monday.replace(hour=5))
    assert parse_schedule("always") is None and parse_schedule(None) is None


def test_no_event_outside_active_hours():
    rule = Rule("pit-day", "zone_intrusion", ("yard",), "pit", dwell=2.0, schedule=parse_schedule({"hours": "07:00-19:00"}))
    frames = walk(lambda t: (0.7, 0.8), 30)
    evening = datetime(2026, 9, 21, 19, 0, 30).astimezone()
    assert run(frames, rules(rule), start=evening) == []
    events = run(frames, rules(rule), start=datetime(2026, 9, 21, 18, 59, 0).astimezone())
    assert len(events) == 1 and events[0].time.startswith("2026-09-21T18:59:0")


def test_someone_already_inside_when_the_rule_switches_on_is_counted_from_then():
    rule = Rule("pit-day", "zone_intrusion", ("yard",), "pit", dwell=2.0, schedule=parse_schedule({"hours": "07:00-19:00"}))
    frames = walk(lambda t: (0.7, 0.8), 20)
    events = run(frames, rules(rule), start=datetime(2026, 9, 21, 6, 59, 50).astimezone())   # on at t = 10 s
    assert len(events) == 1 and 12.0 <= events[0].confirmed <= 13.0


def test_a_rule_switching_off_during_the_dwell_gives_no_event():
    rule = Rule("pit-day", "zone_intrusion", ("yard",), "pit", dwell=2.0, schedule=parse_schedule({"hours": "07:00-19:00"}))
    # enters at 18:59:59 (t = 4 s): the dwell would end after 19:00
    frames = walk(lambda t: (0.7 if t >= 4 else 0.3, 0.8), 15)
    assert run(frames, rules(rule), start=datetime(2026, 9, 21, 18, 59, 55).astimezone()) == []


# -- PPE rules limited to a zone -------------------------------------------------------------------
def test_helmet_required_only_inside_a_zone():
    rs = RuleSet([Rule("pit-helmet", "no_helmet", ("yard",), "pit", "high")], {"yard": CameraZones("yard", [PIT])})
    outside = walk(lambda t: (0.3, 0.8), 15, helmet=False)
    assert run(outside, rs) == []
    goes_in = walk(lambda t: (0.3 if t < 5 else 0.7, 0.8), 15, helmet=False)
    events = run(goes_in, rs)
    assert [(e.kind, e.zone) for e in events] == [("no_helmet", "pit")] and events[0].started >= 5


def test_two_rules_on_one_person_raise_two_events_of_the_same_shape():
    events = run(walk(lambda t: (0.7, 0.8), 10, helmet=False), rules(INTRUSION))
    assert sorted(e.kind for e in events) == ["no_helmet", "zone_intrusion"]
    a, b = (e.to_dict() for e in events)
    assert a.keys() == b.keys()
    for d in (a, b):
        assert d["camera"] == "yard" and d["rule"] and d["severity"] and d["time"] and d["event_id"]
        assert len(d["box"]) == 4 and d["frame_index"] > 0 and d["track_id"] == 1


def test_a_new_track_id_inside_the_zone_joins_the_same_incident():
    engine = EventEngine("yard", rules=[INTRUSION])
    box = (0.67, 0.5, 0.73, 0.8)
    events = []
    for k in range(20 * FPS):
        t = k / FPS
        if 6 <= t < 6.5:
            people = []                      # hidden for half a second ...
        else:
            people = [(1 if t < 6 else 2, box, {"pit-no-entry": True})]   # ... and back under a new ID
        events += engine.update(t, k, people)
    assert len(events) == 1


def test_camera_rules_votes():
    cr = CameraRules("yard", [INTRUSION, Rule("no_helmet", "no_helmet")], [PIT])
    from ppe_monitor.rules.ppe import MISSING, WORN, Verdict
    votes, feet, inside = cr.votes(Verdict(WORN, WORN), (0.67, 0.5, 0.73, 0.8))
    assert votes == {"pit-no-entry": True, "no_helmet": False} and inside == ["pit"]
    votes, _, inside = cr.votes(Verdict(MISSING, WORN), (0.2, 0.5, 0.26, 0.8))
    assert votes == {"pit-no-entry": False, "no_helmet": True} and inside == []
    votes, feet, _ = cr.votes(Verdict(WORN, WORN), (0.67, 0.7, 0.73, 1.0))
    assert feet is None and votes["pit-no-entry"] is None


# -- the rules file ------------------------------------------------------------------------------
ZONES = "cameras:\n  yard:\n    zones:\n      - {id: pit, polygon: [[0.5, 0.5], [1, 0.5], [1, 0.95], [0.5, 0.95]]}\n"


def load(tmp_path, text, zones=ZONES):
    (tmp_path / "rules.yaml").write_text(text)
    (tmp_path / "zones.yaml").write_text(zones)
    return RuleSet.load(tmp_path / "rules.yaml", tmp_path / "zones.yaml")


def test_a_good_rules_file(tmp_path):
    rs = load(tmp_path, """timezone: Asia/Kolkata
rules:
  - {id: no_helmet, type: no_helmet, cameras: all, severity: high}
  - {id: pit, type: zone_intrusion, cameras: [yard], zone: pit, severity: critical, dwell: 2,
     active: {hours: "07:00-19:00", days: mon-sat}}
  - {id: old, type: no_vest, enabled: false}
""")
    assert [r.id for r in rs.rules] == ["no_helmet", "pit"]
    cr = rs.for_camera("yard")
    assert cr.zone("pit") is not None and str(cr.tz) == "Asia/Kolkata"
    # 12:00 Monday in Kolkata is active; 20:00 is not
    assert cr.active(datetime(2026, 9, 21, 6, 30).astimezone(cr.tz).replace(hour=12)) == {"no_helmet", "pit"}
    assert cr.active(datetime(2026, 9, 21, 20, 0, tzinfo=cr.tz)) == {"no_helmet"}


@pytest.mark.parametrize("rule, message", [
    ("{id: a, type: no_gloves}", "type must be one of"),
    ("{id: a, type: no_helmet, colour: red}", "unknown keys"),
    ("{id: a, type: zone_intrusion, cameras: [yard]}", "needs a zone"),
    ("{id: a, type: zone_intrusion, cameras: all, zone: pit}", "must list its cameras"),
    ("{id: a, type: zone_intrusion, cameras: [gate], zone: pit}", "camera gate has no zone 'pit'"),
    ("{id: a, type: no_helmet, severity: urgent}", "severity must be"),
    ("{id: a, type: no_helmet, dwell: -1}", "can't be negative"),
    ("{id: a, type: no_helmet, active: {hours: '7 to 19'}}", "range like"),
    ("{id: a, type: no_helmet, active: {hours: '07:00-19:00', days: someday}}", "unknown day"),
    ("{id: 'No Helmet', type: no_helmet}", "lowercase"),
])
def test_bad_rules_are_refused_with_a_clear_message(tmp_path, rule, message):
    with pytest.raises(ConfigError, match=message):
        load(tmp_path, f"rules:\n  - {rule}\n")


def test_duplicate_ids_and_bad_timezone(tmp_path):
    with pytest.raises(ConfigError, match="twice"):
        load(tmp_path, "rules:\n  - {id: a, type: no_helmet}\n  - {id: a, type: no_vest}\n")
    with pytest.raises(ConfigError, match="timezone"):
        load(tmp_path, "timezone: Mars/Olympus\nrules: []\n")


def test_without_a_rules_file_every_camera_gets_the_phase2_rules(tmp_path):
    rs = RuleSet.load(tmp_path / "none.yaml", tmp_path / "none.yaml")
    assert [r.type for r in rs.for_camera("anything").rules] == ["no_helmet", "no_vest"]


def test_the_projects_own_config_files_load():
    rs = RuleSet.load()
    assert {r.type for r in rs.rules} >= {"no_helmet", "zone_intrusion"}
    assert "no_vest" not in {r.type for r in rs.rules}          # hi-vis isn't required at this site
    for cam in ("cam1", "cam2", "cam3"):
        assert any(r.type == "zone_intrusion" for r in rs.for_camera(cam).rules)
