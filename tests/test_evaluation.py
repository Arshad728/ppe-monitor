"""Phase 6: ground truth, matching alerts to it, the headline numbers, and the annotation tool."""

import json
import math

import numpy as np
import pytest

from ppe_monitor.evaluation.annotator import Annotator, new_truth
from ppe_monitor.evaluation.score import AlertRecord, Summary, match_clip, poisson_upper, wilson
from ppe_monitor.evaluation.truth import TRUTH_DIR, ClipTruth, Episode, from_photo_clips, load_dir, load_yaml, save_yaml


def truth(episodes, complete=True, checked=("no_helmet",)):
    return ClipTruth("c.mp4", None, 15.0, 10.0, episodes, checked, complete=complete)


def alert(kind="no_helmet", confirmed=3.5, box=(0.1, 0.1, 0.3, 0.9), delivered=4.2, eid="e"):
    return AlertRecord(eid, kind, confirmed - 3.0, confirmed, delivered, box)


BOX_A = (0.1, 0.1, 0.3, 0.9)
BOX_B = (0.6, 0.1, 0.8, 0.9)


# -- episodes and truth files ---------------------------------------------------------------------
def test_box_follows_a_straight_line_between_known_boxes():
    e = Episode("no_helmet", 0, 10, boxes=[(0.0, (0.0, 0.0, 0.2, 0.2)), (2.0, (0.2, 0.2, 0.4, 0.4))])
    assert e.box_at(1.0) == pytest.approx((0.1, 0.1, 0.3, 0.3))
    assert e.box_at(-1) == (0.0, 0.0, 0.2, 0.2) and e.box_at(9) == (0.2, 0.2, 0.4, 0.4)
    assert Episode("no_helmet", 0, 1).box_at(0.5) is None


def test_yaml_round_trip_and_validation(tmp_path):
    t = ClipTruth("x.mp4", tmp_path / "x.mp4", 15.0, 12.0,
                  [Episode("no_helmet", 1.0, 6.0, True, [(1.0, BOX_A), (5.0, BOX_B)], "man at the gate"),
                   Episode("no_helmet", 7.0, 8.0, False)], ("no_helmet",), "by hand", private=True)
    save_yaml(t, tmp_path / "x.yaml", "# header")
    back = load_yaml(tmp_path / "x.yaml", tmp_path)
    assert back.clip == "x.mp4" and back.private and back.complete and back.checked == ("no_helmet",)
    assert [(e.start, e.end, e.sustained, len(e.boxes)) for e in back.episodes] == [(1.0, 6.0, True, 2), (7.0, 8.0, False, 0)]
    assert back.episodes[0].box_at(3.0) == pytest.approx((0.35, 0.1, 0.55, 0.9))
    bad = tmp_path / "bad.yaml"
    bad.write_text("clip: x.mp4\nfps: 15\nduration: 3\nchecked: [no_helmet]\nepisodes:\n  - {kind: no_vest, start: 0, end: 1}\n")
    with pytest.raises(ValueError, match="not in 'checked'"):
        load_yaml(bad)
    bad.write_text("clip: x.mp4\nfps: 15\nduration: 3\nchecked: [no_helmet]\nepisodes:\n"
                   "  - {kind: no_helmet, start: 0, end: 1, boxes: {0.5: [0.5, 0.1, 0.2, 0.9]}}\n")
    with pytest.raises(ValueError, match="x1 < x2"):
        load_yaml(bad)


def test_the_site_clips_truth_files_load():
    if not TRUTH_DIR.is_dir():
        pytest.skip("no data/ground_truth")
    ts = load_dir()
    assert ts and all(t.checked and t.fps > 0 and t.duration > 0 for t in ts)
    for t in ts:
        for e in t.episodes:
            assert 0 <= e.start < e.end <= t.duration + 0.01


def test_truth_from_generated_clips(tmp_path):
    box = [0.2, 0.1, 0.4, 0.9, False]
    edge = [0.0, 0.1, 0.2, 0.9, False]              # cut off by the frame's left edge
    clips = [{"clip": "a.mp4", "fps": 10.0, "frames": 100, "people": [
        {"helmet": False, "vest": True, "boxes": [box] * 100},
        {"helmet": True, "vest": None, "boxes": [box] * 100},
        {"helmet": False, "vest": False, "boxes": [edge] * 100}]},
        {"clip": "z.mp4", "fps": 10.0, "frames": 100, "zone": {"id": "zone", "name": "Z", "polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]},
         "people": [{"helmet": None, "vest": None, "boxes": [[0.1, 0.1, 0.3, 0.9, False]] * 20 + [[0.6, 0.1, 0.8, 0.9, False]] * 80},
                    {"helmet": None, "vest": None, "boxes": [[0.1, 0.1, 0.3, 0.9, False]] * 100}]}]
    (tmp_path / "clips.json").write_text(json.dumps({"clips": clips}))
    a, z = from_photo_clips(tmp_path)
    assert not a.complete and a.checked == ("no_helmet", "no_vest")
    kinds = [(e.kind, e.sustained, e.compliant) for e in a.episodes]
    assert kinds == [("no_helmet", True, False), ("no_vest", False, True),      # person 0: no helmet; vest on
                     ("no_helmet", False, True), ("no_vest", False, False),     # person 1: helmet on; vest unknown
                     ("no_helmet", False, False), ("no_vest", False, False)]    # person 2: at the edge: can't expect
    assert z.checked == ("zone_intrusion",) and len(z.sustained) == 1
    assert z.sustained[0].start == pytest.approx(2.0)
    assert [e.compliant for e in z.episodes] == [False, True]                    # person 1 never enters


# -- matching ---------------------------------------------------------------------------------------
def test_one_alert_per_violation_is_caught_the_second_is_a_duplicate():
    t = truth([Episode("no_helmet", 0.0, 10.0, True, [(0.0, BOX_A)])])
    r = match_clip(t, "cam", [alert(eid="1"), alert(confirmed=8.0, eid="2")])
    assert [o.verdict for o in r.alerts] == ["true", "duplicate"] and r.missed == []


def test_alerts_on_nobody_are_false_only_when_the_truth_is_complete():
    ep = Episode("no_helmet", 0.0, 10.0, True, [(0.0, BOX_A)])
    elsewhere = alert(box=BOX_B)
    assert match_clip(truth([ep]), "c", [elsewhere]).alerts[0].verdict == "false"
    assert match_clip(truth([ep], complete=False), "c", [elsewhere]).alerts[0].verdict == "unscored"
    compliant = Episode("no_helmet", 0.0, 10.0, False, [(0.0, BOX_B)], compliant=True)
    r = match_clip(truth([ep, compliant], complete=False), "c", [elsewhere])
    assert r.alerts[0].verdict == "false" and r.missed == [0]


def test_when_and_where_must_both_fit():
    ep = Episode("no_helmet", 5.0, 8.0, True, [(5.0, BOX_A)])
    assert match_clip(truth([ep]), "c", [alert(confirmed=2.0)]).alerts[0].verdict == "false"      # long before it
    assert match_clip(truth([ep]), "c", [alert(confirmed=11.5)]).alerts[0].verdict == "true"      # up to 4 s after
    assert match_clip(truth([ep]), "c", [alert(confirmed=13.0)]).alerts[0].verdict == "false"
    inside = alert(confirmed=6.0, box=(0.12, 0.3, 0.25, 0.6))                                     # small, centre inside
    assert match_clip(truth([ep]), "c", [inside]).alerts[0].verdict == "true"
    no_box = Episode("no_helmet", 5.0, 8.0, True)
    assert match_clip(truth([no_box]), "c", [alert(confirmed=6.0, box=BOX_B)]).alerts[0].verdict == "true"


def test_not_sustained_and_other_kinds_are_not_scored():
    t = truth([Episode("no_helmet", 0.0, 2.0, False, [(0.0, BOX_A)])])
    r = match_clip(t, "c", [alert(confirmed=2.5), alert(kind="no_vest", eid="v")])
    assert [o.verdict for o in r.alerts] == ["unscored", "unscored"] and r.missed == []


def test_the_best_fitting_person_wins_and_a_violation_breaks_a_tie():
    viol = Episode("no_helmet", 0.0, 10.0, True, [(0.0, (0.1, 0.1, 0.3, 0.9))])
    near = Episode("no_helmet", 0.0, 10.0, False, [(0.0, (0.25, 0.1, 0.45, 0.9))], compliant=True)
    on_near = alert(box=(0.25, 0.1, 0.45, 0.9))
    assert match_clip(truth([viol, near]), "c", [on_near]).alerts[0].verdict == "false"
    same = Episode("no_helmet", 0.0, 10.0, False, [(0.0, (0.1, 0.1, 0.3, 0.9))], compliant=True)
    assert match_clip(truth([same, viol]), "c", [alert()]).alerts[0].verdict == "true"


def test_summary_counts_latency_and_rates():
    s = Summary()
    t = truth([Episode("no_helmet", 1.0, 10.0, True, [(0.0, BOX_A)]), Episode("no_helmet", 0.0, 10.0, True, [(0.0, BOX_B)])])
    s.add(match_clip(t, "c", [alert(confirmed=4.5, delivered=5.0), alert(box=(0.4, 0.0, 0.5, 0.1), eid="f")]))
    d = s.as_dict()
    assert (d["sustained"], d["caught"], d["true"], d["false"]) == (2, 1, 1, 1)
    assert d["recall"] == 0.5 and d["precision"] == 0.5
    assert d["latency"]["median"] == pytest.approx(4.0) and d["to_event"]["median"] == pytest.approx(3.5)
    assert d["false_per_hour"] == pytest.approx(360.0)            # 1 in 10 s
    assert d["false_per_hour_ci"][0] < 360 < d["false_per_hour_ci"][1]


def test_intervals():
    assert wilson(0, 0)[0] != wilson(0, 0)[0]                      # nan
    lo, hi = wilson(40, 43)
    assert 0.80 < lo < 0.82 and 0.97 < hi < 0.98
    assert poisson_upper(0) == pytest.approx(3.689, abs=0.01)     # 0 seen: up to 3.7 is plausible
    assert poisson_upper(4, 0.975) == pytest.approx(1.623, abs=0.01)   # lower end for 5 seen
    assert math.isfinite(poisson_upper(50))


# -- the annotation tool -------------------------------------------------------------------------------
def test_annotator_marks_an_episode_with_boxes_and_decides_sustained(tmp_path):
    t = new_truth(tmp_path / "clip.mp4", 10.0, 100)
    a = Annotator(t, 100)
    a.seek(10)
    a.begin("no_helmet")
    assert a.set_box(0.2, 0.1, 0.1, 0.9)                           # dragged right to left: sorted
    a.seek(30)
    a.set_box(0.3, 0.1, 0.5, 0.9)
    a.seek(60)
    assert a.end()
    e = t.episodes[0]
    assert (e.start, e.end, e.sustained) == (1.0, 6.0, True)       # 5 s >= 3 s dwell + 0.5
    assert e.boxes[0][1] == (0.1, 0.1, 0.2, 0.9) and len(e.boxes) == 2
    a.seek(70)
    a.begin("zone_intrusion")
    a.seek(90)
    a.end()
    assert t.episodes[1].sustained is False and "zone_intrusion" in t.checked   # 2 s < 2 s + 0.5
    a.toggle_sustained()
    assert t.episodes[1].sustained
    a.seek(95)
    a.begin("no_vest")
    a.delete()
    assert len(t.episodes) == 2 and a.dirty
    img = a.render(np.zeros((90, 160, 3), np.uint8))
    assert img.shape[1] == 160 and img.shape[0] > 90
    save_yaml(t, tmp_path / "clip.yaml")
    assert len(load_yaml(tmp_path / "clip.yaml", tmp_path).episodes) == 2
