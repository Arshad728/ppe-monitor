"""Test clips made from photos, and the scoring of events against them."""

from pathlib import Path

import cv2
import numpy as np

from ppe_monitor.data.compliance import PersonTruth
from ppe_monitor.rules.check import Case
from ppe_monitor.rules.events import Event
from ppe_monitor.rules.scoring import score_clip
from ppe_monitor.sim.photo_clips import make_clip


def _case(tmp_path: Path) -> Case:
    img = np.full((480, 640, 3), 120, np.uint8)
    cv2.rectangle(img, (100, 100), (180, 400), (40, 40, 200), -1)   # "person" 0
    cv2.rectangle(img, (400, 120), (470, 400), (40, 200, 40), -1)   # "person" 1
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), img)
    truths = [PersonTruth((100 / 640, 100 / 480, 180 / 640, 400 / 480), helmet=False, vest=True),
              PersonTruth((400 / 640, 120 / 480, 470 / 640, 400 / 480), helmet=True, vest=None)]
    return Case(path, 480.0, truths, {})


def test_clip_follows_the_people_and_marks_the_occluder(tmp_path):
    truth = make_clip(_case(tmp_path), tmp_path / "clip.mp4", seconds=10, fps=15)
    assert truth["frames"] == 150 and len(truth["people"]) == 2
    cap = cv2.VideoCapture(str(tmp_path / "clip.mp4"))
    frames = [cap.read()[1] for _ in range(150)]
    assert all(f is not None for f in frames)
    h, w = frames[0].shape[:2]
    for k in (0, 149):  # the red block stays inside person 0's box as the view pans and zooms
        b = truth["people"][0]["boxes"][k]
        cx, cy = int((b[0] + b[2]) / 2 * w), int((b[1] + b[3]) / 2 * h)
        assert frames[k][cy, cx, 2] > 150 and frames[k][cy, cx, 1] < 100
    hidden = [k for k, b in enumerate(truth["people"][0]["boxes"]) if b[4]]
    assert hidden and all(3.0 <= k / 15 < 7.0 for k in hidden)   # only while the bar sweeps past


def _event(kind, box, k, track=1):
    return Event(kind, "clip", track, k / 15 - 3, k / 15, box, 1.0, k)


def test_scoring_counts_caught_duplicate_false_and_unscored(tmp_path):
    truth = make_clip(_case(tmp_path), tmp_path / "clip.mp4", seconds=10, fps=15)
    p0 = truth["people"][0]["boxes"][60][:4]
    p1 = truth["people"][1]["boxes"][60][:4]
    tracks = [{1: p[0]["boxes"][k][:4], 2: p[1]["boxes"][k][:4]} for k, p in ((k, truth["people"]) for k in range(150))]

    one = score_clip(truth, [_event("no_helmet", p0, 60)], tracks)
    assert one.caught_once["helmet"] == 1 and not one.details and one.id_switches == 0

    twice = score_clip(truth, [_event("no_helmet", p0, 60), _event("no_helmet", p0, 120, track=3)], tracks)
    assert twice.caught_more["helmet"] == 1

    wrong = score_clip(truth, [_event("no_helmet", p1, 60, track=2), _event("no_vest", p1, 60, track=2)], tracks)
    assert wrong.missed["helmet"] == 1          # person 0 got nothing
    assert wrong.false_events["helmet"] == 1    # person 1 wears a helmet
    assert wrong.unscored_events == 1           # person 1's vest status isn't known


def test_zone_clip_moves_people_into_the_zone_and_scores_intrusions(tmp_path):
    from ppe_monitor.rules.scoring import score_zone, zone_truth
    case = _case(tmp_path)
    case.gt_boxes["person"] = [t.box for t in case.truths] + [(250 / 640, 150 / 480, 300 / 640, 380 / 480)]
    truth = make_clip(case, tmp_path / "zone.mp4", seconds=10, fps=15, motion="pan")
    assert "zone" in truth and len(truth["people"]) == 3              # an unlabelled person is added
    assert truth["people"][2]["helmet"] is None
    b0, b1 = truth["people"][1]["boxes"][0], truth["people"][1]["boxes"][-1]
    assert b1[0] > b0[0] + 0.2                                        # the photo slides right
    inside = [zone_truth(p, truth["zone"]["polygon"], 150) for p in truth["people"]]
    assert not any(inside[0])                                         # the left person never gets there
    entry = inside[1].index(True)                                     # the right one walks in and stays
    assert all(inside[1][entry:]) and entry / 15 < 10 - 3.5

    tracks = {k: {1: p[0]["boxes"][k][:4], 2: p[1]["boxes"][k][:4]} for k, p in ((k, truth["people"]) for k in range(150))}
    k = min(149, entry + 40)
    ev = Event("zone_intrusion", "clip", 2, entry / 15, k / 15, truth["people"][1]["boxes"][k][:4], 1.0, k)
    sc = score_zone(truth, [ev], tracks, dwell=2.0)
    assert (sc.should_alert, sc.caught_once, sc.never_inside, sc.false_events) == (1, 1, 2, 0)   # 2: the middle one too
    assert abs(sc.delays[0] - (k - entry) / 15) < 1e-9
    wrong = Event("zone_intrusion", "clip", 1, 1.0, 3.0, truth["people"][0]["boxes"][45][:4], 1.0, 45)
    sc = score_zone(truth, [wrong], tracks, dwell=2.0)
    assert sc.missed == 1 and sc.false_events == 1
    assert score_clip(truth, [ev], tracks).unscored_events == 0      # zone events are left to score_zone
