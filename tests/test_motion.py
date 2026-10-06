"""The whole chain (tracker -> PPE rule -> events) with people who walk, get hidden, bend and crouch.

Detections and keypoints are made by hand, so these tests pin down the logic independently of the
detector. A 16:9 camera; a person is 0.3 of the frame tall (108 px at the detector's input).
"""

import numpy as np
import pytest

from ppe_monitor.pipeline import REUSE_POSES, PPEMonitor, Settings
from ppe_monitor.vision.matching import Det
from ppe_monitor.vision.pose import Pose, carried

FRAME = np.zeros((720, 1280, 3), np.uint8)
W_PER_H = 720 / 1280          # one frame-height in units of frame-width
H, WID = 0.30, 0.30 * 0.35 * W_PER_H   # person box height, width (0-1 of the frame)


class FakeModel:
    names = {0: "person", 1: "helmet", 2: "vest"}


def worker(x, y=0.40, helmet=True, vest=True, score=0.9):
    """Detections for an upright worker whose box starts at (x, y)."""
    dets = [Det(0, (x, y, x + WID, y + H), score)]
    if helmet:
        dets.append(Det(1, (x + 0.2 * WID, y - 0.01, x + 0.8 * WID, y + 0.05), 0.8))
    if vest:
        dets.append(Det(2, (x + 0.05 * WID, y + 0.07, x + 0.95 * WID, y + 0.17), 0.8))
    return dets


def run(frames, fps, poses=None, settings=None):
    """frames: list of detection lists. Returns (events, track ids seen)."""
    m = PPEMonitor(FakeModel(), "cam", fps, settings or Settings.load(), pose_model=False)
    events, ids = [], set()
    for k, dets in enumerate(frames):
        r = m.process(FRAME, k / fps, detections=dets, poses=poses[k] if poses else None)
        events += r.events
        ids |= {t.track_id for t in r.tracks}
    return events, ids


def walking(fps, seconds=8, helmet=True, hidden=None, speed=0.8):
    """A worker walking left to right at `speed` person-heights per second."""
    step = speed * H * W_PER_H / fps
    frames = []
    for k in range(int(seconds * fps)):
        t = k / fps
        frames.append([] if hidden and hidden[0] <= t < hidden[1] else worker(0.05 + step * k, helmet=helmet))
    return frames


@pytest.mark.parametrize("fps", [15, 3])
def test_a_walking_worker_without_a_helmet_gives_one_event(fps):
    events, ids = run(walking(fps, helmet=False), fps)
    assert [e.kind for e in events] == ["no_helmet"] and len(ids) == 1


@pytest.mark.parametrize("fps", [15, 3])
def test_a_walking_worker_with_a_helmet_gives_none(fps):
    assert run(walking(fps), fps)[0] == []


@pytest.mark.parametrize("fps", [15, 3])
def test_walking_behind_a_pillar_keeps_the_id_and_the_single_event(fps):
    events, ids = run(walking(fps, seconds=12, helmet=False, hidden=(5.0, 6.0)), fps)
    assert [e.kind for e in events] == ["no_helmet"]
    assert len(ids) == 1


def test_two_walkers_crossing_one_without_helmet():
    fps, frames = 15, []
    step = 0.8 * H * W_PER_H / fps
    for k in range(10 * fps):
        a = worker(0.10 + step * k, helmet=False)          # walks right
        b = worker(0.75 - step * k, y=0.42)                # walks left, helmet on
        frames.append(a + b)
    events, _ = run(frames, fps)
    assert [e.kind for e in events] == ["no_helmet"]
    ev = events[0]
    walker_a_x = 0.10 + step * ev.frame_index
    assert abs(ev.box[0] - walker_a_x) < 0.02            # the event is on the walker without a helmet


def bent_over(x, y, helmet_found: bool, head_in_view: bool = True):
    """A worker bent right over: a wide, short box, head low at the left, helmet (if found) on it.
    head_in_view=False: seen from behind, no nose / eye / ear keypoint."""
    box = (x - 0.02, y + 0.12, x + WID + 0.04, y + H)
    dets = [Det(0, box, 0.85), Det(2, (x, y + 0.14, x + WID, y + 0.22), 0.7)]
    head = (x - 0.01, y + 0.24)
    if helmet_found:
        dets.append(Det(1, (head[0] - 0.012, head[1] - 0.03, head[0] + 0.012, head[1] + 0.01), 0.6))
    xy, conf = np.zeros((17, 2)), np.zeros(17)
    points = {0: head, 3: (head[0] + 0.005, head[1] - 0.01), 5: (x + 0.005, y + 0.19), 6: (x + 0.01, y + 0.2),
              11: (x + WID, y + 0.14), 12: (x + WID + 0.005, y + 0.145), 15: (x + WID, y + H - 0.01),
              16: (x + WID + 0.01, y + H - 0.01)}
    if not head_in_view:
        del points[0], points[3]
    for i, p in points.items():
        xy[i], conf[i] = p, 0.9
    return dets, Pose(box, xy, conf, 1280 / 720)


def standing_pose(x, y):
    xy, conf = np.zeros((17, 2)), np.zeros(17)
    cx = x + WID / 2
    for i, p in {0: (cx, y + 0.03), 5: (cx - 0.008, y + 0.07), 6: (cx + 0.008, y + 0.07), 11: (cx - 0.006, y + 0.16),
                 12: (cx + 0.006, y + 0.16), 15: (cx - 0.006, y + 0.29), 16: (cx + 0.006, y + 0.29)}.items():
        xy[i], conf[i] = p, 0.9
    return Pose((x, y, x + WID, y + H), xy, conf, 1280 / 720)


def bending_scene(helmet_found_while_bent: bool, head_in_view: bool = True, helmet: bool = True, fps=15):
    """Standing 3 s, bent over 6 s (picking something up), standing again 3 s."""
    frames, poses = [], []
    x, y = 0.4, 0.4
    for k in range(12 * fps):
        t = k / fps
        if 3 <= t < 9:
            dets, pose = bent_over(x, y, helmet and helmet_found_while_bent, head_in_view)
        else:
            dets, pose = worker(x, y, helmet=helmet), standing_pose(x, y)
        frames.append(dets)
        poses.append([pose])
    return frames, poses


def test_bending_over_with_the_helmet_on_gives_no_event():
    frames, poses = bending_scene(helmet_found_while_bent=True)
    assert run(frames, 15, poses)[0] == []                     # keypoints find the helmet on the lowered head
    assert [e.kind for e in run(frames, 15)[0]] == ["no_helmet"]  # the upright-only box rule would alert


def test_bending_away_from_the_camera_gives_no_event():
    # seen from behind, the head is out of sight and the helmet isn't found: "unknown", not "missing"
    frames, poses = bending_scene(helmet_found_while_bent=False, head_in_view=False)
    assert run(frames, 15, poses)[0] == []


def test_a_worker_without_a_helmet_who_stays_bent_over_is_caught():
    # bent over the whole time (tying rebar), head in view: caught without ever standing up
    fps, frames, poses = 15, [], []
    for k in range(8 * fps):
        dets, pose = bent_over(0.4, 0.4, helmet_found=False)
        frames.append(dets)
        poses.append([pose])
    assert [e.kind for e in run(frames, fps, poses)[0]] == ["no_helmet"]


def test_a_worker_without_a_helmet_bent_away_is_caught_once_upright():
    frames, poses = bending_scene(helmet_found_while_bent=False, head_in_view=False, helmet=False)
    assert [e.kind for e in run(frames, 15, poses)[0]] == ["no_helmet"]


def test_crouching_keeps_the_track_id():
    fps, frames = 15, []
    for k in range(9 * fps):
        t = k / fps
        if 3 <= t < 6:   # crouched: box half as tall, same feet position
            x, y = 0.4, 0.4 + H / 2
            frames.append([Det(0, (x - 0.005, y, x + WID + 0.005, 0.4 + H), 0.8),
                           Det(1, (x + 0.2 * WID, y - 0.01, x + 0.8 * WID, y + 0.05), 0.8),
                           Det(2, (x, y + 0.04, x + WID, y + 0.11), 0.8)])
        else:
            frames.append(worker(0.4))
    events, ids = run(frames, fps)
    assert events == [] and len(ids) == 1


@pytest.mark.parametrize("fps", [15, 5])
def test_a_group_walking_side_by_side_gives_one_event(fps):
    # three workers shoulder to shoulder, walking together; the middle one has no helmet
    step = 0.8 * H * W_PER_H / fps
    frames = []
    for k in range(10 * fps):
        x = 0.10 + step * k
        frames.append(worker(x) + worker(x + WID, helmet=False) + worker(x + 2 * WID))
    events, ids = run(frames, fps)
    assert [e.kind for e in events] == ["no_helmet"] and len(ids) == 3


# -- keypoints on every second frame (pose.every: 2): the frames between carry each person's last ones --

def run_every(frames, fps, poses, every):
    """Like run(), but the keypoints are only handed over on the monitor's turns, as the multi-camera
    service does; on the other frames it carries each person's last keypoints. Returns (events, turns)."""
    s = Settings.load()
    s.pose["every"] = every
    m = PPEMonitor(FakeModel(), "cam", fps, s, pose_model=False)
    events, turns = [], 0
    for k, dets in enumerate(frames):
        people = any(d.cls == 0 for d in dets)
        if people and not m.pose_due():
            p = REUSE_POSES
        else:
            p = poses[k]
            turns += bool(people)
        r = m.process(FRAME, k / fps, detections=dets, poses=p)
        events += r.events
    return events, turns


@pytest.mark.parametrize("every", [1, 2, 3])
def test_bending_and_crouching_checks_hold_with_keypoints_on_fewer_frames(every):
    frames, poses = bending_scene(helmet_found_while_bent=True)
    events, turns = run_every(frames, 15, poses, every)
    assert events == [] and turns == -(-len(frames) // every)       # the keypoint model ran 1 frame in `every`
    frames, poses = bending_scene(helmet_found_while_bent=False, head_in_view=False)
    assert run_every(frames, 15, poses, every)[0] == []
    frames, poses = bending_scene(helmet_found_while_bent=False, head_in_view=False, helmet=False)
    assert [e.kind for e in run_every(frames, 15, poses, every)[0]] == ["no_helmet"]
    frames, poses = [], []
    for _ in range(8 * 15):
        dets, pose = bent_over(0.4, 0.4, helmet_found=False)
        frames.append(dets)
        poses.append([pose])
    assert [e.kind for e in run_every(frames, 15, poses, every)[0]] == ["no_helmet"]


def test_carried_keypoints_move_with_a_walking_person():
    fps, speed = 10, 1.5                     # brisk walking, 10 frames a second
    step = speed * H * W_PER_H / fps
    s = Settings.load()
    s.pose["every"] = 2
    m = PPEMonitor(FakeModel(), "cam", fps, s, pose_model=False)
    worst = 0.0
    for k in range(40):
        x = 0.05 + step * k
        truth = standing_pose(x, 0.40)
        p = [truth] if m.pose_due() else REUSE_POSES
        r = m.process(FRAME, k / fps, detections=worker(x), poses=p)
        (tid,) = r.poses
        assert r.verdicts[tid].helmet == "worn" and r.verdicts[tid].posture == "upright"
        worst = max(worst, float(np.linalg.norm(r.poses[tid].head - truth.head)))
    assert worst < 0.1 * H                   # the carried head stays on the real one (not 0.1 s behind it)


def test_keypoints_are_carried_onto_the_new_box():
    xy = np.tile([0.15, 0.25], (17, 1))
    p = Pose((0.1, 0.2, 0.2, 0.6), xy, np.full(17, 0.9))
    q = carried(p, (0.1, 0.2, 0.2, 0.6), (0.3, 0.2, 0.5, 0.6))     # moved right, and twice as wide
    assert np.allclose(q.xy[0], (0.4, 0.25)) and q.box == pytest.approx((0.3, 0.2, 0.5, 0.6))


def test_pose_every_must_be_a_whole_number(tmp_path):
    cfg = tmp_path / "ppe.yaml"
    cfg.write_text("pose:\n  every: 0\n")
    with pytest.raises(ValueError, match="every"):
        Settings.load(cfg)
