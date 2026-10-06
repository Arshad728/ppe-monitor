"""Person tracking with hand-made detections: two people walking, occlusions, a crossing."""

from ppe_monitor.vision.matching import Det
from ppe_monitor.vision.tracking import PersonTracker, TrackerSettings

FPS, SIZE = 15, (1280, 720)


def person(x, y=0.3, score=0.9, w=0.06, h=0.35):
    return Det(0, (x, y, x + w, y + h), score)


def run(frames, settings=TrackerSettings()):
    """frames: list of lists of Dets -> list of {track_id: x1} per frame."""
    tracker = PersonTracker(FPS, settings)
    return [{t.track_id: round(t.box[0], 3) for t in tracker.update(dets, SIZE)} for dets in frames]


def test_two_walking_people_keep_their_ids():
    frames = [[person(0.10 + 0.004 * k), person(0.70 - 0.004 * k)] for k in range(60)]
    out = run(frames)
    ids = [sorted(f) for f in out]
    assert all(i == ids[0] for i in ids) and len(ids[0]) == 2
    left = min(out[0], key=out[0].get)
    assert all(f[left] < 0.5 for f in out)  # the left walker's ID never jumps to the right walker


def test_short_occlusion_keeps_the_id():
    # detected for 20 frames, hidden for 15 frames (1 s, less than lost_seconds), then back
    frames = [[person(0.10 + 0.004 * k)] if not 20 <= k < 35 else [] for k in range(60)]
    out = run(frames)
    seen = {tid for f in out for tid in f}
    assert len(seen) == 1
    assert out[25] == {}  # nothing reported while hidden


def test_low_confidence_detections_keep_a_track_alive_but_never_start_one():
    walking = [[person(0.10 + 0.004 * k, score=0.9 if k < 10 else 0.2)] for k in range(40)]
    assert {tid for f in run(walking) for tid in f} == {1} and all(run(walking))
    ghost = [[person(0.5, score=0.2)] for _ in range(20)]  # never above `high`
    assert all(f == {} for f in run(ghost))


def test_a_single_frame_false_person_does_not_become_a_track():
    frames = [[person(0.1)] for _ in range(10)]
    frames[5] = frames[5] + [person(0.7, score=0.95)]  # a one-frame false detection
    out = run(frames)
    assert {tid for f in out for tid in f} == {1}


def test_people_crossing_keep_their_ids():
    # two people walk towards each other, overlap for a few frames, and carry on
    frames = [[person(0.20 + 0.01 * k), person(0.60 - 0.01 * k)] for k in range(40)]
    out = run(frames)
    first = {tid: x < 0.4 for tid, x in out[0].items()}         # which ID started on the left
    last = {tid: x > 0.4 for tid, x in out[-1].items()}          # ... ended on the right
    assert set(first) == set(last) and first == last


def test_one_person_detected_twice_is_tracked_once():
    # a full-body box and a shorter upper-body box on the same person, every frame
    frames = [[person(0.10 + 0.004 * k), Det(0, (0.10 + 0.004 * k, 0.3, 0.16 + 0.004 * k, 0.5), 0.6)] for k in range(30)]
    assert {tid for f in run(frames) for tid in f} == {1}


def test_two_people_one_behind_the_other_are_both_tracked():
    # the far person is smaller and mostly inside the near person's box, but their heads are apart
    frames = [[person(0.40, y=0.30, h=0.50), Det(0, (0.42, 0.22, 0.47, 0.50), 0.8)] for _ in range(30)]
    assert len({tid for f in run(frames) for tid in f}) == 2
