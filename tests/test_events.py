"""The event state machine driven by hand-written tracks (book, "Prove it" for Week 2)."""

import random

import pytest

from ppe_monitor.rules.events import EventEngine, EventSettings
from ppe_monitor.rules.ppe import MISSING, UNKNOWN, WORN, Verdict

FPS = 15
BOX = (0.40, 0.30, 0.50, 0.70)


def run(timeline, settings=EventSettings(), fps=FPS, box=BOX):
    """timeline: function(t) -> list of (track_id, helmet answer) for that frame, or a list of
    (start, end, track_id, answer) segments. Returns the events raised."""
    engine = EventEngine("cam1", settings)
    duration = max(end for _, end, _, _ in timeline) if isinstance(timeline, list) else 60
    for k in range(int(duration * fps)):
        t = k / fps
        if isinstance(timeline, list):
            frame = [(tid, a) for start, end, tid, a in timeline if start <= t < end]
        else:
            frame = timeline(t)
        engine.update(t, k, [(tid, box, Verdict(helmet=a, vest=WORN)) for tid, a in frame])
    return engine.events


def test_a_sustained_violation_gives_exactly_one_event():
    events = run([(0, 5, 1, WORN), (5, 30, 1, MISSING)])
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "no_helmet" and ev.track_id == 1 and ev.camera == "cam1"
    assert 5.0 <= ev.started <= 5.7   # the 1 s majority vote flips about half a window after the helmet comes off
    assert ev.confirmed == pytest.approx(ev.started + EventSettings().dwell, abs=0.1)


def test_random_single_frame_misses_give_no_event():
    rng = random.Random(1)
    answers = [MISSING if rng.random() < 0.25 else WORN for _ in range(FPS * 600)]  # 25% of frames, 10 minutes
    engine = EventEngine("cam1")
    for k, a in enumerate(answers):
        engine.update(k / FPS, k, [(1, BOX, Verdict(a, WORN))])
    assert engine.events == []


def test_short_flicker_bursts_and_brief_removal_give_no_event():
    # a 1.5 s burst of misses every 10 s (e.g. the head turned away), and a real 1.5 s removal
    timeline = []
    for start in range(5, 60, 10):
        timeline += [(start - 5 if start > 5 else 0, start, 1, WORN), (start, start + 1.5, 1, MISSING),
                     (start + 1.5, start + 5, 1, WORN)]
    assert run(timeline) == []


def test_unknown_frames_never_become_an_event():
    assert run([(0, 30, 1, UNKNOWN)]) == []


def test_turning_away_briefly_does_not_restart_the_clock():
    # missing for 2 s, unknown for 0.6 s (looked away), missing again: one event, timed from the first start
    events = run([(0, 3, 1, WORN), (3, 5, 1, MISSING), (5, 5.6, 1, UNKNOWN), (5.6, 20, 1, MISSING)])
    assert len(events) == 1 and events[0].started < 4


def test_one_event_per_episode_and_cooldown_between_episodes():
    two_episodes = [(0, 10, 1, MISSING), (10, 20, 1, WORN), (20, 30, 1, MISSING)]
    assert len(run(two_episodes)) == 1                                           # second one is inside the 60 s cooldown
    assert len(run(two_episodes, EventSettings(cooldown=5))) == 2


def test_an_episode_that_outlasts_the_cooldown_is_raised_when_it_ends():
    events = run([(0, 10, 1, MISSING), (10, 20, 1, WORN), (20, 90, 1, MISSING)])
    assert len(events) == 2 and events[1].confirmed == pytest.approx(events[0].confirmed + 60, abs=0.1)


def test_new_track_id_at_the_same_place_is_the_same_incident():
    # track 1 is in violation, the tracker loses it and the same person comes back as track 2
    events = run([(0, 10, 1, MISSING), (10.2, 40, 2, MISSING)])
    assert len(events) == 1 and events[0].track_id == 1


def test_a_different_person_elsewhere_gets_their_own_event():
    engine = EventEngine("cam1")
    far = (0.80, 0.30, 0.90, 0.70)
    for k in range(40 * FPS):
        t = k / FPS
        people = [(1, BOX, Verdict(MISSING, WORN))] if t < 10 else [(2, far, Verdict(MISSING, WORN))]
        engine.update(t, k, people)
    assert [e.track_id for e in engine.events] == [1, 2]


def test_helmet_and_vest_are_separate_events():
    engine = EventEngine("cam1")
    for k in range(10 * FPS):
        engine.update(k / FPS, k, [(1, BOX, Verdict(MISSING, MISSING))])
    assert sorted(e.kind for e in engine.events) == ["no_helmet", "no_vest"]


@pytest.mark.parametrize("fps", [5, 30])
def test_timing_does_not_depend_on_frame_rate(fps):
    events = run([(0, 5, 1, WORN), (5, 30, 1, MISSING)], fps=fps, settings=EventSettings(dwell=2))
    assert len(events) == 1 and 7.4 <= events[0].confirmed <= 7.8  # 5 s + ~half the vote window + 2 s dwell


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError, match="dwel"):
        EventSettings.from_dict({"dwel": 3})


def test_the_same_person_on_two_tracks_gives_one_event():
    engine = EventEngine("cam1")
    other = (0.40, 0.30, 0.50, 0.62)  # a second, shorter box on the same person (IoU 0.8)
    for k in range(20 * FPS):
        engine.update(k / FPS, k, [(1, BOX, Verdict(MISSING, WORN)), (2, other, Verdict(MISSING, WORN))])
    assert len(engine.events) == 1


def test_a_track_id_handed_to_a_neighbour_does_not_duplicate_the_event():
    # person A (left, no helmet) is track 1; person B (right, helmet on) is track 2.
    # After an occlusion the tracker gives B the ID 1 and A a new ID 3.
    a, b = (0.30, 0.30, 0.40, 0.70), (0.45, 0.30, 0.55, 0.70)
    engine = EventEngine("cam1")
    for k in range(40 * FPS):
        t = k / FPS
        if t < 10:
            people = [(1, a, Verdict(MISSING, WORN)), (2, b, Verdict(WORN, WORN))]
        else:
            people = [(3, a, Verdict(MISSING, WORN)), (1, b, Verdict(WORN, WORN))]
        engine.update(t, k, people)
    assert [(e.kind, e.track_id) for e in engine.events] == [("no_helmet", 1)]
