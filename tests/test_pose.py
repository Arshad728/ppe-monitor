"""Keypoints: posture, and helmets on people who are bent over, crouching or lying."""

import numpy as np

from ppe_monitor.rules.ppe import MISSING, UNKNOWN, WORN, Rules, judge
from ppe_monitor.vision.pose import BENDING, CROUCHING, LYING, UPRIGHT, Pose, match_poses


def pose(points: dict, box, aspect=1.0) -> Pose:
    """points: {keypoint index: (x, y)} in 0-1 frame fractions; the rest are invisible."""
    xy, conf = np.zeros((17, 2)), np.zeros(17)
    for i, (x, y) in points.items():
        xy[i], conf[i] = (x, y), 0.9
    return Pose(box, xy, conf, aspect)


HEAD = {0: (0.45, 0.26), 1: (0.44, 0.25), 2: (0.46, 0.25), 3: (0.43, 0.25), 4: (0.47, 0.25)}
STANDING = pose({**HEAD, 5: (0.42, 0.32), 6: (0.48, 0.32), 11: (0.43, 0.50), 12: (0.47, 0.50),
                 13: (0.43, 0.65), 14: (0.47, 0.65), 15: (0.43, 0.78), 16: (0.47, 0.78)}, (0.40, 0.22, 0.50, 0.80))
# bent right over, facing left: head near the bottom-left of a wide box, hips up at the right
BENT_BOX = (0.30, 0.40, 0.62, 0.80)
BENT = pose({0: (0.34, 0.66), 1: (0.335, 0.65), 3: (0.34, 0.64),
             5: (0.39, 0.60), 6: (0.40, 0.61), 11: (0.54, 0.47), 12: (0.55, 0.48),
             13: (0.55, 0.63), 14: (0.56, 0.63), 15: (0.55, 0.78), 16: (0.56, 0.78)}, BENT_BOX)
SQUATTING = pose({**HEAD, 5: (0.42, 0.32), 6: (0.48, 0.32), 11: (0.43, 0.48), 12: (0.47, 0.48),
                  13: (0.40, 0.45), 14: (0.50, 0.45), 15: (0.42, 0.56), 16: (0.48, 0.56)}, (0.38, 0.22, 0.52, 0.58))
LYING_DOWN = pose({0: (0.22, 0.70), 1: (0.22, 0.69), 5: (0.28, 0.70), 6: (0.28, 0.72), 11: (0.45, 0.70),
                   12: (0.45, 0.72), 15: (0.70, 0.71), 16: (0.70, 0.73)}, (0.18, 0.64, 0.74, 0.78))


def test_posture():
    assert STANDING.posture() == UPRIGHT
    assert BENT.posture() == BENDING
    assert SQUATTING.posture() == CROUCHING
    assert LYING_DOWN.posture() == LYING


def test_a_bent_workers_helmet_is_found_with_keypoints():
    helmet = (0.31, 0.62, 0.36, 0.69)  # on the head, near the bottom of the box: outside the upright head region
    assert judge([BENT_BOX], [helmet], [])[0].helmet == MISSING          # box rule alone gets it wrong
    v = judge([BENT_BOX], [helmet], [], poses=[BENT])[0]
    assert v.helmet == WORN and v.posture == BENDING


def test_lying_workers_helmet_is_found_with_keypoints():
    box = LYING_DOWN.box
    helmet = (0.19, 0.66, 0.24, 0.74)
    assert judge([box], [helmet], [], poses=[LYING_DOWN])[0].helmet == WORN


def test_a_helmet_carried_at_the_waist_is_still_not_worn():
    in_hand = (0.47, 0.50, 0.51, 0.55)
    assert judge([STANDING.box], [in_hand], [], poses=[STANDING])[0].helmet == MISSING


def test_no_helmet_while_bent_over_is_missing_if_the_head_is_in_view():
    # a worker without a helmet who stays bent over (tying rebar) must still be caught
    v = judge([BENT_BOX], [], [], poses=[BENT])[0]
    assert v.helmet == MISSING and v.posture == BENDING


def test_no_helmet_while_bent_over_with_the_head_out_of_sight_is_unknown():
    # seen from behind: shoulders, hips and legs in view, no nose / eye / ear keypoint
    back = pose({5: (0.39, 0.60), 6: (0.40, 0.61), 11: (0.54, 0.47), 12: (0.55, 0.48),
                 13: (0.55, 0.63), 14: (0.56, 0.63), 15: (0.55, 0.78), 16: (0.56, 0.78)}, BENT_BOX)
    assert back.posture() == BENDING and back.head is None
    assert judge([BENT_BOX], [], [], poses=[back])[0].helmet == UNKNOWN
    assert judge([BENT_BOX], [], [], Rules(bent_head_hidden_is_unknown=False), poses=[back])[0].helmet == MISSING
    # an upright person seen from behind is judged by the box rule as before
    upright_back = pose({5: (0.42, 0.32), 6: (0.48, 0.32), 11: (0.43, 0.50), 12: (0.47, 0.50),
                         15: (0.43, 0.78), 16: (0.47, 0.78)}, STANDING.box)
    assert judge([STANDING.box], [], [], poses=[upright_back])[0].helmet == MISSING


def test_keypoints_are_matched_to_the_right_person():
    far = (0.70, 0.22, 0.80, 0.80)
    assert match_poses([far, STANDING.box], [STANDING]) == [None, STANDING]


def test_distances_respect_the_frame_shape():
    # on a 16:9 frame, 0.01 of the width is 1.78x as far as 0.01 of the height
    wide = pose({0: (0.5, 0.3), 5: (0.49, 0.4), 6: (0.51, 0.4)}, (0.45, 0.25, 0.55, 0.9), aspect=16 / 9)
    assert abs(wide.head[0] - 0.5 * 16 / 9) < 1e-9
