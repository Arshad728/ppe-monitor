import pytest

from ppe_monitor.data.compliance import person_truths
from ppe_monitor.data.labels import ClassMap
from ppe_monitor.data.readers import Box
from ppe_monitor.rules.ppe import MISSING, UNKNOWN, WORN, Rules, judge

# A standing person 0.4 of the frame tall; helmet on the head, vest on the chest.
PERSON = (0.40, 0.30, 0.50, 0.70)
HELMET = (0.43, 0.29, 0.47, 0.34)
VEST = (0.41, 0.38, 0.49, 0.52)


def states(verdicts):
    return [(v.helmet, v.vest) for v in verdicts]


def test_helmet_on_head_and_vest_on_chest_are_worn():
    assert states(judge([PERSON], [HELMET], [VEST])) == [(WORN, WORN)]


def test_nothing_detected_means_missing():
    assert states(judge([PERSON], [], [])) == [(MISSING, MISSING)]


def test_helmet_carried_at_waist_is_not_worn():
    in_hand = (0.47, 0.50, 0.51, 0.55)  # centre at 55% of the person's height
    assert states(judge([PERSON], [in_hand], [VEST])) == [(MISSING, WORN)]


def test_vest_lying_on_the_ground_is_not_worn():
    on_ground = (0.52, 0.64, 0.60, 0.70)  # beside the person's feet
    assert states(judge([PERSON], [HELMET], [on_ground])) == [(WORN, MISSING)]


def test_each_helmet_goes_to_one_person():
    # two people side by side, only the left one has a helmet
    left, right = PERSON, (0.48, 0.30, 0.58, 0.70)  # boxes overlap a little
    verdicts = judge([left, right], [HELMET], [])
    assert [v.helmet for v in verdicts] == [WORN, MISSING]


def test_in_a_crowd_the_helmet_goes_to_the_head_it_sits_on():
    front = (0.40, 0.40, 0.50, 0.80)      # nearer the camera, head lower in the frame
    behind = (0.41, 0.30, 0.49, 0.62)     # further back, head higher; boxes overlap a lot
    helmet_behind = (0.43, 0.29, 0.47, 0.33)
    verdicts = judge([front, behind], [helmet_behind], [])
    assert [v.helmet for v in verdicts] == [MISSING, WORN]


def test_cut_off_or_tiny_people_are_unknown_not_violations():
    head_out_of_frame = (0.40, 0.0, 0.50, 0.40)
    feet_out_of_frame = (0.40, 0.70, 0.50, 1.0)
    tiny = (0.40, 0.30, 0.41, 0.36)  # 0.06 x 640 = 38 px tall
    rules = Rules(min_height_helmet=64, min_height_vest=48)
    assert states(judge([head_out_of_frame, feet_out_of_frame, tiny], [], [], rules)) == [
        (UNKNOWN, UNKNOWN), (MISSING, UNKNOWN), (UNKNOWN, UNKNOWN)]


def test_size_limit_uses_the_detectors_input_height():
    person = (0.40, 0.30, 0.50, 0.45)  # 0.15 of the frame: 96 px of a 640 square, 54 px of a 16:9 frame (360)
    rules = Rules(min_height_helmet=64)
    assert judge([person], [], [], rules, input_height=640)[0].helmet == MISSING
    assert judge([person], [], [], rules, input_height=360)[0].helmet == UNKNOWN


def test_unknown_rule_settings_are_rejected():
    with pytest.raises(ValueError, match="head_bottm"):
        Rules.from_dict({"head_bottm": 0.3})


CLASSES = ClassMap(["person", "helmet", "vest"], {"person": [], "helmet": ["hardhat"], "vest": []}, ["no_helmet", "none"])
ABSENT = {"nohelmet": "helmet", "none": "vest"}


def test_ground_truth_from_labels():
    boxes = [Box("person", 0.1, 0.1, 0.3, 0.9), Box("hardhat", 0.15, 0.1, 0.25, 0.2), Box("none", 0.12, 0.3, 0.28, 0.6),
             Box("person", 0.6, 0.1, 0.8, 0.9), Box("no_helmet", 0.65, 0.1, 0.75, 0.2),
             Box("person", 0.85, 0.1, 0.99, 0.9)]  # nothing labelled on this one
    truths = person_truths(boxes, CLASSES, ABSENT)
    assert [(t.helmet, t.vest) for t in truths] == [(True, False), (False, None), (None, None)]


def test_ambiguous_or_contradictory_labels_give_no_ground_truth():
    overlapping = [Box("person", 0.1, 0.1, 0.4, 0.9), Box("person", 0.12, 0.1, 0.42, 0.9),
                   Box("helmet", 0.2, 0.1, 0.3, 0.2)]           # inside both people
    contradictory = [Box("person", 0.1, 0.1, 0.4, 0.9), Box("helmet", 0.2, 0.1, 0.3, 0.2),
                     Box("no_helmet", 0.2, 0.1, 0.3, 0.2)]
    assert [t.helmet for t in person_truths(overlapping, CLASSES, ABSENT)] == [None, None]
    assert [t.helmet for t in person_truths(contradictory, CLASSES, ABSENT)] == [None]
