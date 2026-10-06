from ppe_monitor.vision.matching import Det
from ppe_monitor.vision.review import classify, miss_cause

NAMES = ["person", "helmet", "vest"]
PERSON, HELMET, VEST = 0, 1, 2


def kinds(errors):
    return sorted(k for k, _ in errors)


def test_each_error_type_is_recognised():
    gts = [Det(PERSON, (0.1, 0.1, 0.4, 0.9)),
           Det(HELMET, (0.2, 0.1, 0.3, 0.2)),
           Det(VEST, (0.15, 0.3, 0.35, 0.6)),
           Det(PERSON, (0.6, 0.1, 0.9, 0.9))]           # nobody finds this one
    preds = [Det(PERSON, (0.1, 0.1, 0.4, 0.9), 0.9),     # correct
             Det(PERSON, (0.11, 0.1, 0.41, 0.9), 0.8),   # duplicate of the first person
             Det(HELMET, (0.15, 0.3, 0.35, 0.6), 0.7),   # a helmet box on the vest: wrong class
             Det(VEST, (0.7, 0.92, 0.8, 0.99), 0.6),     # nothing there: false alarm
             Det(HELMET, (0.2, 0.1, 0.3, 0.2), 0.95)]    # correct
    errors, good, found = classify(preds, gts, NAMES)
    assert kinds(errors) == ["false helmet (wrong class)", "false person (duplicate)", "false vest (false alarm)",
                             "missed person", "missed vest"]
    assert good == {0, 4}
    assert found == {0, 1}


def test_perfect_predictions_have_no_errors():
    gts = [Det(PERSON, (0.1, 0.1, 0.4, 0.9)), Det(VEST, (0.15, 0.3, 0.35, 0.6))]
    preds = [Det(g.cls, g.box, 0.9) for g in gts]
    errors, good, found = classify(preds, gts, NAMES)
    assert errors == [] and good == {0, 1} and found == {0, 1}


def test_a_box_near_an_object_but_too_far_off_is_misplaced():
    gts = [Det(PERSON, (0.1, 0.1, 0.3, 0.9))]
    preds = [Det(PERSON, (0.1, 0.1, 0.3, 0.4), 0.8)]         # covers only the top third: IoU 0.375
    errors, _, _ = classify(preds, gts, NAMES)
    assert kinds(errors) == ["false person (misplaced)", "missed person"]


def test_miss_causes():
    person = Det(PERSON, (0.1, 0.1, 0.3, 0.9))
    t = 0.35
    assert miss_cause(person, [Det(PERSON, (0.1, 0.1, 0.3, 0.9), 0.9)], t) == "crowded"       # good box, went elsewhere
    assert miss_cause(person, [Det(PERSON, (0.1, 0.1, 0.3, 0.4), 0.9)], t) == "misplaced box"
    assert miss_cause(person, [Det(PERSON, (0.1, 0.1, 0.3, 0.9), 0.2)], t) == "below threshold"
    assert miss_cause(person, [Det(VEST, (0.1, 0.1, 0.3, 0.9), 0.9),                            # other class: ignored
                               Det(PERSON, (0.6, 0.1, 0.9, 0.9), 0.9)], t) == "not detected"
