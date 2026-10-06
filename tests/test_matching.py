import math

import numpy as np
import pytest

from ppe_monitor.vision.matching import Det, average_precision, best_f1_threshold, f1_by_threshold, iou_matrix, match_image

PERSON, HELMET = 0, 1


def test_iou_known_values():
    a = np.array([[0, 0, 0.5, 0.5]])
    b = np.array([[0, 0, 0.5, 0.5], [0.25, 0, 0.75, 0.5], [0.6, 0.6, 1, 1]])
    assert iou_matrix(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_match_counts_hits_false_alarms_and_misses():
    gts = [Det(PERSON, (0.1, 0.1, 0.4, 0.9)), Det(HELMET, (0.2, 0.1, 0.3, 0.2)), Det(PERSON, (0.6, 0.1, 0.9, 0.9))]
    preds = [Det(PERSON, (0.11, 0.1, 0.41, 0.9), 0.9),   # hits person 0
             Det(PERSON, (0.12, 0.1, 0.42, 0.9), 0.8),   # duplicate of the same person -> false positive
             Det(HELMET, (0.6, 0.1, 0.9, 0.9), 0.7),     # right place, wrong class -> false positive
             Det(HELMET, (0.2, 0.1, 0.3, 0.2), 0.6)]     # hits the helmet
    m = match_image(preds, gts)
    assert sorted(m.tp) == [0, 3] and sorted(m.fp) == [1, 2] and m.fn == [2]
    assert m.pairs == {0: 0, 3: 1}


def test_more_confident_prediction_claims_the_box():
    gts = [Det(PERSON, (0.1, 0.1, 0.5, 0.5))]
    preds = [Det(PERSON, (0.1, 0.1, 0.5, 0.5), 0.3), Det(PERSON, (0.12, 0.1, 0.5, 0.5), 0.9)]
    m = match_image(preds, gts)
    assert m.tp == [1] and m.fp == [0]


def test_average_precision_edge_cases():
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([0.9, 0.8], [False, True], 1) == pytest.approx(0.5)
    assert average_precision([0.9], [True], 2) == pytest.approx(0.5)   # half the objects never found
    assert average_precision([], [], 3) == 0.0
    assert math.isnan(average_precision([0.5], [False], 0))


def test_average_precision_matches_hand_computed_curve():
    # hits at ranks 1 and 3 out of 2 objects: precision 1 at recall .5, then 2/3 at recall 1
    assert average_precision([0.9, 0.8, 0.7], [True, False, True], 2) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_f1_picks_threshold_that_drops_low_confidence_noise():
    gts = [Det(PERSON, (0.1, 0.1, 0.4, 0.9))]
    preds = [Det(PERSON, (0.1, 0.1, 0.4, 0.9), 0.9), Det(PERSON, (0.6, 0.1, 0.9, 0.9), 0.2)]
    f1 = f1_by_threshold([(preds, gts)], np.array([0.1, 0.5]))
    assert f1[0] == pytest.approx(2 / 3) and f1[1] == pytest.approx(1.0)


def test_best_threshold_works_whatever_the_score_range():
    """Regression: an under-trained model scored at most 0.012, below a fixed 0.05..0.95 grid."""
    gts = [Det(PERSON, (0.1, 0.1, 0.4, 0.9)), Det(PERSON, (0.6, 0.1, 0.9, 0.9))]
    preds = [Det(PERSON, (0.1, 0.1, 0.4, 0.9), 0.012), Det(PERSON, (0.6, 0.1, 0.9, 0.9), 0.011),
             Det(PERSON, (0.4, 0.5, 0.5, 0.6), 0.003)]
    t, f1 = best_f1_threshold([(preds, gts)])
    assert t == pytest.approx(0.011) and f1 == pytest.approx(1.0)


def test_best_threshold_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    images = []
    for _ in range(20):
        gts = [Det(int(rng.integers(2)), tuple(np.sort(rng.random(2)).tolist()[:1] + [0.1] + [0.9, 0.9])) for _ in range(3)]
        gts = [Det(g.cls, (g.box[0] * 0.5, 0.1, g.box[0] * 0.5 + 0.3, 0.9)) for g in gts]
        preds = [Det(g.cls, (g.box[0] + rng.normal(0, 0.03), 0.1, g.box[2] + rng.normal(0, 0.03), 0.9), float(rng.random()))
                 for g in gts] + [Det(int(rng.integers(2)), (0.7, 0.7, 0.8, 0.8), float(rng.random()))]
        images.append((preds, gts))
    t, f1 = best_f1_threshold(images)
    grid = np.array(sorted({p.score for preds, _ in images for p in preds}))
    brute = f1_by_threshold(images, grid)
    assert f1 == pytest.approx(brute.max())
    assert f1_by_threshold(images, np.array([t]))[0] == pytest.approx(f1)
