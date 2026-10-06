"""Phase 6: false alarms from the dashboard as training data."""

import json

import cv2
import numpy as np
import yaml

from ppe_monitor.data.feedback import (FeedbackItem, build_dataset, draw_review, eligible_camera, label_false_alarm,
                                       load_index, save_index, write_label_file)
from ppe_monitor.vision.matching import Det

TH = {"person": 0.4, "helmet": 0.25, "vest": 0.25}
GEOM = {"head_top": -0.10, "head_bottom": 0.35, "head_side": 0.05, "torso_top": 0.15, "torso_bottom": 0.85, "torso_side": 0.0}
PERSON = (0.40, 0.20, 0.60, 0.90)
ON_HEAD = (0.46, 0.18, 0.54, 0.28)


def item(kind="no_helmet"):
    return FeedbackItem("e1", "gate", kind, "2026-09-26T12:00:00", list(PERSON))


def test_a_missed_helmet_is_found_again_at_low_confidence_and_added():
    dets = [Det(0, PERSON, 0.9), Det(1, ON_HEAD, 0.12), Det(1, (0.9, 0.9, 0.95, 0.95), 0.6)]
    it = label_false_alarm(item(), dets, TH, GEOM)
    assert it.status == "ready" and "0.12" in it.reason
    assert [1, *ON_HEAD] in it.labels and it.added == [[1, *ON_HEAD]]
    assert [0, *PERSON] in it.labels and len(it.labels) == 3      # the other confident helmet stays


def test_a_small_helmet_of_someone_behind_is_not_taken_for_the_persons_own():
    tiny = (0.47, 0.20, 0.49, 0.22)                                # 10 % of the person's width
    assert label_false_alarm(item(), [Det(0, PERSON, 0.9), Det(1, tiny, 0.3)], TH, GEOM).status == "needs_box"


def test_no_helmet_anywhere_near_the_head_needs_a_box_by_hand():
    it = label_false_alarm(item(), [Det(0, PERSON, 0.9), Det(1, (0.46, 0.7, 0.54, 0.8), 0.3)], TH, GEOM)
    assert it.status == "needs_box" and it.added == []
    assert "not a person" in it.reason              # a machine taken for a person: the person line goes instead


def test_when_the_detector_was_right_the_image_is_not_needed():
    it = label_false_alarm(item(), [Det(0, PERSON, 0.9), Det(1, ON_HEAD, 0.8)], TH, GEOM)
    assert it.status == "not_needed"


def test_the_person_is_added_when_the_model_missed_them_too():
    it = label_false_alarm(item(), [Det(1, ON_HEAD, 0.1)], TH, GEOM)
    assert it.status == "ready" and [0, *PERSON] in it.labels and len(it.added) == 2


def test_a_vest_is_looked_for_on_the_torso_and_zone_alarms_go_to_a_person():
    torso = (0.42, 0.40, 0.58, 0.65)
    it = label_false_alarm(item("no_vest"), [Det(0, PERSON, 0.9), Det(2, torso, 0.07)], TH, GEOM)
    assert it.status == "ready" and it.added == [[2, *torso]]
    assert label_false_alarm(item("zone_intrusion"), [Det(0, PERSON, 0.9)], TH, GEOM).status == "review"


def test_test_footage_is_never_training_data():
    cat = {"workshop_handheld_day_01.mp4": "test", "site_morning_01.mp4": "train", "synthetic_cam1.mp4": "plumbing-only"}
    assert eligible_camera("gate", "Main gate", cat, set()) == (True, "")
    assert eligible_camera("cam1", "Main gate (simulated)", cat, {"cam1"})[0] is False
    assert eligible_camera("clip1", "workshop_handheld_day_01.mp4", cat, set())[0] is False
    assert eligible_camera("clip2", "site_morning_01.mp4", cat, set())[0] is True
    ok, why = eligible_camera("mv-000_violation", "moving: 000_violation.mp4", cat, set())
    assert not ok and "not in data/clips/clips_catalog.csv" in why


def test_label_file_and_review_picture(tmp_path):
    write_label_file(tmp_path / "a.txt", [[1, 0.4, 0.2, 0.6, 0.4], [0, -0.1, 0.0, 0.5, 1.2], [2, 0.5, 0.5, 0.5, 0.6]])
    lines = (tmp_path / "a.txt").read_text().splitlines()
    assert lines[0] == "1 0.500000 0.300000 0.200000 0.200000"
    assert lines[1] == "0 0.250000 0.500000 0.500000 1.000000" and len(lines) == 2   # clipped; empty box dropped
    it = label_false_alarm(item(), [Det(0, PERSON, 0.9), Det(1, ON_HEAD, 0.1)], TH, GEOM)
    img = draw_review(np.zeros((400, 600, 3), np.uint8), it, width=300)
    assert img.shape == (200, 300, 3) and img.any()


def test_build_adds_ready_feedback_to_training_only(tmp_path):
    base = tmp_path / "ppe4"
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True)
        for k in range(3):
            cv2.imwrite(str(base / "images" / split / f"{split}{k}.jpg"), np.zeros((8, 8, 3), np.uint8))
    (base / "lists").mkdir()
    (base / "lists" / "test_extra.txt").write_text("images/test/test0.jpg\n")
    (base / "data.yaml").write_text(yaml.safe_dump({"path": str(base), "train": "images/train", "val": "images/val",
                                                    "test": "images/test", "test_extra": "lists/test_extra.txt",
                                                    "names": ["person", "helmet", "vest"]}))
    fb = tmp_path / "feedback"
    (fb / "images").mkdir(parents=True)
    items = []
    for eid, status in (("a", "ready"), ("b", "ready"), ("c", "needs_box"), ("d", "ready")):
        cv2.imwrite(str(fb / "images" / f"{eid}.jpg"), np.zeros((8, 8, 3), np.uint8))
        items.append(FeedbackItem(eid, "gate", "no_helmet", "t", list(PERSON), status))
    save_index(fb, items)
    (fb / "skip.txt").write_text("d  # not sure about this one\n")
    assert [i.status for i in load_index(fb)] == ["ready", "ready", "needs_box", "ready"]
    c = build_dataset(base / "data.yaml", tmp_path / "ppe5", fb, repeat=3)
    assert c == {"base_train": 3, "feedback": 2, "repeat": 3, "train": 9, "skipped": 1}
    cfg = yaml.safe_load((tmp_path / "ppe5" / "data.yaml").read_text())
    assert cfg["train"] == "train.txt" and cfg["test_extra"] == "test_extra.txt" and cfg["names"][1] == "helmet"
    train = (tmp_path / "ppe5" / "train.txt").read_text().split()
    assert sum(p.endswith("/a.jpg") for p in train) == 3 and not any(p.endswith(("/c.jpg", "/d.jpg")) for p in train)
    val = (tmp_path / "ppe5" / "val.txt").read_text().split()
    assert len(val) == 3 and all("/images/val/" in p for p in val)
    assert (tmp_path / "ppe5" / "test_extra.txt").read_text().strip().endswith("ppe4/images/test/test0.jpg")
    assert len(json.loads((fb / "feedback.json").read_text())) == 4


def test_a_retrain_builds_on_the_data_the_model_in_use_was_trained_on(tmp_path, monkeypatch):
    import os
    import time

    from ppe_monitor.data import feedback as fb
    monkeypatch.setattr(fb, "PROJECT_ROOT", tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (tmp_path / "datasets" / "ppe4caps").mkdir(parents=True)
    (tmp_path / "datasets" / "ppe4caps" / "data.yaml").write_text("path: x\n")
    (models / "old.pt").write_bytes(b"")
    (models / "new.pt").write_bytes(b"")
    os.utime(models / "old.pt", (time.time() - 100, time.time() - 100))
    (models / "new.json").write_text(json.dumps({"data": "datasets/ppe4caps/data.yaml"}))
    assert fb.base_dataset(models) == tmp_path / "datasets" / "ppe4caps" / "data.yaml"
    (models / "new.json").write_text(json.dumps({"data": "datasets/gone/data.yaml"}))
    assert fb.base_dataset(models) == tmp_path / "datasets" / "ppe4" / "data.yaml"     # the default when it's gone
