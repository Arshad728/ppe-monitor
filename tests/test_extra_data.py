"""The extra datasets for the fine-tuned model: readers, label check, merge with ppe3, overfitting check."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ppe_monitor.data import audit as audit_mod
from ppe_monitor.data.audit import audit_source, head_has_person
from ppe_monitor.data.build_ext import build_extended
from ppe_monitor.data.compliance import absence_labels
from ppe_monitor.data.extra_sources import EXTRA_SOURCES
from ppe_monitor.data.labels import ClassMap
from ppe_monitor.data.readers import Box, ReadStats, Sample, read_chv, read_voc
from ppe_monitor.vision.detector import split_items
from ppe_monitor.vision.matching import Det
from ppe_monitor.vision.overfit import overfitting_check


def image(path: Path, seed: int = 0, size=(64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    return path


def test_new_labels_are_mapped():
    m = ClassMap.from_yaml()
    for colour in ("blue", "red", "white", "yellow"):
        assert m.decide(f"{colour} helmet").class_name == "helmet"
    assert m.decide("safety-vest").class_name == "vest"
    for dropped in ("head", "bare head", "safety-suit", "tool", "ear-mufs"):
        assert m.decide(dropped).kind == "ignore"
    absent = absence_labels()
    assert absent["barehead"] == "helmet" and "head" not in absent     # SH17's head is any head, not a bare one


def test_read_chv(tmp_path):
    root = tmp_path / "CHV_dataset"
    image(root / "images" / "ppe_0001.jpg")
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "ppe_0001.txt").write_text("2 0.5 0.2 0.2 0.2\n0 0.5 0.5 0.4 0.9\n")
    (root / "data split").mkdir()
    (root / "data split" / "train.txt").write_text("CHV_dataset/images/ppe_0001.jpg\n")
    samples, stats = read_chv(root, "chv", {"train": "train"})
    assert [b.label for b in samples[0].boxes] == ["blue helmet", "person"]
    assert samples[0].split == "train" and stats.images == 1


def test_read_voc_renames_colours_and_bare_heads(tmp_path):
    root = tmp_path / "gdut"
    image(root / "JPEGImages" / "00001.jpg", size=(200, 100))
    (root / "Annotations").mkdir(parents=True)
    (root / "Annotations" / "00001.xml").write_text(
        "<annotation><size><width>200</width><height>100</height></size>"
        "<object><name>blue</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>"
        "<object><name>none</name><bndbox><xmin>50</xmin><ymin>10</ymin><xmax>70</xmax><ymax>30</ymax></bndbox></object>"
        "</annotation>")
    (root / "ImageSets" / "Main").mkdir(parents=True)
    (root / "ImageSets" / "Main" / "trainval.txt").write_text("00001\n")
    samples, _ = read_voc(root, "gdut-hwd", {"trainval": "train"})
    b = samples[0].boxes
    assert [x.label for x in b] == ["blue helmet", "bare head"]
    assert b[0].x1 == pytest.approx(0.05) and b[0].y2 == pytest.approx(0.3)
    assert samples[0].split in ("train", "val")


def fake_predict(table):
    """predict() stand-in: model name -> {image name: [Det, ...]}."""
    def predict(model, images, **kw):
        return [table[model].get(Path(p).name, []) for p in images]
    return predict


def test_audit_adds_people_and_drops_incomplete_images(tmp_path, monkeypatch):
    src = EXTRA_SOURCES["gdut-hwd"]                      # persons "pseudo", audit helmet 0.5 / vest 0.3
    head = Box("blue helmet", 0.45, 0.1, 0.55, 0.2)
    ok = Sample("gdut-hwd", "train", tmp_path / "ok.jpg", [head])
    lonely = Sample("gdut-hwd", "train", tmp_path / "lonely.jpg", [head])     # the COCO model finds nobody
    vest = Sample("gdut-hwd", "train", tmp_path / "vest.jpg", [head])         # a vest nobody labelled
    person = Det(0, (0.4, 0.05, 0.6, 0.95), 0.9)
    table = {"coco": {"ok.jpg": [person], "vest.jpg": [person]},
             "ppe": {"vest.jpg": [Det(2, (0.42, 0.3, 0.58, 0.6), 0.4)]}}
    monkeypatch.setattr("ppe_monitor.vision.detector.predict", fake_predict(table))
    kept, stats = audit_source(src, [ok, lonely, vest], ClassMap.from_yaml(), ppe_model="ppe", coco_model="coco",
                               absent_helmet={"barehead"}, log=lambda *a: None)
    assert [s.image.name for s in kept] == ["ok.jpg"]
    assert [b.label for b in kept[0].boxes] == ["blue helmet", "person"]
    assert stats.dropped["a labelled head with no person found around it"] == 1
    assert stats.dropped["an unlabelled vest"] == 1


def test_head_must_sit_at_the_top_of_a_person():
    person = (0.4, 0.1, 0.6, 0.9)
    assert head_has_person(Box("h", 0.45, 0.1, 0.55, 0.2), [person])
    assert not head_has_person(Box("h", 0.45, 0.7, 0.55, 0.8), [person])      # at the feet: someone else's


def make_base(tmp: Path) -> Path:
    base = tmp / "ppe3"
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        image(base / "images" / split / f"construction-ppe__{split}__a.jpg", seed)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split / f"construction-ppe__{split}__a.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    (base / "data.yaml").write_text(yaml.safe_dump({"path": str(base), "train": "images/train", "val": "images/val",
                                                    "test": "images/test", "names": ["person", "helmet", "vest"]}))
    (base / "report.json").write_text(json.dumps({"sources": [{"name": "construction-ppe", "title": "C", "licence": "L",
                                                                "attribution": "A", "sha256": "", "notes": ""}]}))
    return base


def test_merge_keeps_ppe3_and_keeps_its_tests_clean(tmp_path):
    base = make_base(tmp_path)
    twin_of_test = image(tmp_path / "x" / "twin.jpg", 3)          # the same pixels as the ppe3 test image
    fresh_train = image(tmp_path / "x" / "fresh.jpg", 7)
    fresh_val = image(tmp_path / "x" / "fresh_val.jpg", 7)        # same photo as fresh_train, but in val
    fresh_test = image(tmp_path / "x" / "blue.jpg", 9)
    helmet = [Box("blue helmet", 0.4, 0.1, 0.5, 0.2), Box("person", 0.35, 0.05, 0.55, 0.9)]
    samples = [Sample("chv", "train", twin_of_test, helmet), Sample("chv", "train", fresh_train, helmet),
               Sample("chv", "val", fresh_val, helmet), Sample("chv", "test", fresh_test, helmet)]
    stats = ReadStats()
    stats.labels.update({"blue helmet": 4, "person": 4})
    audit = audit_mod.AuditStats(checked=4, kept=4)
    out = tmp_path / "ppe4"
    r = build_extended(base, [(EXTRA_SOURCES["chv"], samples, stats, audit)], out, ClassMap.from_yaml(), log=lambda *a: None)
    assert r.left_out_near_ppe3_test == {"chv": 1}                 # it would leak the ppe3 test image
    assert r.moved_to_train == {"chv val": 1}
    assert (out / "images" / "test" / "construction-ppe__test__a.jpg").is_file()     # ppe3 untouched
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert {"test_ppe3", "test_chv", "test_extra"} <= set(cfg)
    assert [p.name for p, _ in split_items(out / "data.yaml", "test_extra")] == ["chv__test__blue.jpg"]
    assert [p.name for p, _ in split_items(out / "data.yaml", "test_chv")] == ["chv__test__blue.jpg"]
    assert json.loads((out / "helmet_colours.json").read_text())["chv__test__blue.jpg"][0][0] == "blue"
    labels = (out / "labels" / "test" / "chv__test__blue.txt").read_text().split()
    assert labels[0] == "1" and "0" in labels[::5]                 # helmet and person, in our class ids


def write_results(path: Path, train, val, m95):
    lines = ["epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss"]
    for i, (t, v, m) in enumerate(zip(train, val, m95)):
        lines.append(f"{i + 1},{t / 2},{t / 2},{m + 0.3},{m},{v / 2},{v / 2}")
    path.write_text("\n".join(lines) + "\n")


def test_overfitting_check(tmp_path):
    csv = tmp_path / "results.csv"
    # healthy: best near the end, small gap
    write_results(csv, [3, 2.5, 2.2, 2.0, 1.9], [3, 2.6, 2.4, 2.35, 2.36], [0.3, 0.4, 0.45, 0.48, 0.47])
    text, info = overfitting_check(csv, 0.90, 300, 0.88)
    assert not info["kept_model_overfit"] and info["best_epoch"] == 4 and "No sign" in text
    # later epochs memorise: val loss climbs while train loss falls; the kept (best) model is fine
    write_results(csv, [3, 2, 1.5, 1.2, 1.0, 0.8], [3, 2.2, 2.3, 2.5, 2.7, 2.9], [0.3, 0.5, 0.49, 0.47, 0.45, 0.44])
    text, info = overfitting_check(csv, 0.92, 300, 0.89)
    assert info["later_epochs_memorising"] and not info["kept_model_overfit"] and info["best_epoch"] == 2
    # a large train/val gap in the kept model
    text, info = overfitting_check(csv, 0.99, 300, 0.80)
    assert info["kept_model_overfit"] and "Signs of overfitting" in text


def test_accept_names_the_new_model_in_the_config(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    cm = importlib.import_module("compare_models")
    cfg = tmp_path / "ppe.yaml"
    cfg.write_text("# the model: line below names the detector\nmodel: ppe3_yolo26n_baseline\nthresholds:\n  helmet: 0.25\n")
    assert cm.mark_config(cfg, "ppe4_yolo26n_finetune")
    text = cfg.read_text()
    assert yaml.safe_load(text) == {"model": "ppe4_yolo26n_finetune", "thresholds": {"helmet": 0.25}}
    assert text.startswith("# the model: line below")            # comments untouched
    assert not cm.mark_config(cfg, "ppe4_yolo26n_finetune")      # already right: nothing to write
