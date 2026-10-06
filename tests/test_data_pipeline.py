"""The dataset pipeline on tiny made-up datasets: label mapping, both input formats,
near-duplicate handling and the written output."""

import json

import cv2
import numpy as np
import pytest
import yaml

from ppe_monitor.data.build import UnknownLabels, build_dataset, size_bucket
from ppe_monitor.data.dedup import dhash, groups_from_pairs, near_duplicate_pairs
from ppe_monitor.data.labels import ClassMap, normalise
from ppe_monitor.data.readers import read_coco, read_yolo
from ppe_monitor.data.sources import Source

CLASS_MAP = ClassMap(["person", "helmet", "vest"],
                     {"person": ["Person", "worker"], "helmet": ["hardhat"], "vest": ["safety vest"]},
                     ["no-helmet", "none", "gloves"])


def scene(seed: int, size=(96, 128)) -> np.ndarray:
    """A random but structured picture; different seeds look different to the fingerprint."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), size[::-1], interpolation=cv2.INTER_CUBIC)
    return img


def make_yolo(root, split_images: dict[str, dict[str, np.ndarray]], names: list[str], labels: dict[str, str]):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    for split, images in split_images.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for stem, img in images.items():
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img)
            if stem in labels:
                (root / "labels" / split / f"{stem}.txt").write_text(labels[stem], encoding="utf-8")


def make_coco(root, split: str, images: dict[str, np.ndarray], anns: dict[str, list[tuple[str, list[float]]]]):
    d = root / split
    d.mkdir(parents=True)
    cats = sorted({c for boxes in anns.values() for c, _ in boxes})
    coco = {"categories": [{"id": i, "name": c} for i, c in enumerate(cats)], "images": [], "annotations": []}
    for n, (name, img) in enumerate(images.items()):
        cv2.imwrite(str(d / f"{name}.jpg"), img)
        coco["images"].append({"id": n, "file_name": f"{name}.jpg", "width": img.shape[1], "height": img.shape[0]})
        for c, bbox in anns.get(name, []):
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": n,
                                        "category_id": cats.index(c), "bbox": bbox})
    (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def test_normalise_and_decisions():
    assert normalise("Hard-Hat") == normalise("hard hat") == normalise("HARD_HAT") == "hardhat"
    assert CLASS_MAP.decide("Person").class_name == "person"
    assert CLASS_MAP.decide("Hard Hat").class_id == 1
    assert CLASS_MAP.decide("NO_HELMET").kind == "ignore"
    assert CLASS_MAP.decide("drone").kind == "unknown"


def test_class_map_rejects_contradictions():
    with pytest.raises(ValueError):
        ClassMap(["person"], {"person": ["worker"]}, ["Worker"])


def test_project_class_config_covers_both_real_datasets():
    cm = ClassMap.from_yaml()
    real_labels = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person", "no_helmet", "no_goggle",
                   "no_gloves", "no_boots", "construction-safety", "no-helmet", "no-vest", "person"]
    assert all(cm.decide(label).kind != "unknown" for label in real_labels)
    assert [cm.decide(x).class_name for x in ("Person", "helmet", "vest")] == ["person", "helmet", "vest"]


def test_read_yolo_converts_centres_and_counts_problems(tmp_path):
    make_yolo(tmp_path, {"train": {"a": scene(1), "b": scene(2)}}, ["Person", "hardhat"],
              {"a": "0 0.5 0.5 0.4 0.6\n1 0.5 0.2 0.1 0.1\n0 0.5 0.5 0 0.3\n"})
    (tmp_path / "labels" / "train" / "zzz.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    samples, stats = read_yolo(tmp_path, "src", {"train": "train"})
    a = next(s for s in samples if s.image.stem == "a")
    person = a.boxes[0]
    assert (person.label, round(person.x1, 3), round(person.y1, 3), round(person.x2, 3), round(person.y2, 3)) == \
        ("Person", 0.3, 0.2, 0.7, 0.8)
    assert stats.degenerate_boxes == 1           # the zero-width box
    assert stats.images_without_label_file == 1  # image b
    assert stats.orphan_label_files == 1         # zzz.txt has no image


def test_read_coco_normalises_pixel_boxes(tmp_path):
    make_coco(tmp_path, "valid", {"x": scene(3, (100, 200))}, {"x": [("worker", [20, 10, 40, 50])]})
    samples, _ = read_coco(tmp_path, "src", {"valid": "val"})
    b = samples[0].boxes[0]
    assert samples[0].split == "val"
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.1, 0.1, 0.3, 0.6))


def test_fingerprint_matches_resized_copy_but_not_other_scene(tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), scene(10, (240, 320)))
    cv2.imwrite(str(tmp_path / "a_small.jpg"), cv2.resize(scene(10, (240, 320)), (160, 120)))
    cv2.imwrite(str(tmp_path / "b.jpg"), scene(11, (240, 320)))
    h = [dhash(tmp_path / n) for n in ("a.jpg", "a_small.jpg", "b.jpg")]
    pairs = near_duplicate_pairs(h, 6)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert groups_from_pairs(3, pairs)[:2] == [0, 0] and groups_from_pairs(3, pairs)[2] == 2


def test_build_moves_leaked_test_images_and_drops_cross_source_copies(tmp_path):
    shared = scene(100)
    y = tmp_path / "y"
    make_yolo(y, {"train": {"t1": shared, "t2": scene(101)}, "test": {"leak": shared.copy(), "clean": scene(102)}},
              ["Person", "hardhat", "none"],
              {"t1": "0 0.5 0.5 0.4 0.6\n2 0.5 0.5 0.2 0.2\n", "t2": "1 0.5 0.2 0.1 0.1\n",
               "leak": "0 0.5 0.5 0.4 0.6\n", "clean": "0 0.5 0.5 0.4 0.6\n1 0.5 0.2 0.05 0.05\n"})
    c = tmp_path / "c"
    make_coco(c, "train", {"dup": scene(101), "own": scene(103)},
              {"dup": [("hardhat", [10, 10, 20, 20])], "own": [("safety vest", [30, 30, 40, 40])]})
    ys = Source("yolo-src", "Y", "", "", 0, "", "", "yolo", {"train": "train", "test": "test"})
    cs = Source("coco-src", "C", "", "", 0, "", "", "coco", {"train": "train"})
    out = tmp_path / "out"
    report = build_dataset([(ys, y), (cs, c)], out, CLASS_MAP, log=lambda *_: None)

    assert report.moved_to_train == {"test": 1}                         # "leak" had a twin in train
    assert report.exact_duplicates_dropped == 1                         # coco "dup" == yolo "t2"
    assert report.images == {"train": 4, "val": 0, "test": 1}
    assert sorted(p.name for p in (out / "images" / "test").iterdir()) == ["yolo-src__test__clean.jpg"]
    assert report.boxes["test"] == {"person": 1, "helmet": 1, "vest": 0}
    t1 = (out / "labels" / "train" / "yolo-src__train__t1.txt").read_text().split()
    assert t1[0] == "0" and len(t1) == 5                                # "none" was dropped
    own = (out / "labels" / "train" / "coco-src__train__own.txt").read_text().split()
    assert own[0] == "2"                                                # "safety vest" -> vest
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["person", "helmet", "vest"] and data["test"] == "images/test"
    assert "Keeping the test honest" in (out / "report.md").read_text()


def test_unknown_label_stops_the_build(tmp_path):
    make_yolo(tmp_path / "y", {"train": {"a": scene(1)}}, ["Person", "drone"], {"a": "1 0.5 0.5 0.2 0.2\n"})
    ys = Source("y", "Y", "", "", 0, "", "", "yolo", {"train": "train"})
    with pytest.raises(UnknownLabels, match="drone"):
        build_dataset([(ys, tmp_path / "y")], tmp_path / "out", CLASS_MAP, log=lambda *_: None)


def test_size_buckets_follow_coco_at_640():
    assert size_bucket((30 / 640) ** 2) == "small"
    assert size_bucket((60 / 640) ** 2) == "medium"
    assert size_bucket((200 / 640) ** 2) == "large"
