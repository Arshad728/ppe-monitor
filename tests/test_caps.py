"""Photos of people in hats and caps (not helmets) as training data: data/caps.py, select_caps.py."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ppe_monitor.data import caps
from ppe_monitor.vision.matching import Det

ROOT = Path(__file__).resolve().parents[1]


def _image(path: Path, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path.read_bytes()


def _entry(i: str, split: str, data: bytes, url: str) -> dict:
    person = [0.3, 0.2, 0.6, 0.9]
    return {"id": i, "oi_split": "validation", "split": split, "url": url, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "labels": [[0, *person], [0, 0.7, 0.3, 0.8, 0.9]],
            "hats": [{"box": [0.38, 0.17, 0.52, 0.27], "kinds": ["hat"], "person": person, "helmet_conf": 0.6}],
            "credit": {"author": "A. Photographer", "licence": "https://creativecommons.org/licenses/by/2.0/",
                       "page": "https://www.flickr.com/photos/x/1", "title": "t"}}


def _base(tmp: Path) -> Path:
    base = tmp / "ppe4"
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        _image(base / "images" / split / f"{split}.jpg", seed)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.6\n")
    (base / "lists").mkdir()
    (base / "lists" / "test_chv.txt").write_text(str(base / "images" / "test" / "test.jpg") + "\n")
    cfg = {"path": str(base), "train": "images/train", "val": "images/val", "test": "images/test",
           "test_chv": "lists/test_chv.txt", "names": ["person", "helmet", "vest"]}
    (base / "data.yaml").write_text(yaml.safe_dump(cfg))
    return base


def test_labels_are_written_in_yolo_format(tmp_path):
    caps.write_label_file(tmp_path / "a.txt", [[0, 0.2, 0.4, 0.6, 1.2], [2, 0.3, 0.5, 0.3, 0.6]])
    assert (tmp_path / "a.txt").read_text() == "0 0.400000 0.700000 0.400000 0.600000\n"    # clipped; empty box dropped


def test_photos_are_downloaded_and_a_changed_one_is_refused(tmp_path):
    src = tmp_path / "src"
    good = _image(src / "good.jpg", 5)
    _image(src / "changed.jpg", 6)
    manifest = {"images": [_entry("good", "train", good, (src / "good.jpg").as_uri()),
                           _entry("changed", "train", good, (src / "changed.jpg").as_uri())]}   # its checksum is good's
    raw = tmp_path / "raw"
    failed = caps.fetch(manifest, raw, workers=2, log=lambda *a: None)
    assert (raw / "images" / "good.jpg").read_bytes() == good
    assert "checksum" in failed["changed"] and not (raw / "images" / "changed.jpg").exists()
    assert (raw / "labels" / "good.txt").read_text().count("\n") == 2
    assert caps.fetch(manifest, raw, workers=2, log=lambda *a: None).keys() == {"changed"}   # the good one isn't fetched again


def test_the_dataset_adds_training_and_validation_photos_and_keeps_test_photos_apart(tmp_path):
    base = _base(tmp_path)
    raw = tmp_path / "raw"
    entries = []
    for i, split, seed in (("c_train", "train", 11), ("c_val", "val", 12), ("c_test", "test", 13), ("c_twin", "train", 3)):
        data = _image(raw / "images" / f"{i}.jpg", seed)          # c_twin is the base test photo again
        entries.append(_entry(i, split, data, ""))
    manifest = {"images": entries}
    out = tmp_path / "ppe4caps"
    counts = caps.build(base / "data.yaml", manifest, out, raw, log=lambda *a: None)
    cfg = yaml.safe_load((out / "data.yaml").read_text())

    def listed(key):
        return [Path(p).name for p in (out / cfg[key]).read_text().split()]
    assert listed("train") == ["train.jpg", "c_train.jpg"]
    assert listed("val") == ["val.jpg", "c_val.jpg"]
    assert listed("test") == ["test.jpg"] and listed("test_chv") == ["test.jpg"]     # the base's tests, unchanged
    assert listed("test_caps") == ["c_test.jpg"]
    assert counts["left_out"]["like a base validation or test photo"] == 1           # the twin of a test photo
    assert [im["id"] for im in caps.test_hats(manifest, out / "data.yaml")] == ["c_test"]
    credits = (out / "credits.csv").read_text().splitlines()
    assert credits[0].startswith("file,split,author") and len(credits) == 4 and "c_twin" not in "".join(credits)


def test_hat_wearers_are_known_not_to_wear_a_helmet(tmp_path):
    raw = tmp_path / "raw"
    data = _image(raw / "images" / "x.jpg", 7)
    im = _entry("x", "test", data, "")
    (case,) = caps.cases([im], raw)
    assert [t.helmet for t in case.truths] == [False]                     # only the person with the hat
    assert tuple(case.truths[0].box) == tuple(im["hats"][0]["person"]) and len(case.gt_boxes["person"]) == 2
    assert case.input_height == 480.0


def test_a_helmet_box_on_a_hat_counts_as_calling_it_a_helmet():
    im = _entry("x", "test", b"", "")
    on = Det(1, (0.40, 0.16, 0.50, 0.26), 0.4)
    beside = Det(1, (0.70, 0.30, 0.78, 0.38), 0.9)
    assert caps.hats_called_helmets([im], [[on]], 0.25)["called_helmet"] == 1
    assert caps.hats_called_helmets([im], [[on]], 0.5)["called_helmet"] == 0          # below the threshold
    assert caps.hats_called_helmets([im], [[beside]], 0.25)["called_helmet"] == 0     # not on the hat


def _select():
    spec = importlib.util.spec_from_file_location("select_caps", ROOT / "scripts" / "select_caps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_photographer_always_lands_in_one_split():
    sel = _select()
    authors = [f"https://www.flickr.com/people/u{k}/" for k in range(4000)]
    splits = [sel.split_of(a) for a in authors]
    assert splits == [sel.split_of(a) for a in authors]
    share = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert abs(share["train"] - 0.7) < 0.03 and abs(share["val"] - 0.1) < 0.02 and abs(share["test"] - 0.2) < 0.03


def test_one_hat_with_several_classes_counts_once():
    sel = _select()
    hats = sel.dedupe([{"box": [0.1, 0.1, 0.3, 0.2], "kinds": {"hat"}}, {"box": [0.1, 0.1, 0.31, 0.2], "kinds": {"fedora"}},
                       {"box": [0.6, 0.1, 0.8, 0.2], "kinds": {"hat"}}], 0.5)
    assert len(hats) == 2 and hats[0]["kinds"] == {"hat", "fedora"}


def test_the_manifest_in_the_project_is_consistent():
    m = json.loads(caps.MANIFEST.read_text())
    ids = [im["id"] for im in m["images"]]
    assert len(ids) == len(set(ids)) and m["counts"] == {s: sum(im["split"] == s for im in m["images"]) for s in m["counts"]}
    for im in m["images"]:
        assert im["url"].startswith("https://open-images-dataset.s3.amazonaws.com/") and len(im["sha256"]) == 64
        assert im["hats"] and all(lab[0] in (0, 2) for lab in im["labels"])       # people and vests only: no helmet
        people = {tuple(lab[1:]) for lab in im["labels"] if lab[0] == 0}
        assert all(tuple(h["person"]) in people for h in im["hats"])
        assert im["credit"]["licence"].startswith("https://creativecommons.org/licenses/by/")
    by_author = {}
    for im in m["images"]:
        by_author.setdefault(im["credit"]["author"] + im["credit"]["page"].split("/photos/")[-1].split("/")[0], set()).add(im["split"])
    assert all(len(s) == 1 for s in by_author.values())                      # one photographer, one split


def test_a_failed_check_is_accepted_only_with_a_persons_reason():
    spec = importlib.util.spec_from_file_location("compare_models", ROOT / "scripts" / "compare_models.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    checks = [("a", True, ""), ("b", False, ""), ("c", True, "")]
    assert cm.verdict(checks, None) == (False, ["b"])
    assert cm.verdict(checks, "  ") == (False, ["b"])
    assert cm.verdict(checks, "looked at: 5 small helmets lost of 350") == (True, ["b"])
    assert cm.verdict([("a", True, "")], None) == (True, [])


def test_the_everyday_script_lists_its_commands_and_refuses_unknown_ones():
    import subprocess
    r = subprocess.run(["bash", str(ROOT / "scripts" / "ppe.sh"), "help"], capture_output=True, text=True)
    assert r.returncode == 0 and "ppe.sh start" in r.stdout and "ppe.sh photo" in r.stdout
    r = subprocess.run(["bash", str(ROOT / "scripts" / "ppe.sh"), "no-such-command"], capture_output=True, text=True)
    assert r.returncode == 2 and "Unknown command" in r.stdout
