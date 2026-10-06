"""Hats and caps that are not helmets: training photos from Open Images (book, Chapters 7 and 23).

On real video from other sites (Phase 6), a man in a baseball cap was never alerted: the model took
his cap for a helmet. On Open Images photos of people in hats, the model in use calls about one hat
in four a helmet (bare heads: 4 %). Its training data has hard hats and bare heads, but almost no
other headwear. These photos add it.

**What they are.** Photos from Google's Open Images (V7; validation, test and training splits) in
which a person wears a hat: Open Images' "Hat" (which covers baseball caps), "Sun hat", "Fedora",
"Cowboy hat" and "Sombrero" boxes, drawn by people. Boxes: CC BY 4.0 (Google). Photos: listed as
CC BY 2.0 by their Flickr authors; each one's author and page are kept (`credit`).

**How they were chosen** (in the cloud, 28 Sep 2026; `data/caps/README.md`):
- a hat worn by a labelled person (its centre in the top of the person's box); no helmet of any
  kind labelled; no crowd box, no drawing or poster of a person; no hat that isn't worn;
- the current model, the COCO model and a person checked the labels:
  - a person the COCO model is sure of (>= 0.6) that Open Images didn't box (often someone in the
    background) is added as a label; one it is unsure of (0.35-0.6): the image is left out; so is
    an image where most labelled people aren't found (a photo stored turned, or wrong boxes);
  - a helmet the current model is sure of (>= 0.5) away from every hat: left out (maybe a real one);
  - a vest it is unsure of (0.25-0.5): left out; a vest it is sure of (>= 0.5) is labelled;
  - every hat the current model calls a helmet (>= 0.25) was looked at by eye, in sheets of crops.
    Real helmets labelled "Hat" (hard hats, pith, bicycle, rafting, jockey and police helmets, toy
    fire helmets) and unclear ones were left out, with their whole image.
- **Splits by photographer:** all photos by one Flickr author go to the same split, so no test
  photo shows a scene or person that training has seen.

**Labels in the project's three classes:** person (Open Images' own boxes, one per person, plus
the COCO model's sure ones above) and vest (the current model at >= 0.5). A hat gets no label at all: to the detector it is background,
which is the lesson. The test split's hats are kept apart (`hats`) to measure how often each model
still calls a hat a helmet.

The manifest (`data/caps/openimages_caps.json`) lists every photo with its download address,
SHA-256, labels, hats and credit. `fetch()` downloads the photos and refuses any whose checksum
differs; `build()` adds them to a copy of datasets/ppe4 (training and validation only; the test
photos are a separate `test_caps` list).
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from ..config import PROJECT_ROOT
from .sources import RAW_DIR

MANIFEST = PROJECT_ROOT / "data" / "caps" / "openimages_caps.json"
RAW = RAW_DIR / "openimages-caps"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str, tries: int = 4) -> str:
    """"" when `dest` holds the photo with the expected checksum, else why not."""
    error = ""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ppe-monitor"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha256:
                return "checksum differs from the manifest (the photo changed upstream): not used"
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return ""
        except Exception as exc:                        # network hiccups: try again, then give up on this one
            error = str(exc)
            time.sleep(2 ** attempt)
    return error


def fetch(manifest: dict, raw: Path = RAW, workers: int = 12, log=print) -> dict:
    """Download every photo of the manifest not already here (with the right checksum) into
    raw/images, and write its label file into raw/labels. Returns {id: why} for photos that
    couldn't be had; they are left out."""
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "labels").mkdir(parents=True, exist_ok=True)
    todo, failed = [], {}
    for im in manifest["images"]:
        dest = raw / "images" / f"{im['id']}.jpg"
        if not (dest.is_file() and dest.stat().st_size == im["bytes"] and _sha256(dest) == im["sha256"]):
            todo.append(im)
    if todo:
        mb = sum(im["bytes"] for im in todo) / 1e6
        log(f"  downloading {len(todo)} photos ({mb:.0f} MB) from Open Images; "
            f"{len(manifest['images']) - len(todo)} already here")
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(_download, im["url"], raw / "images" / f"{im['id']}.jpg", im["sha256"]): im["id"]
                       for im in todo}
            for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                if fut.result():
                    failed[futures[fut]] = fut.result()
                if k % 250 == 0 or k == len(todo):
                    log(f"    {k} / {len(todo)}")
    for im in manifest["images"]:
        write_label_file(raw / "labels" / f"{im['id']}.txt", im["labels"])
    if failed:
        log(f"  {len(failed)} photos could not be had and are left out")
    return failed


def write_label_file(path: Path, labels: list) -> None:
    """YOLO format from [cls, x1, y1, x2, y2] (0-1 fractions)."""
    lines = []
    for cls, x1, y1, x2, y2 in labels:
        x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
        if x2 - x1 > 1e-4 and y2 - y1 > 1e-4:
            lines.append(f"{int(cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _files(spec: str, root: Path) -> list[str]:
    """The images of a data.yaml entry: a folder, or a text file listing them."""
    p = Path(spec) if Path(spec).is_absolute() else root / spec
    if p.is_dir():
        return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES)
    return [str(x if Path(x).is_absolute() else root / x) for x in
            (line.strip() for line in p.read_text(encoding="utf-8").splitlines()) if x]


def build(base_yaml: Path, manifest: dict, out_dir: Path, raw: Path = RAW, skip: set[str] = frozenset(),
          max_distance: int = 6, log=print) -> dict:
    """out_dir/data.yaml: the base dataset's splits, plus the manifest's photos, each in its own split:
    `train` photos added to training, `val` photos to validation, `test` photos in a separate
    `test_caps` list. The base's own test lists are kept exactly, so old and new models are compared
    on the same photos.

    A photo that looks like any base validation or test photo (near-duplicate, dhash distance <=
    max_distance) is left out, and so is a caps test photo that looks like any training photo:
    no test photo may have a twin in training. `skip`: ids left out by hand. Returns counts."""
    import yaml

    from .dedup import dhash

    base = yaml.safe_load(Path(base_yaml).read_text(encoding="utf-8"))
    root = Path(base["path"])
    if not root.is_absolute():
        root = (Path(base_yaml).parent / root).resolve()
    splits = {k: _files(v, root) for k, v in base.items() if k not in ("path", "names", "nc")}

    have = [im for im in manifest["images"] if im["id"] not in skip and (raw / "images" / f"{im['id']}.jpg").is_file()]
    log(f"  checking {len(have)} photos for near-duplicates of the base dataset's photos ...")
    import numpy as np

    def hashes(paths):
        return np.array([dhash(p) for p in paths], dtype=np.uint64)

    held_out = hashes([p for k, v in splits.items() if k != "train" for p in v])
    train_h = hashes(splits["train"])
    mine = hashes([raw / "images" / f"{im['id']}.jpg" for im in have])

    def near(h, pool):
        return bool(len(pool)) and int(np.bitwise_count(np.bitwise_xor(pool, h)).min()) <= max_distance

    caps_train_h = np.array([h for h, im in zip(mine, have) if im["split"] == "train"], dtype=np.uint64)
    left_out = {"like a base validation or test photo": 0, "a test photo like a training photo": 0}
    use = {"train": [], "val": [], "test": []}
    for h, im in zip(mine, have):
        if near(h, held_out):
            left_out["like a base validation or test photo"] += 1
            continue
        if im["split"] != "train" and (near(h, train_h) or near(h, caps_train_h)):
            left_out["a test photo like a training photo"] += 1
            continue
        use[im["split"]].append(str((raw / "images" / f"{im['id']}.jpg").resolve()))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lists").mkdir(exist_ok=True)
    cfg = {"path": str(out_dir.resolve())}
    for key, files in splits.items():
        add = use["train"] if key == "train" else use["val"] if key == "val" else []
        name = f"lists/{key}.txt"
        (out_dir / name).write_text("\n".join(files + add) + "\n", encoding="utf-8")
        cfg[key] = name
    (out_dir / "lists" / "test_caps.txt").write_text("\n".join(use["test"]) + "\n", encoding="utf-8")
    cfg["test_caps"] = "lists/test_caps.txt"
    cfg["names"] = base["names"]
    header = (f"# Written by scripts/prepare_caps_data.py: {base_yaml} plus Open Images photos of people in hats\n"
              f"# (data/caps/). Rebuild rather than edit.\n")
    (out_dir / "data.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    used = {Path(p).stem for v in use.values() for p in v}
    with open(out_dir / "credits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "author", "licence", "page", "title"])
        for im in manifest["images"]:
            if im["id"] in used:
                c = im["credit"]
                w.writerow([f"{im['id']}.jpg", im["split"], c["author"], c["licence"], c["page"], c.get("title", "")])
    counts = {"base": {k: len(v) for k, v in splits.items()}, "added": {k: len(v) for k, v in use.items()},
              "left_out": left_out, "skipped": len(skip)}
    (out_dir / "caps.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return counts


def test_hats(manifest: dict, data_yaml: Path) -> list[dict]:
    """The manifest's test photos that made it into the dataset's test_caps list, with their hats."""
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    listed = {Path(p).stem for p in _files(cfg["test_caps"], Path(cfg["path"]))}
    return [im for im in manifest["images"] if im["id"] in listed]


def cases(images: list[dict], raw: Path = RAW) -> list:
    """Rule-check cases (rules/check.py): every person wearing a hat is known NOT to wear a helmet."""
    import cv2

    from ..rules.check import Case
    from .compliance import PersonTruth

    out = []
    for im in images:
        path = raw / "images" / f"{im['id']}.jpg"
        h, w = cv2.imread(str(path)).shape[:2]
        people = [tuple(lab[1:]) for lab in im["labels"] if lab[0] == 0]
        wearers = {tuple(hat["person"]) for hat in im["hats"]}
        truths = [PersonTruth(p, helmet=False, vest=None) for p in people if p in wearers]
        if truths:
            out.append(Case(path, 640.0 * h / max(h, w), truths, {"person": people}, aspect=w / h))
    return out


def hats_called_helmets(images: list[dict], detections: list, threshold: float) -> dict:
    """How many test hats have a helmet detection (at `threshold`) on them: its centre inside the hat
    box, or IoU > 0.3. `detections`: per image, a list of vision.matching.Det."""
    hats = called = 0
    for im, dets in zip(images, detections):
        helmets = [d.box for d in dets if d.cls == 1 and d.score >= threshold]
        for hat in im["hats"]:
            hats += 1
            called += any(_on_hat(hat["box"], b) for b in helmets)
    return {"hats": hats, "called_helmet": called, "rate": called / hats if hats else float("nan")}


def _on_hat(hat, box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if hat[0] <= cx <= hat[2] and hat[1] <= cy <= hat[3]:
        return True
    ix = max(0.0, min(hat[2], box[2]) - max(hat[0], box[0]))
    iy = max(0.0, min(hat[3], box[3]) - max(hat[1], box[1]))
    inter = ix * iy
    union = (hat[2] - hat[0]) * (hat[3] - hat[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return union > 0 and inter / union > 0.3
