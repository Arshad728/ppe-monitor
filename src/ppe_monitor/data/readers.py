"""Reading YOLO- and COCO-format datasets into one common form.

Every box ends up as (label name, x1, y1, x2, y2) with coordinates as fractions of the image
size (0-1), whatever the source format was.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Sample:
    source: str
    split: str
    image: Path
    boxes: list[Box]


@dataclass
class ReadStats:
    images: int = 0
    images_without_label_file: int = 0
    orphan_label_files: int = 0
    degenerate_boxes: int = 0
    labels: Counter = field(default_factory=Counter)


def _clean(label: str, x1: float, y1: float, x2: float, y2: float, stats: ReadStats) -> Box | None:
    """Clip to the image and drop boxes with no area (they exist in real datasets)."""
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if (x2 - x1) * (y2 - y1) < 1e-6:
        stats.degenerate_boxes += 1
        return None
    stats.labels[label] += 1
    return Box(label, x1, y1, x2, y2)


def read_yolo(root: Path, source: str, splits: dict[str, str]) -> tuple[list[Sample], ReadStats]:
    """A YOLO dataset: data.yaml with class names, images/<split>/, labels/<split>/<stem>.txt
    holding `class x_center y_center width height` (all 0-1) per line."""
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    names = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}
    samples, stats = [], ReadStats()
    for src_split, split in splits.items():
        image_dir, label_dir = root / "images" / src_split, root / "labels" / src_split
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        stems = {p.stem for p in images}
        stats.orphan_label_files += sum(1 for p in label_dir.glob("*.txt") if p.stem not in stems)
        for image in images:
            stats.images += 1
            label_file = label_dir / f"{image.stem}.txt"
            boxes = []
            if not label_file.is_file():
                stats.images_without_label_file += 1
            else:
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
                    box = _clean(names[cls], x - w / 2, y - h / 2, x + w / 2, y + h / 2, stats)
                    if box:
                        boxes.append(box)
            samples.append(Sample(source, split, image, boxes))
    return samples, stats


def read_coco(root: Path, source: str, splits: dict[str, str]) -> tuple[list[Sample], ReadStats]:
    """A COCO dataset: <split>/_annotations.coco.json with pixel [x, y, width, height] boxes."""
    samples, stats = [], ReadStats()
    for src_split, split in splits.items():
        coco = json.loads((root / src_split / "_annotations.coco.json").read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for img in coco["images"]:
            path = root / src_split / img["file_name"]
            if not path.is_file():
                continue
            stats.images += 1
            w, h = img["width"], img["height"]
            boxes = []
            for ann in by_image.get(img["id"], []):
                x, y, bw, bh = ann["bbox"]
                box = _clean(names[ann["category_id"]], x / w, y / h, (x + bw) / w, (y + bh) / h, stats)
                if box:
                    boxes.append(box)
            samples.append(Sample(source, split, path, boxes))
    return samples, stats


def _stable_share(stem: str) -> float:
    """A number in [0, 1) fixed by the file name: the same image always lands in the same split."""
    return int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) / 16 ** 8


def _yolo_lines(label_file: Path, names: list[str], stats: ReadStats) -> list[Box]:
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
        box = _clean(names[cls], x - w / 2, y - h / 2, x + w / 2, y + h / 2, stats)
        if box:
            boxes.append(box)
    return boxes


CHV_NAMES = ["person", "vest", "blue helmet", "red helmet", "white helmet", "yellow helmet"]


def read_chv(root: Path, source: str, splits: dict[str, str]) -> tuple[list[Sample], ReadStats]:
    """CHV: images/, annotations/<stem>.txt (YOLO), and 'data split/<split>.txt' listing the images."""
    samples, stats = [], ReadStats()
    for src_split, split in splits.items():
        for line in (root / "data split" / f"{src_split}.txt").read_text(encoding="utf-8").split():
            image = root / "images" / Path(line).name
            if not image.is_file():
                continue
            stats.images += 1
            label_file = root / "annotations" / f"{image.stem}.txt"
            if not label_file.is_file():
                stats.images_without_label_file += 1
            boxes = _yolo_lines(label_file, CHV_NAMES, stats) if label_file.is_file() else []
            samples.append(Sample(source, split, image, boxes))
    return samples, stats


VOC_VAL_SHARE = 0.15       # share of GDUT-HWD's trainval kept apart for validation
VOC_TEST_TO_TRAIN = 0.5    # share of its (large, 50 %) test split used for training instead


def read_voc(root: Path, source: str, splits: dict[str, str]) -> tuple[list[Sample], ReadStats]:
    """Pascal VOC (GDUT-HWD): Annotations/<stem>.xml, JPEGImages/, ImageSets/Main/<split>.txt.
    Helmets are named by colour and become '<colour> helmet'; 'none' (a bare head) becomes 'bare head'.
    VOC_VAL_SHARE of the images our split calls train go to val instead, and VOC_TEST_TO_TRAIN of
    its test images go to train: half the dataset is more test than we need."""
    samples, stats = [], ReadStats()
    for src_split, split in splits.items():
        for stem in (root / "ImageSets" / "Main" / f"{src_split}.txt").read_text(encoding="utf-8").split():
            image = root / "JPEGImages" / f"{stem}.jpg"
            xml = root / "Annotations" / f"{stem}.xml"
            if not image.is_file():
                continue
            stats.images += 1
            boxes = []
            if not xml.is_file():
                stats.images_without_label_file += 1
            else:
                r = ET.parse(xml).getroot()
                w, h = float(r.find("size/width").text), float(r.find("size/height").text)
                for o in r.findall("object"):
                    name = o.find("name").text.strip().lower()
                    label = "bare head" if name == "none" else f"{name} helmet"
                    b = o.find("bndbox")
                    x1, y1, x2, y2 = (float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
                    box = _clean(label, x1 / w, y1 / h, x2 / w, y2 / h, stats)
                    if box:
                        boxes.append(box)
            share = _stable_share(stem)
            our = ("val" if share < VOC_VAL_SHARE else "train") if split == "train" else \
                  ("train" if share < VOC_TEST_TO_TRAIN else split) if split == "test" else split
            samples.append(Sample(source, our, image, boxes))
    return samples, stats


def read_sh17(root: Path, source: str, splits: dict[str, str]) -> tuple[list[Sample], ReadStats]:
    """SH17 as fetched by extra_sources.fetch_sh17: images/<stem>.jpg (the selected ones, resized),
    labels/<stem>.txt, meta-data/<stem>.json (original size), and train_files.txt / val_files.txt.
    Its val half goes to our val and half to our test ('val+test'). A photo whose shape doesn't match
    the original's (turned by its orientation tag) is left out: the boxes would not fit it."""
    from PIL import Image

    from .extra_sources import SH17_NAMES

    wanted = set(json.loads((root / "selection.json").read_text())["stems"])
    samples, stats = [], ReadStats()
    for src_split, split in splits.items():
        for name in (root / f"{src_split}_files.txt").read_text(encoding="utf-8").split():
            stem = Path(name).stem
            image = root / "images" / f"{stem}.jpg"
            if stem not in wanted or not image.is_file():
                continue
            meta = json.loads((root / "meta-data" / f"{stem}.json").read_text(encoding="utf-8"))
            with Image.open(image) as im:
                w, h = im.size
            if abs(w / h - meta["width"] / meta["height"]) > 0.02 * (meta["width"] / meta["height"]):
                stats.images_without_label_file += 1       # counted as unusable
                continue
            stats.images += 1
            boxes = _yolo_lines(root / "labels" / f"{stem}.txt", SH17_NAMES, stats)
            our = split if split != "val+test" else ("val" if _stable_share(stem) < 0.5 else "test")
            samples.append(Sample(source, our, image, boxes))
    return samples, stats


def read_source(fmt: str, root: Path, source: str, splits: dict[str, str]) -> tuple[list[Sample], ReadStats]:
    if fmt == "yolo":
        return read_yolo(root, source, splits)
    if fmt == "coco":
        return read_coco(root, source, splits)
    if fmt == "chv":
        return read_chv(root, source, splits)
    if fmt == "voc":
        return read_voc(root, source, splits)
    if fmt == "sh17":
        return read_sh17(root, source, splits)
    raise ValueError(f"unknown dataset format {fmt!r}")
