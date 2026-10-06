"""Running the YOLO detector and reading datasets for evaluation.

Ultralytics collects anonymous usage analytics by default; `load_model` switches that off for
this machine (settings key `sync`).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT
from .matching import Det

# Some PyTorch operations aren't implemented on Apple's GPU yet; this lets them fall back to the CPU
# instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(weights: str | Path):
    from ultralytics import YOLO, settings

    if settings.get("sync", False):
        settings.update({"sync": False})
    weights = Path(weights)
    if not weights.exists() and weights.parent == Path("."):
        # a bare pretrained name like "yolo26n.pt": keep downloads in models/pretrained/
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        weights = PRETRAINED_DIR / weights.name
    return YOLO(str(weights))


def split_items(data_yaml: str | Path, split: str) -> list[tuple[Path, Path]]:
    """(image, label file) pairs of one split of a YOLO dataset."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path") or Path(data_yaml).parent)
    where = root / data[split]
    if where.suffix == ".txt":                 # a list of images, one path (relative to the root) per line
        images = [root / line for line in where.read_text(encoding="utf-8").split()]
    else:
        images = sorted(p for p in where.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [(p, label_path(p)) for p in images]


def label_path(image: Path) -> Path:
    """YOLO convention (same rule as Ultralytics): swap the LAST /images/ in the path for /labels/."""
    a, b = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(b.join(str(image).rsplit(a, 1))).with_suffix(".txt")


def class_names(data_yaml: str | Path) -> list[str]:
    names = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))["names"]
    return list(names) if isinstance(names, list) else [names[k] for k in sorted(names)]


def read_labels(label_file: Path) -> list[Det]:
    """A YOLO label file as Det boxes in (x1, y1, x2, y2) fractions."""
    if not label_file.is_file():
        return []
    dets = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            c, x, y, w, h = int(float(p[0])), *map(float, p[1:5])
            dets.append(Det(c, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return dets


def predict(model, images: list[Path], *, conf: float = 0.001, imgsz: int = 640, device: str | None = None,
            batch: int | None = None) -> list[list[Det]]:
    """Predictions for each image, keeping everything above `conf` (low by default, so precision-recall
    curves can be computed). A model loaded through vision/backends.py carries its own batch size
    (1 for Core ML), device and precision; those win."""
    batch = batch or getattr(model, "ppe_batch", 16)
    extra = dict(getattr(model, "ppe_predict_args", {}))
    device = extra.pop("device", device)
    out = []
    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        for r in model.predict(chunk if batch > 1 else chunk[0], conf=conf, imgsz=imgsz, device=device,
                               verbose=False, max_det=300, **extra):
            b = r.boxes
            out.append([Det(int(c), tuple(float(v) for v in xy), float(s))
                        for c, xy, s in zip(b.cls.tolist(), b.xyxyn.tolist(), b.conf.tolist())])
    return out
