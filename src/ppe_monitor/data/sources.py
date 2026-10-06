"""The public datasets the detector is trained on, and downloading them safely.

Each source is pinned by SHA-256, so a changed or corrupted download is refused rather than
silently producing a different dataset.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    url: str
    sha256: str
    size_mb: float
    licence: str
    attribution: str
    fmt: str  # "yolo", "coco", "chv", "voc" or "sh17" (readers.py)
    # source split folder -> our split name
    splits: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # Added for the extra datasets of the fine-tuned model (extra_sources.py); defaults keep the
    # Phase 1 sources exactly as they were.
    archive: str = ""            # "zip" or "tar.gz"; "" = from the URL's ending
    marker: str = ""             # a file or folder whose parent is the dataset root; "" = by format
    persons: str = "labelled"    # "labelled", or "pseudo": the dataset has no person boxes (audit.py adds them)
    audit: dict = field(default_factory=dict)   # {class: confidence}: drop images where the current
                                                # model sees that class this surely and no label has it


SOURCES: dict[str, Source] = {
    "construction-ppe": Source(
        name="construction-ppe",
        title="Construction-PPE (Ultralytics)",
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
        sha256="bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc",
        size_mb=178.4,
        licence="AGPL-3.0",
        attribution="Construction-PPE dataset by Ultralytics, https://docs.ultralytics.com/datasets/detect/construction-ppe/",
        fmt="yolo",
        splits={"train": "train", "val": "val", "test": "test"},
        notes="Mostly close-up photos; includes non-site scenes of people without PPE. Many images "
              "come from one staged photo session (same person and background, different PPE).",
    ),
    "rf100-construction-safety": Source(
        name="rf100-construction-safety",
        title="Construction Safety (Roboflow 100)",
        url="https://huggingface.co/datasets/Francesco/construction-safety-gsnvb/resolve/main/dataset.tar.gz",
        sha256="70e7b0c3f597f8e6b45e41fb22259de96feba8d763a93305a273384b13842763",
        size_mb=75.2,
        licence="CC BY 4.0",
        attribution="construction safety dataset, Roboflow Universe (Roboflow 100 benchmark), CC BY 4.0, "
                    "https://universe.roboflow.com/object-detection/construction-safety-gsnvb",
        fmt="coco",
        splits={"train": "train", "valid": "val", "test": "test"},
        notes="Construction and outdoor work scenes, people smaller in frame. All images were "
              "stretched to 640x640 when exported, so proportions are distorted.",
    ),
}


class SourceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, size_mb: float) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ppe-monitor"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or size_mb * 1e6)
        done, next_report = 0, 0.1
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done / total >= next_report:
                print(f"    {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                next_report += 0.25
    tmp.replace(dest)


def _safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise SourceError(f"refusing to extract {member!r}: it points outside {dest}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")  # the "data" filter blocks path tricks and device files


def dataset_root(source: Source, extracted: Path) -> Path:
    """The folder inside the extracted archive that holds the splits."""
    marker = source.marker or ("data.yaml" if source.fmt == "yolo" else "_annotations.coco.json")
    hits = sorted((h for h in extracted.rglob(marker) if "__MACOSX" not in h.parts), key=lambda p: len(p.parts))
    if not hits:
        raise SourceError(f"{source.name}: no {marker} found under {extracted}")
    return hits[0].parent.parent if source.fmt == "coco" and not source.marker else hits[0].parent


def fetch(source: Source, raw_dir: Path = RAW_DIR) -> Path:
    """Download (if needed), verify and extract a source. Returns the dataset root folder."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = raw_dir / source.name
    if (extracted / ".complete").is_file():
        return dataset_root(source, extracted)

    kind = source.archive or ("zip" if source.url.endswith(".zip") else "tar.gz")
    archive = raw_dir / f"{source.name}.{kind}"
    if not archive.is_file() or _sha256(archive) != source.sha256:
        print(f"  downloading {source.title} ({source.size_mb:.0f} MB)")
        _download(source.url, archive, source.size_mb)
    actual = _sha256(archive)
    if actual != source.sha256:
        raise SourceError(f"{source.name}: checksum mismatch (expected {source.sha256[:12]}..., got {actual[:12]}...). "
                          "The download is corrupted or the file changed upstream; not using it.")
    print(f"  {source.name}: checksum verified, extracting")
    if extracted.exists():
        shutil.rmtree(extracted)
    _safe_extract(archive, extracted)
    (extracted / ".complete").write_text(source.sha256 + "\n", encoding="utf-8")
    return dataset_root(source, extracted)
