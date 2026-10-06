"""Extra public datasets for the fine-tuned detector (datasets/ppe4).

The Phase 1 model (datasets/ppe3) misses navy, brown and white helmets in an indoor workshop
(docs/real_clips.md). These three datasets add helmets of every colour and indoor industrial
scenes. They are kept apart from SOURCES in sources.py so that nothing built on ppe3 (the rule
checks, the benchmark) ever downloads them.

| Dataset | What it adds | Caveat |
|---|---|---|
| CHV | people, hi-vis vests, blue / red / white / yellow helmets | small (1,330 images) |
| GDUT-HWD | 18,893 helmets in four colours, bare heads, many small people | no person or vest boxes: people are added by a COCO model (audit.py), and images with a vest are left out |
| SH17 | indoor factories and workshops | 14 GB; only 528 images have a helmet or vest, and its vest labels are incomplete. Its labels come from the Kaggle archive; the images themselves from Pexels at 1280 px |
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import random
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

from .sources import RAW_DIR, Source, SourceError, fetch

USER_AGENT = "ppe-monitor"          # Pexels refuses Python's default user agent

EXTRA_SOURCES: dict[str, Source] = {
    "chv": Source(
        name="chv",
        title="CHV: Colour Helmet and Vest (Wang et al., 2021)",
        url="https://drive.usercontent.google.com/download?id=1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a&export=download&confirm=t",
        sha256="e2a2ebef7b9a69fd2d7f5152eb808b14a3a0a76de015c802f3f187c437a8e577",
        size_mb=439.7,
        licence="free to use, with citation (no formal licence)",
        attribution="Z. Wang, Y. Wu, L. Yang, A. Thirunavukarasu, C. Evison, Y. Zhao, \"Fast personal protective "
                    "equipment detection for real construction sites using deep learning approaches\", Sensors 21(10), "
                    "2021. https://github.com/ZijianWang-ZW/PPE_detection",
        fmt="chv",
        splits={"train": "train", "valid": "val", "test": "test"},
        notes="Construction sites; person, vest and helmets in four colours (blue, red, white, yellow).",
        archive="zip",
        marker="data split",
        audit={"person": 0.6, "helmet": 0.5, "vest": 0.5},
    ),
    "gdut-hwd": Source(
        name="gdut-hwd",
        title="GDUT-HWD: Hardhat Wearing Detection (Wu et al., 2019)",
        url="https://huggingface.co/datasets/worstprogrammer/GDUT-HWD/resolve/main/GDUT-HWD.zip",
        sha256="7eebc70964e7139384fcd80b7d7201e44d6741b99be4d72d8bed2dd77b637b67",
        size_mb=678.1,
        licence="no licence stated (published for research)",
        attribution="J. Wu, N. Cai, W. Chen, H. Wang, G. Wang, \"Automatic detection of hardhats worn by construction "
                    "personnel: A deep learning approach and benchmark dataset\", Automation in Construction 106, 2019. "
                    "https://github.com/wujixiu/helmet-detection (copy used: Hugging Face worstprogrammer/GDUT-HWD)",
        fmt="voc",
        splits={"trainval": "train", "test": "test"},
        notes="Construction sites, many small and distant people; helmets labelled by colour (blue, red, white, "
              "yellow) and bare heads. No person or vest boxes.",
        archive="zip",
        marker="Annotations",
        persons="pseudo",
        audit={"helmet": 0.5, "vest": 0.3},
    ),
    "sh17": Source(
        name="sh17",
        title="SH17: Human Safety and PPE in Manufacturing (Ahmad & Rahimi, 2024)",
        url="https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection",
        sha256="35a517dad8b669f388bc0fcde555f2c3d748ca3fa888a9068f626b0212b7bf17",   # of the label bundle, see fetch_sh17
        size_mb=0,
        licence="CC BY-NC-SA 4.0 (non-commercial); images under the Pexels licence",
        attribution="H. M. Ahmad, A. Rahimi, \"SH17: A dataset for human safety and personal protective equipment "
                    "detection in manufacturing industry\", Journal of Safety Science and Resilience, 2024. "
                    "https://github.com/ahmadmughees/SH17dataset. Photos from Pexels (https://www.pexels.com).",
        fmt="sh17",
        splits={"train": "train", "val": "val+test"},
        notes="Factories, workshops and other industrial scenes (Pexels photos). Used: every image with a helmet "
              "or vest, plus a sample of images of people without either.",
        audit={"vest": 0.5},
    ),
}

SH17_NAMES = ["person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot", "tool", "glasses", "gloves",
              "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest"]
SH17_KEEP_IDS = {SH17_NAMES.index("helmet"), SH17_NAMES.index("safety-vest")}


# -- reading parts of a remote zip file, without downloading all 14 GB -------------------------------
class _HTTPRange(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (enough for zipfile to read the index)."""

    def __init__(self, url: str):
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.url = r.geturl()                                    # Kaggle redirects to signed storage
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read_range(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        data = self.read_range(self.pos, min(self.size, self.pos + n) - 1)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _bundle_digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name] + b"\0")
    return h.hexdigest()


def _sh17_labels(source: Source, dest: Path) -> None:
    """The labels, metadata and split lists: the last ~6 MB of the Kaggle archive."""
    remote = _HTTPRange(source.url)
    infos = [i for i in zipfile.ZipFile(remote).infolist()
             if i.filename.startswith(("labels/", "meta-data/")) or "/" not in i.filename]
    lo = min(i.header_offset for i in infos)
    hi = max(i.header_offset + 30 + len(i.filename.encode()) + 1024 + i.compress_size for i in infos)
    blob = remote.read_range(lo, min(hi, remote.size - 1))
    files = {}
    for i in infos:
        p = i.header_offset - lo
        sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", blob[p:p + 30])
        if sig != 0x04034B50:
            raise SourceError(f"sh17: unexpected archive layout at {i.filename}")
        start = p + 30 + name_len + extra_len
        data = blob[start:start + i.compress_size]
        files[i.filename] = zlib.decompress(data, -15) if i.compress_type == zipfile.ZIP_DEFLATED else data
    digest = _bundle_digest(files)
    if digest != source.sha256:
        raise SourceError(f"sh17: label bundle checksum mismatch (expected {source.sha256[:12]}..., got {digest[:12]}...). "
                          "The dataset changed upstream; not using it.")
    for name, data in files.items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes(data)


def sh17_selection(root: Path, negatives: int, seed: int = 0) -> list[str]:
    """Stems to use: every image with a helmet or vest label, and `negatives` images of people with
    neither whose head is in view (a bare head at work: what the model must not call a helmet).
    Images of only hands or tools at work are not used."""
    positive, negative = [], []
    person, head = SH17_NAMES.index("person"), SH17_NAMES.index("head")
    for f in sorted((root / "labels").glob("*.txt")):
        ids = {int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()}
        if ids & SH17_KEEP_IDS:
            positive.append(f.stem)
        elif person in ids and head in ids:
            negative.append(f.stem)
    random.Random(seed).shuffle(negative)
    return positive + sorted(negative[:negatives])


def _download_image(url: str, dest: Path, tries: int = 4) -> str:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if data[:3] != b"\xff\xd8\xff":
                raise OSError("not a JPEG")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return ""
        except Exception as exc:                      # network hiccups: try again, then give up on this one
            error = str(exc)
            time.sleep(2 ** attempt)
    return error


def fetch_sh17(source: Source, raw_dir: Path = RAW_DIR, negatives: int = 700, long_side: int = 1280,
               workers: int = 4, log=print) -> Path:
    """Labels from the Kaggle archive (checked against a pinned checksum), images from Pexels at
    `long_side` pixels. Resumes where it stopped; a photo that can't be downloaded is left out."""
    root = raw_dir / source.name
    if not (root / ".labels-complete").is_file():
        log("  sh17: reading the labels from the Kaggle archive (about 6 MB of 14 GB)")
        _sh17_labels(source, root)
        (root / ".labels-complete").write_text(source.sha256 + "\n")
    stems = sh17_selection(root, negatives)
    (root / "images").mkdir(exist_ok=True)
    todo = []
    for stem in stems:
        dest = root / "images" / f"{stem}.jpg"
        if not dest.is_file():
            meta = json.loads((root / "meta-data" / f"{stem}.json").read_text())
            todo.append((meta["src"]["original"] + f"?auto=compress&cs=tinysrgb&w={long_side}&h={long_side}", dest))
    failed = {}
    if todo:
        log(f"  sh17: downloading {len(todo)} photos from Pexels at {long_side} px ({len(stems) - len(todo)} already here)")
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(_download_image, url, dest): dest for url, dest in todo}
            for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                if fut.result():
                    failed[futures[fut].stem] = fut.result()
                if k % 200 == 0 or k == len(todo):
                    log(f"    {k} / {len(todo)}")
    (root / "selection.json").write_text(json.dumps({"stems": stems, "negatives": negatives, "long_side": long_side,
                                                     "failed": failed}, indent=1))
    if failed:
        log(f"  sh17: {len(failed)} photos could not be downloaded and are left out")
    return root


def fetch_extra(source: Source, raw_dir: Path = RAW_DIR, **kw) -> Path:
    return fetch_sh17(source, raw_dir, **kw) if source.fmt == "sh17" else fetch(source, raw_dir)
