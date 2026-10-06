"""Merging the source datasets into one YOLO dataset with our three classes.

Steps, each recorded in the report:
  1. read every source and rename its labels via configs/classes.yaml; an unknown label stops
     the build, so every label is a deliberate decision
  2. fingerprint every image and group near-duplicates (same photo, or frames of the same shoot)
  3. a group that spans splits is moved entirely into train: a test image whose twin is in
     training measures memory, not detection (book, Chapter 18)
  4. the same web photo appearing in two datasets is kept once
  5. write images, labels, data.yaml, and report.md / report.json
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .dedup import dhash, groups_from_pairs, near_duplicate_pairs
from .labels import ClassMap
from .readers import Sample, read_source
from .sources import Source

SPLITS = ("train", "val", "test")
# Box size buckets at the 640-pixel training resolution (the COCO convention):
# small < 32x32 px, medium < 96x96 px, large otherwise.
SMALL, MEDIUM = (32 / 640) ** 2, (96 / 640) ** 2


def size_bucket(area_fraction: float) -> str:
    return "small" if area_fraction < SMALL else "medium" if area_fraction < MEDIUM else "large"


class UnknownLabels(ValueError):
    pass


@dataclass
class BuildReport:
    out_dir: str
    classes: list[str]
    sources: list[dict] = field(default_factory=list)
    label_decisions: dict[str, dict[str, dict]] = field(default_factory=dict)
    images: dict[str, int] = field(default_factory=dict)
    boxes: dict[str, dict[str, int]] = field(default_factory=dict)
    sizes: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    moved_to_train: dict[str, int] = field(default_factory=dict)
    largest_moved_groups: list[int] = field(default_factory=list)
    exact_duplicates_dropped: int = 0
    near_duplicate_pairs: int = 0
    images_without_boxes: dict[str, int] = field(default_factory=dict)
    source_stats: dict[str, dict] = field(default_factory=dict)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # a hard link costs no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def build_dataset(
    sources: list[tuple[Source, Path]],
    out_dir: Path,
    class_map: ClassMap,
    *,
    max_distance: int = 6,
    exact_distance: int = 2,
    log=print,
) -> BuildReport:
    report = BuildReport(out_dir=str(out_dir), classes=class_map.classes)

    # 1. read and map labels
    samples: list[Sample] = []
    unknown: dict[str, Counter] = {}
    for source, root in sources:
        s, stats = read_source(source.fmt, root, source.name, source.splits)
        log(f"  read {source.name}: {stats.images} images, {sum(stats.labels.values())} boxes")
        report.sources.append({"name": source.name, "title": source.title, "licence": source.licence,
                               "attribution": source.attribution, "sha256": source.sha256, "notes": source.notes})
        report.source_stats[source.name] = {
            "images": stats.images, "images_without_label_file": stats.images_without_label_file,
            "orphan_label_files": stats.orphan_label_files, "degenerate_boxes_dropped": stats.degenerate_boxes}
        decisions = {}
        for label, count in sorted(stats.labels.items(), key=lambda kv: -kv[1]):
            d = class_map.decide(label)
            decisions[label] = {"count": count, "decision": d.kind, "class": d.class_name, "via": d.via}
            if d.kind == "unknown":
                unknown.setdefault(source.name, Counter())[label] = count
        report.label_decisions[source.name] = decisions
        samples.extend(s)
    if unknown:
        lines = [f"  {src}: " + ", ".join(f"{lbl!r} ({n} boxes)" for lbl, n in c.items()) for src, c in unknown.items()]
        raise UnknownLabels("These labels are neither mapped nor ignored in configs/classes.yaml:\n" + "\n".join(lines) +
                            "\nAdd each one under `aliases` (keep it) or `ignore` (drop it), then build again.")

    # 2. fingerprints and near-duplicate groups
    log(f"  fingerprinting {len(samples)} images ...")
    hashes = [dhash(s.image) for s in samples]
    pairs = near_duplicate_pairs(hashes, max_distance)
    report.near_duplicate_pairs = len(pairs)
    groups = groups_from_pairs(len(samples), pairs)

    # 3. a group spanning more than one split goes entirely to train
    splits_of_group: dict[int, set] = defaultdict(set)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        splits_of_group[g].add(samples[i].split)
        members[g].append(i)
    final_split = [s.split for s in samples]
    moved = Counter()
    moved_group_sizes = []
    for g, splits in splits_of_group.items():
        if len(splits) > 1:
            moved_group_sizes.append(len(members[g]))
            for i in members[g]:
                if final_split[i] != "train":
                    moved[final_split[i]] += 1
                    final_split[i] = "train"
    report.moved_to_train = dict(moved)
    report.largest_moved_groups = sorted(moved_group_sizes, reverse=True)[:5]

    # 4. the same photo in two different sources, now in the same split: keep the first one.
    # (Within one source, near-identical fingerprints are usually different shots of the same
    # scene - e.g. the same pose with a different helmet colour - so those are all kept.)
    drop = set()
    for i, j, d in pairs:
        if (d <= exact_distance and samples[i].source != samples[j].source
                and final_split[i] == final_split[j] and i not in drop):
            drop.add(j)
    report.exact_duplicates_dropped = len(drop)

    # 5. write the merged dataset
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
    images, boxes = Counter(), {s: Counter() for s in SPLITS}
    sizes = {s: {c: Counter() for c in class_map.classes} for s in SPLITS}
    empty = Counter()
    for i, sample in enumerate(samples):
        if i in drop:
            continue
        split = final_split[i]
        stem = f"{sample.source}__{sample.split}__{sample.image.stem}"
        _link_or_copy(sample.image, out_dir / "images" / split / f"{stem}{sample.image.suffix.lower()}")
        lines = []
        for b in sample.boxes:
            d = class_map.decide(b.label)
            if d.kind != "class":
                continue
            cx, cy, w, h = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            boxes[split][d.class_name] += 1
            sizes[split][d.class_name][size_bucket(b.area)] += 1
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        images[split] += 1
        if not lines:
            empty[split] += 1

    report.images = {s: images[s] for s in SPLITS}
    report.boxes = {s: {c: boxes[s][c] for c in class_map.classes} for s in SPLITS}
    report.sizes = {s: {c: dict(sizes[s][c]) for c in class_map.classes} for s in SPLITS}
    report.images_without_boxes = {s: empty[s] for s in SPLITS}

    data_yaml = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": list(class_map.classes)}
    (out_dir / "data.yaml").write_text("# Written by scripts/prepare_data.py - rebuild rather than edit.\n"
                                       + yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r: BuildReport) -> str:
    out = ["# Training dataset report", "",
           f"Merged dataset: `{r.out_dir}` - classes {', '.join(r.classes)}.", "", "## Sources", ""]
    for s in r.sources:
        st = r.source_stats[s["name"]]
        out += [f"- **{s['title']}** - licence {s['licence']}. {s['attribution']}",
                f"  {st['images']} images; {st['images_without_label_file']} without a label file, "
                f"{st['orphan_label_files']} label files without an image, {st['degenerate_boxes_dropped']} zero-size boxes dropped.",
                f"  {s['notes']}"]
    out += ["", "## How each label was treated", "", "| Source | Label | Boxes | Decision |", "|---|---|---|---|"]
    for src, decisions in r.label_decisions.items():
        for label, d in decisions.items():
            what = f"-> **{d['class']}**" if d["decision"] == "class" else "dropped (ignore list)"
            out.append(f"| {src} | {label} | {d['count']} | {what} |")
    total_moved = sum(r.moved_to_train.values())
    out += ["", "## Keeping the test honest", "",
            f"{r.near_duplicate_pairs} near-duplicate image pairs found. Groups spanning more than one split were "
            f"moved into train: {total_moved} images "
            f"({', '.join(f'{n} from {s}' for s, n in r.moved_to_train.items()) or 'none'}). "
            f"Largest moved groups: {r.largest_moved_groups}. "
            f"{r.exact_duplicates_dropped} photos that appear in both sources were kept only once.",
            "", "## Result", "", "| Split | Images | " + " | ".join(r.classes) + " | images with no boxes |",
            "|---|---|" + "---|" * len(r.classes) + "---|"]
    for s in SPLITS:
        out.append(f"| {s} | {r.images[s]} | " + " | ".join(str(r.boxes[s][c]) for c in r.classes)
                   + f" | {r.images_without_boxes[s]} |")
    out += ["", "Box sizes at 640 px (small < 32x32, medium < 96x96, large otherwise):", "",
            "| Split | Class | small | medium | large |", "|---|---|---|---|---|"]
    for s in SPLITS:
        for c in r.classes:
            z = r.sizes[s][c]
            out.append(f"| {s} | {c} | {z.get('small', 0)} | {z.get('medium', 0)} | {z.get('large', 0)} |")
    return "\n".join(out) + "\n"
