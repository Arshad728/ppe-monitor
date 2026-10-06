"""datasets/ppe4: the Phase 1 dataset plus the extra datasets, for fine-tuning.

ppe3 is taken exactly as built, split for split, so the old and the new model can be compared on
the same ppe3 test images, which neither has trained on. The extra images are added around it:

  1. near-duplicates are found across everything (ppe3 and the extra images);
  2. an extra image that is a near-duplicate of a ppe3 val or test image is left out: in training
     it would leak that test image, and in val or test it would count it twice;
  3. otherwise a group of near-duplicates that spans splits goes to train (as in build.py);
  4. the same photo in two extra datasets, in the same split, is kept once.

Besides train / val / test, lists/ holds each source's test images (test_ppe3, test_chv, ...) and
all the extra datasets' test images together (test_extra), so results can be given per source,
and helmet_colours.json the colour of every labelled helmet in the test splits of the datasets
that record it (CHV, GDUT-HWD).
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .audit import AuditStats
from .build import SPLITS, UnknownLabels, _link_or_copy, size_bucket
from .dedup import dhash, groups_from_pairs, near_duplicate_pairs
from .labels import ClassMap
from .readers import IMAGE_SUFFIXES, Box, ReadStats, Sample
from .sources import Source

COLOURS = ("blue", "red", "white", "yellow")


@dataclass
class ExtReport:
    out_dir: str
    base: str
    classes: list[str]
    sources: list[dict] = field(default_factory=list)
    label_decisions: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)            # split -> source -> images
    boxes: dict = field(default_factory=dict)             # split -> class -> boxes
    sizes: dict = field(default_factory=dict)
    left_out_near_ppe3_test: dict = field(default_factory=dict)
    moved_to_train: dict = field(default_factory=dict)
    exact_duplicates_dropped: int = 0
    near_duplicate_pairs: int = 0
    helmet_colours_test: dict = field(default_factory=dict)


def read_base(base_dir: Path) -> list[Sample]:
    """A built dataset (ppe3) as samples: source name from the file name, split from the folder."""
    names = yaml.safe_load((base_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = list(names) if isinstance(names, list) else [names[k] for k in sorted(names)]
    samples = []
    for split in SPLITS:
        for image in sorted((base_dir / "images" / split).iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            boxes = []
            label = base_dir / "labels" / split / f"{image.stem}.txt"
            for line in (label.read_text(encoding="utf-8").splitlines() if label.is_file() else []):
                c, x, y, w, h = line.split()[:5]
                x, y, w, h = map(float, (x, y, w, h))
                boxes.append(Box(names[int(c)], x - w / 2, y - h / 2, x + w / 2, y + h / 2))
            samples.append(Sample(image.stem.split("__")[0], split, image, boxes))
    return samples


def build_extended(base_dir: Path, extras: list[tuple[Source, list[Sample], ReadStats, AuditStats]], out_dir: Path,
                   class_map: ClassMap, *, max_distance: int = 6, exact_distance: int = 2, log=print) -> ExtReport:
    report = ExtReport(out_dir=str(out_dir), base=str(base_dir), classes=class_map.classes)
    base = read_base(base_dir)
    base_report = json.loads((base_dir / "report.json").read_text(encoding="utf-8"))
    report.sources = [dict(s, part="ppe3") for s in base_report["sources"]]
    log(f"  base {base_dir.name}: {len(base)} images")

    unknown = {}
    new: list[Sample] = []
    for source, samples, read_stats, audit_stats in extras:
        report.sources.append({"name": source.name, "title": source.title, "licence": source.licence,
                               "attribution": source.attribution, "sha256": source.sha256, "notes": source.notes,
                               "part": "extra"})
        decisions = {}
        for label, count in sorted(read_stats.labels.items(), key=lambda kv: -kv[1]):
            d = class_map.decide(label)
            decisions[label] = {"count": count, "decision": d.kind, "class": d.class_name}
            if d.kind == "unknown":
                unknown.setdefault(source.name, Counter())[label] = count
        report.label_decisions[source.name] = decisions
        report.audit[source.name] = {"read": read_stats.images, "checked": audit_stats.checked,
                                     "kept": audit_stats.kept, "people_added": audit_stats.pseudo_people,
                                     "dropped": dict(audit_stats.dropped),
                                     "dropped_images": audit_stats.dropped_images}
        new.extend(samples)
    if unknown:
        lines = [f"  {src}: " + ", ".join(f"{lbl!r} ({n})" for lbl, n in c.items()) for src, c in unknown.items()]
        raise UnknownLabels("These labels are neither mapped nor ignored in configs/classes.yaml:\n" + "\n".join(lines))

    samples = base + new
    is_base = [True] * len(base) + [False] * len(new)
    log(f"  fingerprinting {len(samples)} images ...")
    hashes = [dhash(s.image) for s in samples]
    pairs = near_duplicate_pairs(hashes, max_distance)
    report.near_duplicate_pairs = len(pairs)
    groups = groups_from_pairs(len(samples), pairs)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    split = [s.split for s in samples]
    drop = set()
    near_test, moved = Counter(), Counter()
    for g, idx in members.items():
        base_held_out = any(is_base[i] and split[i] in ("val", "test") for i in idx)
        if base_held_out:
            for i in idx:
                if not is_base[i]:
                    drop.add(i)
                    near_test[samples[i].source] += 1
            continue
        if len({split[i] for i in idx}) > 1:
            for i in idx:
                if not is_base[i] and split[i] != "train":
                    moved[f"{samples[i].source} {split[i]}"] += 1
                    split[i] = "train"
    exact = 0
    for i, j, d in pairs:
        if (d <= exact_distance and not is_base[i] and not is_base[j] and samples[i].source != samples[j].source
                and split[i] == split[j] and i not in drop and j not in drop):
            drop.add(j)
            exact += 1
    report.left_out_near_ppe3_test = dict(near_test)
    report.moved_to_train = dict(moved)
    report.exact_duplicates_dropped = exact

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in SPLITS:
        (out_dir / "images" / s).mkdir(parents=True)
        (out_dir / "labels" / s).mkdir(parents=True)
    (out_dir / "lists").mkdir()
    images = {s: Counter() for s in SPLITS}
    boxes = {s: Counter() for s in SPLITS}
    sizes = {s: {c: Counter() for c in class_map.classes} for s in SPLITS}
    test_lists = defaultdict(list)
    colours = {}
    colour_count = defaultdict(Counter)
    for i, s in enumerate(samples):
        if i in drop:
            continue
        sp = split[i]
        stem = s.image.stem if is_base[i] else f"{s.source}__{s.split}__{s.image.stem}"
        dst = out_dir / "images" / sp / f"{stem}{s.image.suffix.lower()}"
        _link_or_copy(s.image, dst)
        lines, cols = [], []
        for b in s.boxes:
            d = class_map.decide(b.label)
            if d.kind != "class":
                continue
            cx, cy, w, h = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            boxes[sp][d.class_name] += 1
            sizes[sp][d.class_name][size_bucket(b.area)] += 1
            colour = b.label.split()[0].lower()
            if d.class_name == "helmet" and colour in COLOURS:
                cols.append([colour, round(b.x1, 6), round(b.y1, 6), round(b.x2, 6), round(b.y2, 6)])
        (out_dir / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        images[sp][s.source] += 1
        if sp == "test":
            test_lists["ppe3" if is_base[i] else s.source].append(f"images/test/{dst.name}")
            if cols:
                colours[dst.name] = cols
                for c in cols:
                    colour_count[s.source][c[0]] += 1
    test_lists["extra"] = [x for name, items in test_lists.items() if name != "ppe3" for x in items]
    for name, items in test_lists.items():
        (out_dir / "lists" / f"test_{name}.txt").write_text("\n".join(sorted(items)) + "\n", encoding="utf-8")
    (out_dir / "helmet_colours.json").write_text(json.dumps(colours), encoding="utf-8")

    report.images = {s: dict(images[s]) for s in SPLITS}
    report.boxes = {s: {c: boxes[s][c] for c in class_map.classes} for s in SPLITS}
    report.sizes = {s: {c: dict(sizes[s][c]) for c in class_map.classes} for s in SPLITS}
    report.helmet_colours_test = {k: dict(v) for k, v in colour_count.items()}
    data_yaml = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                 **{f"test_{name}": f"lists/test_{name}.txt" for name in sorted(test_lists)},
                 "names": list(class_map.classes)}
    (out_dir / "data.yaml").write_text("# Written by scripts/prepare_more_data.py - rebuild rather than edit.\n"
                                       + yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_ext_report(report), encoding="utf-8")
    return report


def render_ext_report(r: ExtReport) -> str:
    sources = sorted({src for s in SPLITS for src in r.images.get(s, {})})
    out = ["# Training dataset report: ppe4 (ppe3 + extra datasets)", "",
           f"Built from `{r.base}` (taken as it is, split for split) and the extra datasets below. "
           f"Classes: {', '.join(r.classes)}.", "", "## Sources", ""]
    for s in r.sources:
        out.append(f"- **{s['title']}** ({s['part']}): licence {s['licence']}. {s['attribution']}")
    out += ["", "## Checking the extra datasets' labels (data/audit.py)", "",
            "| Source | Images read | Kept | People added by the COCO model | Left out, and why |", "|---|---|---|---|---|"]
    for name, a in r.audit.items():
        why = "; ".join(f"{n}: {reason}" for reason, n in a["dropped"].items()) or "none"
        out.append(f"| {name} | {a['read']} | {a['kept']} | {a['people_added']} | {why} |")
    out += ["", "## Keeping the tests honest", "",
            f"- {r.near_duplicate_pairs} near-duplicate pairs in all.",
            f"- Extra images left out because they look like a ppe3 val or test image: "
            f"{r.left_out_near_ppe3_test or 'none'}.",
            f"- Extra val/test images moved into train (a near-duplicate was in train): {r.moved_to_train or 'none'}.",
            f"- The same photo in two extra datasets, kept once: {r.exact_duplicates_dropped}.", "",
            "## Result", "", "Images per split and source:", "",
            "| Split | " + " | ".join(sources) + " | total |", "|---|" + "---|" * (len(sources) + 1)]
    for s in SPLITS:
        row = r.images.get(s, {})
        out.append(f"| {s} | " + " | ".join(str(row.get(src, 0)) for src in sources) + f" | {sum(row.values())} |")
    out += ["", "Boxes per split:", "", "| Split | " + " | ".join(r.classes) + " |", "|---|" + "---|" * len(r.classes)]
    for s in SPLITS:
        out.append(f"| {s} | " + " | ".join(str(r.boxes[s][c]) for c in r.classes) + " |")
    if r.helmet_colours_test:
        out += ["", "Labelled helmet colours in the test split (for the colour check in compare_models.py):", "",
                "| Source | " + " | ".join(COLOURS) + " |", "|---|" + "---|" * len(COLOURS)]
        for src, c in r.helmet_colours_test.items():
            out.append(f"| {src} | " + " | ".join(str(c.get(k, 0)) for k in COLOURS) + " |")
    return "\n".join(out) + "\n"
